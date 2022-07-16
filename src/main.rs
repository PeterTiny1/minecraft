mod camera;
mod chunk;
mod physics;
pub mod ray;
mod texture;

use itertools::Itertools;
use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
    video::{FullscreenType, Window},
};
use std::{
    collections::{HashMap, VecDeque},
    env,
    fs::File,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    thread, vec,
};

use chunk::{
    generate_chunk_mesh, BlockType, ChunkData, Rotation, CHUNK_DEPTH, CHUNK_HEIGHT, CHUNK_WIDTH,
};
use futures::executor::block_on;

use vek::Mat4;
use wgpu::util::DeviceExt;

use noise::{NoiseFn, OpenSimplex};

const MAX_DEPTH: f32 = 256.0;

#[derive(Debug)]
struct ChunkBuffers {
    index: wgpu::Buffer,
    vertex: wgpu::Buffer,
    num_indices: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex([f32; 3], [f32; 2], f32);

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Mat4::<f32>::identity().into_col_arrays(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into_col_arrays();
    }
}

const LARGE_SCALE: f64 = 50.0;
const SMALL_SCALE: f64 = 20.0;
const LARGE_HEIGHT: f64 = 40.0;
const TERRAIN_HEIGHT: f64 = 0.7;

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: (u32, u32),
    render_pipeline: wgpu::RenderPipeline,
    diffuse_bind_group: wgpu::BindGroup,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,
    // right_pressed: bool,
    depth_texture: texture::Texture,
    generated_chunkdata: Arc<Mutex<HashMap<[i32; 2], chunk::ChunkData>>>,
    generated_chunk_buffers: HashMap<[i32; 2], ChunkBuffers>,
    generating_chunks: Arc<Mutex<VecDeque<[i32; 2]>>>,
    returned_buffers: Arc<Mutex<Vec<(Vec<Vertex>, Vec<u32>, [i32; 2])>>>,
    noise: noise::OpenSimplex,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
            unclipped_depth: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.size();
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.0,
            height: size.1,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });
        let diffuse_bytes = include_bytes!("atlas.png");
        let diffuse_texture =
            texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "atlas.png").unwrap();
        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
            ],
            label: Some("diffuse_bind_group"),
        });
        let camera = camera::Camera::new(
            (0.0, 64.5, 8.0),
            -45.0_f32.to_radians(),
            -20.0_f32.to_radians(),
        );
        let projection = camera::Projection::new(
            config.width,
            config.height,
            90.0_f32.to_radians(),
            0.1,
            MAX_DEPTH,
        );
        let camera_controller = camera::CameraController::new(4.0, 0.05);
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera, &projection);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[Vertex::desc()],
                shader,
            )
        };
        let noise = OpenSimplex::new();
        let path = Path::new("0,0.bin");
        let chunk = if path.exists() {
            let buffer = std::fs::read(path).unwrap();
            bincode::deserialize::<ChunkData>(&buffer).unwrap().contents
        } else {
            let heightmap: Vec<Vec<i32>> = (0..CHUNK_WIDTH)
                .map(|x| {
                    (0..CHUNK_DEPTH)
                        .map(|y| {
                            ((noise.get([x as f64 / LARGE_SCALE, y as f64 / LARGE_SCALE])
                                + TERRAIN_HEIGHT)
                                * LARGE_HEIGHT
                                + (noise.get([
                                    x as f64 / SMALL_SCALE + 10.0,
                                    y as f64 / SMALL_SCALE + 10.0,
                                ]) * 20.0)) as i32
                        })
                        .collect()
                })
                .collect();
            // Generate chunk:
            let mut chunk = [[[BlockType::Air; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];
            for x in 0..CHUNK_WIDTH {
                for y in 0..CHUNK_HEIGHT {
                    for z in 0..CHUNK_DEPTH {
                        let y_i32 = y as i32;
                        chunk[x][y][z] = if y_i32 < heightmap[x][z] {
                            BlockType::Stone
                        } else if y_i32 == heightmap[x][z] {
                            BlockType::GrassBlock
                        } else if y_i32 > heightmap[x][z] && y_i32 <= heightmap[x][z] + 5 {
                            if noise.get([x as f64, heightmap[x][z] as f64, z as f64]) > 0.4 {
                                if y_i32 == heightmap[x][z] + 5 {
                                    BlockType::Leaf
                                } else {
                                    BlockType::Wood(Rotation::Up)
                                }
                            } else if y_i32 == heightmap[x][z] + 1
                                && noise.get([x as f64 / 4.0, z as f64 / 4.0, y as f64 / 4.0]) > 0.3
                            {
                                BlockType::Grass
                            } else {
                                BlockType::Air
                            }
                        } else {
                            BlockType::Air
                        };
                    }
                }
            }
            chunk
        };
        let (mesh, chunk_indices) = generate_chunk_mesh([0; 2], chunk, [None; 4]);
        let generated_chunkdata = Arc::new(Mutex::new(HashMap::from([(
            [0; 2],
            chunk::ChunkData { contents: chunk },
        )])));
        let generated_chunk_buffers = HashMap::from([(
            [0_i32; 2],
            ChunkBuffers {
                vertex: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&mesh),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                index: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&chunk_indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                num_indices: chunk_indices.len() as u32,
            },
        )]);
        let generating_chunks: Arc<std::sync::Mutex<VecDeque<[i32; 2]>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let returned_buffers = Arc::new(Mutex::new(vec![]));
        let thread_arc = Arc::clone(&generating_chunks);
        let chunkdata_arc = Arc::clone(&generated_chunkdata);
        let returning_arc = Arc::clone(&returned_buffers);
        // Chunk generator thread
        thread::spawn(move || loop {
            if let Some(chunk_location) = thread_arc.lock().unwrap().pop_front() {
                let generated_chunkdata = chunkdata_arc.lock().unwrap();
                let chunk_contents = generated_chunkdata[&chunk_location].contents;
                let [x, y] = chunk_location;
                let chunk_locations = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]];
                let (mesh, index_buffer) = generate_chunk_mesh(
                    chunk_location,
                    chunk_contents,
                    chunk_locations.map(|chunk| generated_chunkdata.get(&chunk)),
                );
                let further_chunks = [
                    [[x + 2, y], [x + 1, y + 1], [x + 1, y - 1]],
                    [[x - 2, y], [x - 1, y + 1], [x - 1, y - 1]],
                    [[x + 1, y + 1], [x - 1, y + 1], [x, y + 2]],
                    [[x + 1, y - 1], [x - 1, y - 1], [x, y - 2]],
                ];
                let new_chunkdata = chunk::ChunkData {
                    contents: chunk_contents,
                };
                for (index, (chunk_index, surrounding_chunks)) in
                    chunk_locations.iter().zip(further_chunks).enumerate()
                {
                    let get_chunk = |a, b| {
                        if index == a {
                            Some(&new_chunkdata)
                        } else {
                            generated_chunkdata.get(&surrounding_chunks[b])
                        }
                    };
                    if let Some(chunk) = generated_chunkdata.get(chunk_index) {
                        let (mesh, indices) = generate_chunk_mesh(
                            chunk_locations[index],
                            chunk.contents,
                            [
                                get_chunk(1, 0),
                                get_chunk(0, if index == 1 { 0 } else { 1 }),
                                get_chunk(3, if index < 2 { 1 } else { 2 }),
                                get_chunk(2, 2),
                            ],
                        );
                        returning_arc
                            .lock()
                            .unwrap()
                            .push((mesh, indices, *chunk_index))
                    }
                }
                returning_arc
                    .lock()
                    .unwrap()
                    .push((mesh, index_buffer, chunk_location));
            }
        });
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            mouse_pressed: false,
            // right_pressed: false,
            camera,
            projection,
            camera_controller,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            generated_chunkdata,
            generated_chunk_buffers,
            generating_chunks,
            returned_buffers,
            noise,
        }
    }

    fn resize(&mut self, new_size: (u32, u32)) {
        if new_size.1 > 0 {
            self.projection.resize(new_size.0, new_size.1);
            self.size = new_size;
            self.config.width = new_size.0;
            self.config.height = new_size.1;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn keydown(&mut self, keycode: Keycode, window_focused: bool) {
        if window_focused {
            self.camera_controller.process_keyboard(keycode, true);
        }
    }

    fn keyup(&mut self, keycode: Keycode, window_focused: bool) {
        if window_focused {
            self.camera_controller.process_keyboard(keycode, false);
        }
    }

    fn mouse_motion(&mut self, xrel: i32, yrel: i32) {
        self.camera_controller
            .process_mouse(xrel.into(), yrel.into());
    }

    fn mouse_scroll(&mut self, delta: i32) {
        self.camera_controller.process_scroll(delta);
    }

    fn update(&mut self, dt: std::time::Duration) {
        let mut generated_chunkdata = self.generated_chunkdata.lock().unwrap();
        self.camera_controller
            .update_camera(&mut self.camera, dt, &generated_chunkdata);
        self.uniforms
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        let chunk_location = chunk::get_nearest_chunk_location(
            self.camera.position.x,
            self.camera.position.z,
            &generated_chunkdata,
        );
        if let Some(chunk_location) = chunk_location {
            if !generated_chunkdata.contains_key(&chunk_location) {
                let location = &format!("{}.bin", chunk_location.iter().join(","));
                let path = Path::new(location);
                let chunk_contents = if path.exists() {
                    let buffer = std::fs::read(path).unwrap();
                    bincode::deserialize::<ChunkData>(&buffer).unwrap().contents
                } else {
                    let heightmap: Vec<Vec<i32>> = (0..CHUNK_WIDTH)
                        .map(|x| {
                            (0..CHUNK_DEPTH)
                                .map(|z| {
                                    (((chunk::noise_at(
                                        &self.noise,
                                        x as i32,
                                        z as i32,
                                        chunk_location,
                                        LARGE_SCALE,
                                        0.0,
                                    ) + TERRAIN_HEIGHT)
                                        * LARGE_HEIGHT)
                                        + (chunk::noise_at(
                                            &self.noise,
                                            x as i32,
                                            z as i32,
                                            chunk_location,
                                            SMALL_SCALE,
                                            10.0,
                                        ) * 20.0)) as i32
                                })
                                .collect::<Vec<i32>>()
                        })
                        .collect();
                    let mut chunk_contents =
                        [[[BlockType::Air; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];
                    for x in 0..CHUNK_WIDTH {
                        for y in 0..CHUNK_HEIGHT {
                            for z in 0..CHUNK_DEPTH {
                                let y_i32 = y as i32;
                                chunk_contents[x][y][z] = if y_i32 < heightmap[x][z] {
                                    BlockType::Stone
                                } else if y_i32 == heightmap[x][z] {
                                    BlockType::GrassBlock
                                } else if y_i32 > heightmap[x][z] && y_i32 <= heightmap[x][z] + 5 {
                                    if self.noise.get([x as f64, heightmap[x][z] as f64, z as f64])
                                        > 0.4
                                    {
                                        if y_i32 == heightmap[x][z] + 5 {
                                            BlockType::Leaf
                                        } else {
                                            BlockType::Wood(Rotation::Up)
                                        }
                                    } else if y_i32 == heightmap[x][z] + 1
                                        && self.noise.get([
                                            x as f64 / 4.0,
                                            z as f64 / 4.0,
                                            y as f64 / 4.0,
                                        ]) > 0.3
                                    {
                                        BlockType::Grass
                                    } else {
                                        BlockType::Air
                                    }
                                } else {
                                    BlockType::Air
                                };
                            }
                        }
                    }
                    chunk_contents
                };
                generated_chunkdata.insert(
                    chunk_location,
                    ChunkData {
                        contents: chunk_contents,
                    },
                );
                self.generating_chunks
                    .lock()
                    .unwrap()
                    .push_back(chunk_location);
            }
        }
        if let Some(location) = self.camera_controller.looking_at_block {
            let chunk_x = location.x.div_euclid(CHUNK_WIDTH as i32);
            let chunk_z = location.z.div_euclid(CHUNK_DEPTH as i32);
            generated_chunkdata
                .get_mut(&[chunk_x, chunk_z])
                .unwrap()
                .contents[location.x as usize % CHUNK_WIDTH][location.y as usize]
                [location.z as usize % CHUNK_DEPTH] = BlockType::Air;

            self.generating_chunks
                .lock()
                .unwrap()
                .push_back([chunk_x, chunk_z]);
        }
        for (mesh, indices, index) in self.returned_buffers.lock().unwrap().drain(..) {
            self.generated_chunk_buffers.insert(
                index,
                ChunkBuffers {
                    vertex: self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        }),
                    index: self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        }),
                    num_indices: indices.len() as u32,
                },
            );
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.3,
                            b: 0.4,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            for chunk in self.generated_chunk_buffers.values() {
                render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
                render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
    let mut save = true;
    let mut args = env::args();
    let _path = args.next().unwrap();
    if let Some(arg) = args.next() {
        match &*arg {
            "-no-save" => save = false,
            _ => println!("Invalid argument {arg}!"),
        }
    }
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let mut window = video_subsystem
        .window("My Minecraft Clone", 720, 480)
        .position_centered()
        .build()
        .unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    window.set_grab(true);
    sdl_context.mouse().show_cursor(false);
    sdl_context.mouse().set_relative_mouse_mode(true);
    window
        .set_fullscreen(sdl2::video::FullscreenType::Desktop)
        .expect("Failed to make the window full screen");
    let mut state = block_on(State::new(&window));
    let mut last_render_time = std::time::Instant::now();
    let mut window_focused = true;
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => match keycode {
                    Keycode::F11 => {
                        if window.fullscreen_state() == FullscreenType::Desktop {
                            window
                                .set_fullscreen(FullscreenType::Off)
                                .expect("Failed to leave fullscreen");
                        } else {
                            window
                                .set_fullscreen(FullscreenType::Desktop)
                                .expect("Failed to make the window fullscreen");
                        }
                    }
                    Keycode::Escape => {
                        break 'running;
                    }
                    _ => {
                        state.keydown(keycode, window_focused);
                    }
                },
                Event::KeyUp {
                    keycode: Some(keycode),
                    ..
                } => {
                    state.keyup(keycode, window_focused);
                }
                Event::MouseMotion {
                    xrel,
                    yrel,
                    window_id,
                    ..
                } if window_id == window.id() => state.mouse_motion(xrel, yrel),
                Event::MouseWheel { y, .. } => state.mouse_scroll(y),
                Event::Window {
                    ref win_event,
                    window_id,
                    ..
                } if window_id == window.id() => match win_event {
                    WindowEvent::Close => break 'running,
                    WindowEvent::FocusGained => window_focused = true,
                    WindowEvent::FocusLost => window_focused = false,
                    WindowEvent::Resized(width, height) => {
                        state.resize((*width as u32, *height as u32))
                    }
                    _ => {}
                },
                Event::MouseButtonDown { .. } => {
                    state.mouse_pressed = true;
                }
                Event::MouseButtonUp { .. } => {
                    state.mouse_pressed = false;
                }
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }
        let now = std::time::Instant::now();
        let dt = now - last_render_time;
        last_render_time = now;
        state.update(dt);
        match state.render() {
            Ok(_) => {}
            Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
            Err(wgpu::SurfaceError::OutOfMemory) => break 'running,
            Err(e) => eprintln!("{:?}", e),
        }
    }
    if save {
        for (location, data) in state.generated_chunkdata.lock().unwrap().iter() {
            let location = format!("{}.bin", location.iter().join(","));
            let path = Path::new(&location);
            if let Ok(mut file) = File::create(path) {
                file.write_all(&bincode::serialize(data).unwrap()).unwrap();
            }
        }
    }
}
