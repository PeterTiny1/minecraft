mod camera;
mod chunk;
pub mod ray;
mod texture;
mod ui;

use half::f16;
use itertools::Itertools;
use std::{
    collections::{hash_map::Entry, HashMap},
    env,
    fs::File,
    io::Write,
    path::Path,
    sync::{mpsc, Arc, Mutex},
    time::Instant,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey, PhysicalKey},
    window::Window,
};

use chunk::{
    chunkcoord_to_aabb, generate, start_chunkgen, BlockType, ChunkData, Index, CHUNK_DEPTH_I32,
    CHUNK_WIDTH_I32,
};
use futures::executor::block_on;

use vek::{Aabb, Mat4, Vec3, Vec4};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};

use noise::OpenSimplex;

const MAX_DEPTH: f32 = 512.0;

pub fn cuboid_intersects_frustum(
    cuboid: &Aabb<f32>,
    camera_matrix: Mat4<f32>,
    projection_matrix: Mat4<f32>,
) -> bool {
    let transform_matrix = projection_matrix * camera_matrix;

    let vertices = [
        Vec4::new(cuboid.min.x, cuboid.min.y, cuboid.min.z, 1.0),
        Vec4::new(cuboid.min.x, cuboid.min.y, cuboid.max.z, 1.0),
        Vec4::new(cuboid.min.x, cuboid.max.y, cuboid.min.z, 1.0),
        Vec4::new(cuboid.min.x, cuboid.max.y, cuboid.max.z, 1.0),
        Vec4::new(cuboid.max.x, cuboid.min.y, cuboid.min.z, 1.0),
        Vec4::new(cuboid.max.x, cuboid.min.y, cuboid.max.z, 1.0),
        Vec4::new(cuboid.max.x, cuboid.max.y, cuboid.min.z, 1.0),
        Vec4::new(cuboid.max.x, cuboid.max.y, cuboid.max.z, 1.0),
    ];

    let vertices_clip: Vec<Vec4<f32>> = vertices.iter().map(|&v| transform_matrix * v).collect();

    let planes = [
        Vec4::new(1.0, 0.0, 0.0, 1.0),
        Vec4::new(-1.0, 0.0, 0.0, 1.0),
        Vec4::new(0.0, 1.0, 0.0, 1.0),
        Vec4::new(0.0, -1.0, 0.0, 1.0),
        Vec4::new(0.0, 0.0, 1.0, 1.0),
        Vec4::new(0.0, 0.0, -1.0, 1.0),
    ];

    for plane in &planes {
        let outside = vertices_clip.iter().all(|v| plane.dot(*v) < 0.0);
        if outside {
            return false;
        }
    }

    true
}

#[derive(Debug)]
struct ChunkBuffers {
    index: wgpu::Buffer,
    vertex: wgpu::Buffer,
    num_indices: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// location, uv, lightlevel
pub struct Vertex([f32; 3], [f16; 2], f32);

impl Vertex {
    const fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float16x2,
                },
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[u16; 2]>())
                        as wgpu::BufferAddress,
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

    fn update_view_proj(&mut self, camera: &camera::CameraData, projection: &camera::Projection) {
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into_col_arrays();
    }
}

struct ChunkGenComms {
    sender: mpsc::SyncSender<[i32; 2]>,
    receiver: mpsc::Receiver<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
}

#[derive(Default)]
struct InputState {
    left_pressed: bool,
    right_pressed: bool,
}

struct WindowDependent<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    diffuse_bind_group: wgpu::BindGroup,
    camera: camera::Camera,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    ui: ui::State,
    generated_chunk_buffers: HashMap<[i32; 2], ChunkBuffers>,
    depth_texture: texture::Texture,
    generated_chunkdata: Arc<Mutex<HashMap<[i32; 2], chunk::ChunkData>>>,
    chunkgen_comms: ChunkGenComms,
    input: InputState,
    last_break: Instant,
    noise: noise::OpenSimplex,
    window: &'static Window,
}

impl WindowDependent<'_> {
    fn new(window: &'static Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface: wgpu::Surface<'_> = instance.create_surface(window).unwrap();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .unwrap();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            desired_maximum_frame_latency: 2,
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
        let diffuse_bind_group = load_texture(
            &device,
            &texture_bind_group_layout,
            &texture::Texture::from_bytes_mip(
                &device,
                &queue,
                include_bytes!("atlas.png"),
                "atlas.png",
            )
            .unwrap(),
            Some("diffuse_bind_group"),
        );
        let camera = camera::CameraData::new(
            (0.0, 100.0, 0.0),
            -45.0_f32.to_radians(),
            -20.0_f32.to_radians(),
        );
        let projection = camera::Projection::new(
            config.width,
            config.height,
            90.0_f32.to_radians(),
            0.05,
            MAX_DEPTH,
        );
        let camera_controller = camera::Controller::new(4.0, 0.05);
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera, &projection);
        let camera = camera::Camera {
            data: camera,
            projection,
            controller: camera_controller,
        };
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
        let render_pipeline = create_render_pipeline(
            &device,
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
                push_constant_ranges: &[],
            }),
            config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Vertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            },
        );
        let ui = ui::init_state(&device, &queue, &texture_bind_group_layout, &config, size);
        let generated_chunk_buffers = HashMap::new();
        let (send_generate, recv_generate) = mpsc::sync_channel(10);
        let (send_chunk, recv_chunk) = mpsc::sync_channel(10);
        let chunkgen_comms = ChunkGenComms {
            sender: send_generate,
            receiver: recv_chunk,
        };
        let generated_chunkdata = Arc::new(Mutex::new(HashMap::new()));
        start_chunkgen(recv_generate, Arc::clone(&generated_chunkdata), send_chunk);
        let noise = OpenSimplex::new(SEED);
        WindowDependent {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            camera,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            ui,
            generated_chunk_buffers,
            depth_texture,
            generated_chunkdata,
            chunkgen_comms,
            input: InputState::default(),
            last_break: Instant::now(),
            noise,
            window,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.height > 0 {
            self.camera.resize(new_size.width, new_size.height);
            self.ui.uniform.aspect = new_size.width as f32 / new_size.height as f32;
            self.queue.write_buffer(
                &self.ui.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.ui.uniform]),
            );
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn refresh(&mut self) {
        self.resize(self.size);
    }

    fn keydown(&mut self, key: PhysicalKey) {
        self.camera.controller.process_keyboard(key, true);
    }

    fn keyup(&mut self, key: PhysicalKey) {
        self.camera.controller.process_keyboard(key, false);
    }

    fn mouse_motion(&mut self, (dx, dy): (f64, f64)) {
        self.camera.controller.process_mouse(dx, dy);
    }

    fn mouse_scroll(&mut self, delta: f32) {
        self.camera.controller.process_scroll(delta);
    }

    fn process_keyevent(&mut self, event: &KeyEvent) {
        match event.state {
            ElementState::Pressed => self.keydown(event.physical_key),
            ElementState::Released => self.keyup(event.physical_key),
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        let mut generated_chunkdata = self.generated_chunkdata.lock().unwrap();
        self.camera.update(dt, &generated_chunkdata);
        self.uniforms
            .update_view_proj(&self.camera.data, &self.camera.projection);
        let chunk_location = chunk::nearest_visible_unloaded(
            self.camera.get_position().x,
            self.camera.get_position().z,
            &generated_chunkdata,
            &self.camera,
        );
        if let Some(chunk_location) = chunk_location {
            if let Entry::Vacant(e) = generated_chunkdata.entry(chunk_location) {
                let location = &format!("{}.bin", chunk_location.iter().join(","));
                let path = Path::new(location);
                let chunk_contents = if path.exists() {
                    let buffer = std::fs::read(path).unwrap();
                    bincode::deserialize::<ChunkData>(&buffer).unwrap().contents
                } else {
                    generate(&self.noise, chunk_location)
                };
                e.insert(ChunkData {
                    contents: chunk_contents,
                });
                self.chunkgen_comms.sender.send(chunk_location).unwrap();
            }
        }
        if let Some((location, previous_step)) = self.camera.get_looking_at() {
            let now = Instant::now();
            if self.input.right_pressed {
                let location = location
                    + match previous_step {
                        0 => Vec3 { x: 1, y: 0, z: 0 },
                        1 => Vec3 { x: -1, y: 0, z: 0 },
                        2 => Vec3 { x: 0, y: 1, z: 0 },
                        3 => Vec3 { x: 0, y: -1, z: 0 },
                        4 => Vec3 { x: 0, y: 0, z: 1 },
                        5 => Vec3 { x: 0, y: 0, z: -1 },
                        _ => panic!("Invalid return"),
                    };
                if location.y < 256 && location.y > -1 {
                    let chunk_x = location.x.div_euclid(CHUNK_WIDTH_I32);
                    let chunk_z = location.z.div_euclid(CHUNK_DEPTH_I32);
                    let chunk = generated_chunkdata.get_mut(&[chunk_x, chunk_z]);
                    let local_x = location.x.rem_euclid(CHUNK_WIDTH_I32) as usize;
                    let local_z = location.z.rem_euclid(CHUNK_DEPTH_I32) as usize;
                    if let Some(chunk) = chunk {
                        if chunk.contents[local_x][location.y as usize][local_z] == BlockType::Air {
                            chunk.contents[local_x][location.y as usize][local_z] =
                                BlockType::Stone;
                            self.chunkgen_comms.sender.send([chunk_x, chunk_z]).unwrap();
                        }
                    }
                }
            } else if self.input.left_pressed && (now - self.last_break).as_millis() > 250 {
                self.last_break = Instant::now();
                let chunk_x = location.x.div_euclid(CHUNK_WIDTH_I32);
                let chunk_z = location.z.div_euclid(CHUNK_DEPTH_I32);
                let local_x = location.x.rem_euclid(CHUNK_WIDTH_I32) as usize;
                let local_z = location.z.rem_euclid(CHUNK_DEPTH_I32) as usize;
                generated_chunkdata
                    .get_mut(&[chunk_x, chunk_z])
                    .unwrap()
                    .contents[local_x][location.y as usize][local_z] = BlockType::Air;
                self.chunkgen_comms.sender.send([chunk_x, chunk_z]).unwrap();
            }
        }
        drop(generated_chunkdata);
        while let Ok((mesh, indices, index)) = self.chunkgen_comms.receiver.try_recv() {
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
            // let (vertsize, indexsize) = self
            //     .generated_chunk_buffers
            //     .iter()
            //     .fold((0, 0), |acc, (_, item)| {
            //         (acc.0 + item.vertex.size(), acc.1 + item.index.size())
            //     });

            // println!("Index space: {indexsize}");
            // println!("Vertex space: {vertsize}");
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
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.3,
                            b: 0.4,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            for chunk_loc in self
                .generated_chunk_buffers
                .keys()
                .sorted_by(|&x, &y| {
                    ((y[0] * CHUNK_WIDTH_I32 - self.camera.get_position().x as i32).pow(2)
                        + (y[1] * CHUNK_DEPTH_I32 - self.camera.get_position().z as i32).pow(2))
                    .cmp(
                        &((x[0] * CHUNK_WIDTH_I32 - self.camera.get_position().x as i32).pow(2)
                            + (x[1] * CHUNK_DEPTH_I32 + -self.camera.get_position().z as i32)
                                .pow(2)),
                    )
                })
                .filter(|c| {
                    cuboid_intersects_frustum(
                        &chunkcoord_to_aabb(**c),
                        self.camera.calc_matrix(),
                        self.camera.calc_projection_matrix(),
                    )
                })
            {
                let chunk = &self.generated_chunk_buffers[chunk_loc];
                render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
                render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
            }
            render_pass.set_pipeline(&self.ui.pipeline);
            render_pass.set_bind_group(0, &self.ui.crosshair_bind_group, &[]);
            render_pass.set_bind_group(1, &self.ui.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.ui.crosshair.0.slice(..));
            render_pass.set_index_buffer(self.ui.crosshair.1.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

struct State<'a> {
    last_render_time: Instant,
    save: bool,
    window_dependent: Option<WindowDependent<'a>>,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{shader:?}")),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: PipelineCompilationOptions::default(),
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

const SEED: u32 = 0;

impl<'a> State<'a> {
    fn new(save: bool) -> Self {
        Self {
            last_render_time: Instant::now(),
            save,
            window_dependent: None,
        }
    }
}

impl<'a> ApplicationHandler for State<'a> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = Box::leak(Box::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Blockcraft")
                        .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
                )
                .unwrap(),
        ));
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .unwrap();
        window.set_cursor_visible(false);
        self.window_dependent = Some(WindowDependent::new(window));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => {
                event_loop.exit();
                if self.save {
                    save_file(self);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.window_dependent
                    .as_mut()
                    .unwrap()
                    .process_keyevent(&event);
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                MouseScrollDelta::LineDelta(_, delta) => {
                    self.window_dependent.as_mut().unwrap().mouse_scroll(delta);
                }
                MouseScrollDelta::PixelDelta(_) => todo!(),
            },
            WindowEvent::MouseInput { state, button, .. } => match button {
                winit::event::MouseButton::Left => {
                    self.window_dependent.as_mut().unwrap().input.left_pressed = state.is_pressed();
                }
                winit::event::MouseButton::Right => {
                    self.window_dependent.as_mut().unwrap().input.right_pressed =
                        state.is_pressed();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                let current_time = std::time::Instant::now();
                let dt = current_time - self.last_render_time;
                let window_dependent = self.window_dependent.as_mut().unwrap();
                window_dependent.update(dt);
                self.last_render_time = current_time;
                match window_dependent.render() {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost) => window_dependent.refresh(),
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => eprintln!("{e:?}"),
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.window_dependent.as_mut().unwrap().mouse_motion(delta);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(window) = self.window_dependent.as_mut() {
            window.window.request_redraw();
        }
    }
}

fn load_texture(
    device: &wgpu::Device,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    texture: &texture::Texture,
    label: Option<&str>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
        label,
    })
}

fn create_index_buffer(device: &wgpu::Device, chunk_indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(chunk_indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}

fn main() -> Result<(), impl std::error::Error> {
    env_logger::init();
    let mut save = false;
    let mut args = env::args();
    let _path = args.next().unwrap();
    if let Some(arg) = args.next() {
        match &*arg {
            "-save" | "-s" => save = true,
            _ => println!("Invalid argument {arg}!"),
        }
    }
    let event_loop = EventLoop::new().unwrap();

    let mut state = State::new(save);

    event_loop.run_app(&mut state)
}

fn save_file(state: &State) {
    let generated_chunkdata = state
        .window_dependent
        .as_ref()
        .unwrap()
        .generated_chunkdata
        .lock()
        .unwrap()
        .clone();
    let iterator = generated_chunkdata.iter();
    for (location, data) in iterator {
        let location = format!("{}.bin", location.iter().join(","));
        let path = Path::new(&location);
        if let Ok(mut file) = File::create(path) {
            file.write_all(&bincode::serialize(data).unwrap()).unwrap();
        }
    }
}
