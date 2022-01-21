mod camera;
mod chunk;
mod texture;

use sdl2::{
    event::{Event, WindowEvent},
    keyboard::Keycode,
    video::{FullscreenType, Window},
};
use std::{convert::TryInto, f64::consts::PI, vec};

use chunk::generate_chunk_mesh;
use futures::executor::block_on;

use vek::Mat4;
// use rand::{thread_rng, Rng};
use wgpu::util::DeviceExt;

use noise::{NoiseFn, OpenSimplex};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

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
    // mouse_pressed: bool,
    // right_pressed: bool,
    depth_texture: texture::Texture,
    generated_chunks: Vec<chunk::Chunk>,
    noise: OpenSimplex,
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
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "main",
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
            clamp_depth: false,
            conservative: false,
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
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
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
            (0.0, 51.5, 8.0),
            -45.0_f32.to_radians(),
            -20.0_f32.to_radians(),
        );
        let projection = camera::Projection::new(
            config.width,
            config.height,
            90.0_f32.to_radians(),
            0.1,
            256.0,
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
        let heightmap: Vec<Vec<i32>> = (0..16)
            .map(|x| {
                (0..16)
                    .map(|y| {
                        ((noise.get([x as f64 / LARGE_SCALE, y as f64 / LARGE_SCALE]) + PI) * 16.0
                            + (noise.get([
                                x as f64 / SMALL_SCALE + 10.0,
                                y as f64 / SMALL_SCALE + 10.0,
                            ]) * 5.0)) as i32
                    })
                    .collect()
            })
            .collect();
        // Generate chunk:
        let chunk: [[[u16; 16]; 256]; 16] = (0..16)
            .map(|x| {
                (0..256)
                    .map(|y| {
                        (0..16)
                            .map(|z| (y < heightmap[x][z]) as u16)
                            .collect::<Vec<u16>>()
                            .try_into()
                            .unwrap()
                    })
                    .collect::<Vec<[u16; 16]>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[[u16; 16]; 256]>>()
            .try_into()
            .unwrap();
        let (mesh, chunk_indices) = generate_chunk_mesh([0, 0], chunk, None, None, None, None);
        let generated_chunks = vec![chunk::Chunk {
            location: [0, 0],
            contents: chunk,
            vertex_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&mesh),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            index_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&chunk_indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            num_indicies: chunk_indices.len() as u32,
        }];
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            // mouse_pressed: false,
            // right_pressed: false,
            camera,
            projection,
            camera_controller,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            generated_chunks,
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
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.uniforms
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
        let chunk_location = [
            (self.camera.position.x / 16.0).floor() as i32,
            (self.camera.position.z / 16.0).floor() as i32,
        ];
        if !self
            .generated_chunks
            .iter()
            .any(|chunk| chunk.location == chunk_location)
        {
            let heightmap: Vec<Vec<i32>> = (0..16)
                .map(|x| {
                    (0..16)
                        .map(|z| {
                            (((self.noise_at(x, z, chunk_location, LARGE_SCALE, 0.0) + PI) * 16.0)
                                + (self.noise_at(x, z, chunk_location, SMALL_SCALE, 10.0) * 5.0))
                                as i32
                        })
                        .collect::<Vec<i32>>()
                })
                .collect();
            let chunk_contents: [[[u16; 16]; 256]; 16] = (0..16)
                .map(|x| {
                    (0..256)
                        .map(|y| {
                            (0..16)
                                .map(|z| (y < heightmap[x][z]) as u16)
                                .collect::<Vec<u16>>()
                                .try_into()
                                .unwrap()
                        })
                        .collect::<Vec<[u16; 16]>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<[[u16; 16]; 256]>>()
                .try_into()
                .unwrap();
            let north_chunk = self
                .generated_chunks
                .iter()
                .position(|a| a.location == [chunk_location[0] + 1, chunk_location[1]]);
            let south_chunk = self
                .generated_chunks
                .iter()
                .position(|a| a.location == [chunk_location[0] - 1, chunk_location[1]]);
            let east_chunk = self
                .generated_chunks
                .iter()
                .position(|a| a.location == [chunk_location[0], chunk_location[1] + 1]);
            let west_chunk = self
                .generated_chunks
                .iter()
                .position(|a| a.location == [chunk_location[0], chunk_location[1] - 1]);
            let (mesh, index_buffer) = generate_chunk_mesh(
                chunk_location,
                chunk_contents,
                north_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                south_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                east_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                west_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
            );
            let new_chunk = chunk::Chunk {
                location: chunk_location,
                contents: chunk_contents,
                vertex_buffer: self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(&mesh),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                index_buffer: self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Index Buffer"),
                        contents: bytemuck::cast_slice(&index_buffer),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                num_indicies: index_buffer.len() as u32,
            };
            if let Some(chunk) = north_chunk {
                let (mesh, indices) = generate_chunk_mesh(
                    [chunk_location[0] + 1, chunk_location[1]],
                    self.generated_chunks[chunk].contents,
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] + 2, chunk_location[1]]),
                    Some(&new_chunk),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] + 1, chunk_location[1] + 1]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] + 1, chunk_location[1] - 1]),
                );
                self.generated_chunks[chunk].vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                self.generated_chunks[chunk].index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                self.generated_chunks[chunk].num_indicies = indices.len() as u32;
            }
            if let Some(chunk) = south_chunk {
                let (mesh, indices) = generate_chunk_mesh(
                    [chunk_location[0] - 1, chunk_location[1]],
                    self.generated_chunks[chunk].contents,
                    Some(&new_chunk),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] - 2, chunk_location[1]]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] - 1, chunk_location[1] + 1]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] - 1, chunk_location[1] - 1]),
                );
                self.generated_chunks[chunk].vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                self.generated_chunks[chunk].index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                self.generated_chunks[chunk].num_indicies = indices.len() as u32;
            }
            if let Some(chunk) = east_chunk {
                let (mesh, indices) = generate_chunk_mesh(
                    [chunk_location[0], chunk_location[1] + 1],
                    self.generated_chunks[chunk].contents,
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] + 1, chunk_location[1] + 1]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] - 1, chunk_location[1] + 1]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0], chunk_location[1] + 2]),
                    Some(&new_chunk),
                );
                self.generated_chunks[chunk].vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                self.generated_chunks[chunk].index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                self.generated_chunks[chunk].num_indicies = indices.len() as u32;
            }
            if let Some(chunk) = west_chunk {
                let (mesh, indices) = generate_chunk_mesh(
                    [chunk_location[0], chunk_location[1] - 1],
                    self.generated_chunks[chunk].contents,
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] + 1, chunk_location[1] - 1]),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0] - 1, chunk_location[1] - 1]),
                    Some(&new_chunk),
                    self.generated_chunks
                        .iter()
                        .find(|a| a.location == [chunk_location[0], chunk_location[1] - 2]),
                );
                self.generated_chunks[chunk].vertex_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                self.generated_chunks[chunk].index_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        });
                self.generated_chunks[chunk].num_indicies = indices.len() as u32;
            }
            self.generated_chunks.push(new_chunk);
        }
    }

    fn noise_at(
        &mut self,
        x: i32,
        z: i32,
        chunk_location: [i32; 2],
        scale: f64,
        offset: f64,
    ) -> f64 {
        self.noise.get([
            (x + (chunk_location[0] * 16)) as f64 / scale + offset,
            (z + (chunk_location[1] * 16)) as f64 / scale + offset,
        ])
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
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
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
            for chunk in &self.generated_chunks {
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.num_indicies, 0, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

fn main() {
    env_logger::init();
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
                Event::KeyDown { keycode, .. } => {
                    if let Some(key) = keycode {
                        if key == Keycode::F11 {
                            if window.fullscreen_state() == FullscreenType::Desktop {
                                window
                                    .set_fullscreen(FullscreenType::Off)
                                    .expect("Failed to leave fullscreen");
                            } else {
                                window
                                    .set_fullscreen(FullscreenType::Desktop)
                                    .expect("Failed to make the window fullscreen");
                            }
                        } else if key == Keycode::Escape {
                            break 'running;
                        } else {
                            state.keydown(key, window_focused);
                        }
                    }
                }
                Event::KeyUp { keycode, .. } => {
                    if let Some(key) = keycode {
                        state.keyup(key, window_focused);
                    }
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
}
