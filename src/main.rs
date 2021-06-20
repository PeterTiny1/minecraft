mod camera;
mod chunk;
mod noise;
mod texture;

use std::{convert::TryInto, f64::consts::PI, vec};

use chunk::generate_chunk_mesh;
use futures::executor::block_on;

// use rand::{thread_rng, Rng};
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window},
};

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
            step_mode: wgpu::InputStepMode::Vertex,
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
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    diffuse_bind_group: wgpu::BindGroup,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    mouse_pressed: bool,
    depth_texture: texture::Texture,
    vertices: Vec<Vertex>,
    generated_chunks: Vec<chunk::Chunk>,
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
                write_mask: wgpu::ColorWrite::ALL,
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
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
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
        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            format: adapter.get_swap_chain_preferred_format(&surface).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
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
        let camera = camera::Camera::new((0.0, 51.5, 8.0), cgmath::Deg(-45.0), cgmath::Deg(-20.0));
        let projection =
            camera::Projection::new(sc_desc.width, sc_desc.height, cgmath::Deg(90.0), 0.1, 256.0);
        let camera_controller = camera::CameraController::new(8.0, 0.2);
        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera, &projection);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX | wgpu::ShaderStage::FRAGMENT,
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
            texture::Texture::create_depth_texture(&device, &sc_desc, "depth_texture");
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
                flags: wgpu::ShaderFlags::all(),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                sc_desc.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[Vertex::desc()],
                shader,
            )
        };
        let mut vertices = vec![];
        let heightmap: Vec<Vec<i32>> = (0..16)
            .map(|x| {
                (0..16)
                    .map(|z| {
                        ((noise::noise(x as f64 / 180.0, z as f64 / 180.0) + PI) * 16.0
                            + (noise::noise(x as f64 / 50.0 + 10.0, z as f64 / 50.0 + 10.0) * 5.0))
                            as i32
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
        let mut generated_chunks = vec![chunk::Chunk {
            location: [0, 0],
            contents: chunk,
            mesh: generate_chunk_mesh([0, 0], chunk, None, None, None, None),
        }];
        for chunk in &mut generated_chunks {
            vertices.extend(&mut chunk.mesh.iter());
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsage::VERTEX,
        });
        let num_vertices = vertices.len() as u32;
        Self {
            surface,
            device,
            queue,
            sc_desc,
            swap_chain,
            size,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            diffuse_bind_group,
            mouse_pressed: false,
            camera,
            projection,
            camera_controller,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            vertices,
            generated_chunks,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.sc_desc.width = new_size.width;
            self.sc_desc.height = new_size.height;
            self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
            self.depth_texture = texture::Texture::create_depth_texture(
                &self.device,
                &self.sc_desc,
                "depth_texture",
            );
        }
    }

    fn input(&mut self, event: &DeviceEvent, window_focused: bool) -> bool {
        if window_focused {
            match event {
                DeviceEvent::Key(KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                }) => self.camera_controller.process_keyboard(*key, *state),
                DeviceEvent::MouseWheel { delta, .. } => {
                    self.camera_controller.process_scroll(&*delta);
                    true
                }
                DeviceEvent::Button {
                    button: 1, // Left Mouse Button
                    state,
                } => {
                    self.mouse_pressed = *state == ElementState::Pressed;
                    true
                }
                DeviceEvent::MouseMotion { delta } => {
                    self.camera_controller.process_mouse(delta.0, delta.1);
                    true
                }
                _ => false,
            };
        } else {
            self.camera_controller.release_all();
            self.mouse_pressed = false;
        }
        false
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
                            ((noise::noise(
                                (x + chunk_location[0] * 16) as f64 / 180.0,
                                (z + chunk_location[1] * 16) as f64 / 180.0,
                            ) + PI)
                                * 16.0
                                + noise::noise(
                                    (x + chunk_location[0] * 16) as f64 / 50.0 + 10.0,
                                    (z + chunk_location[1] * 16) as f64 / 50.0 + 10.0,
                                ) * 5.0) as i32
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
                .enumerate()
                .find(|a| a.1.location == [chunk_location[0] + 1, chunk_location[1]])
                .map(|a| a.0);
            let south_chunk = self
                .generated_chunks
                .iter()
                .enumerate()
                .find(|a| a.1.location == [chunk_location[0] - 1, chunk_location[1]])
                .map(|a| a.0);
            let east_chunk = self
                .generated_chunks
                .iter()
                .enumerate()
                .find(|a| a.1.location == [chunk_location[0], chunk_location[1] + 1])
                .map(|a| a.0);
            let west_chunk = self
                .generated_chunks
                .iter()
                .enumerate()
                .find(|a| a.1.location == [chunk_location[0], chunk_location[1] - 1])
                .map(|a| a.0);
            let new_chunk = chunk::Chunk {
                location: chunk_location,
                contents: chunk_contents,
                mesh: generate_chunk_mesh(
                    chunk_location,
                    chunk_contents,
                    north_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                    south_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                    east_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                    west_chunk.and_then(|chunk| self.generated_chunks.get(chunk)),
                ),
            };
            if north_chunk.is_some() {
                self.generated_chunks[north_chunk.unwrap()].mesh = generate_chunk_mesh(
                    [chunk_location[0] + 1, chunk_location[1]],
                    self.generated_chunks[north_chunk.unwrap()].contents,
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
                )
            }
            if south_chunk.is_some() {
                self.generated_chunks[south_chunk.unwrap()].mesh = generate_chunk_mesh(
                    [chunk_location[0] - 1, chunk_location[1]],
                    self.generated_chunks[south_chunk.unwrap()].contents,
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
                )
            }
            if east_chunk.is_some() {
                self.generated_chunks[east_chunk.unwrap()].mesh = generate_chunk_mesh(
                    [chunk_location[0], chunk_location[1] + 1],
                    self.generated_chunks[east_chunk.unwrap()].contents,
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
                )
            }
            if west_chunk.is_some() {
                self.generated_chunks[west_chunk.unwrap()].mesh = generate_chunk_mesh(
                    [chunk_location[0], chunk_location[1] - 1],
                    self.generated_chunks[west_chunk.unwrap()].contents,
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
                )
            }
            self.generated_chunks.push(new_chunk);
            self.vertices = vec![];
            for chunk in &mut self.generated_chunks {
                self.vertices.extend(&mut chunk.mesh.iter());
            }
            self.num_vertices = self.vertices.len() as u32;
            self.vertex_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Vertex Buffer"),
                        contents: bytemuck::cast_slice(&self.vertices),
                        usage: wgpu::BufferUsage::VERTEX,
                    });
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SwapChainError> {
        let frame = self.swap_chain.get_current_frame()?.output;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &frame.view,
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
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("My Minecraft Clone")
        .build(&event_loop)
        .unwrap();
    window.set_cursor_grab(true).expect("Failed to grab");
    window.set_cursor_visible(false);
    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
    let mut state = block_on(State::new(&window));
    let mut last_render_time = std::time::Instant::now();
    let mut window_focused = true;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent { ref event, .. } => {
                state.input(event, window_focused);
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::Focused(focused) => window_focused = *focused,
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::F11),
                        ..
                    } => {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None)
                        } else {
                            window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                        }
                    }
                    _ => {}
                },
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    state.resize(**new_inner_size)
                }
                _ => {}
            },
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SwapChainError::Lost) => state.resize(state.size),
                    Err(wgpu::SwapChainError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    });
}
