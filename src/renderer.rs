use crate::{camera, chunk::ChunkManager, texture, ui};
use futures::executor::block_on;
use half::f16;
use vek::{Aabb, Mat4, Vec4};
use wgpu::{util::DeviceExt, PipelineCompilationOptions};
use winit::{dpi::PhysicalSize, window::Window};

#[must_use]
pub fn cuboid_intersects_frustum(cuboid: &Aabb<f32>, camera: &camera::Camera) -> bool {
    const PLANES: [Vec4<f32>; 6] = [
        Vec4::new(1.0, 0.0, 0.0, 1.0),  // Left   (x = -w)
        Vec4::new(-1.0, 0.0, 0.0, 1.0), // Right  (x =  w)
        Vec4::new(0.0, 1.0, 0.0, 1.0),  // Bottom (y = -w)
        Vec4::new(0.0, -1.0, 0.0, 1.0), // Top    (y =  w)
        Vec4::new(0.0, 0.0, 1.0, 0.0),  // Near   (z =  0)
        Vec4::new(0.0, 0.0, -1.0, 1.0), // Far    (z =  w)
    ];

    let transform_matrix = camera.get_transformation();

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

    let vertices_clip = vertices.map(|v| transform_matrix * v);

    for plane in PLANES {
        let mut all = true;
        for vert in vertices_clip {
            all &= plane.dot(vert) < 0.0;
        }
        if all {
            return false;
        }
    }

    true
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
/// location, uv, lightlevel
pub struct Vertex {
    pub position: [f32; 3],
    pub uv: [f16; 2],
    pub light_level: f32,
}

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
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<[f16; 2]>())
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
pub struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Mat4::<f32>::identity().into_col_arrays(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &camera::Camera) {
        self.view_proj = (camera.get_transformation()).into_col_arrays();
    }
}

pub fn create_render_pipeline(
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
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
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
        cache: None,
    })
}

pub struct RenderContext<'a> {
    surface: wgpu::Surface<'a>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    diffuse_bind_group: wgpu::BindGroup,
    pub uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
}

impl RenderContext<'_> {
    pub fn new(window: &'static Window, size: PhysicalSize<u32>) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface: wgpu::Surface<'_> = instance.create_surface(window).unwrap();
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .unwrap();
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::default(),
            required_limits: if cfg!(target_arch = "wasm32") {
                wgpu::Limits::downlevel_webgl2_defaults()
            } else {
                wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
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
        let uniforms = Uniforms::new();
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
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            diffuse_bind_group,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
            texture_bind_group_layout,
        }
    }
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }
    pub fn render(
        &self,
        chunk_manager: &ChunkManager,
        camera: &camera::Camera,
        ui: &ui::State,
    ) -> Result<(), wgpu::SurfaceError> {
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
                    depth_slice: None,
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
            chunk_manager.render_chunks(&mut render_pass, camera);
            render_pass.set_pipeline(&ui.pipeline);
            render_pass.set_bind_group(0, &ui.crosshair_bind_group, &[]);
            render_pass.set_bind_group(1, &ui.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, ui.crosshair.0.slice(..));
            render_pass.set_index_buffer(ui.crosshair.1.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
    pub fn write_uniforms(&self) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }
}

pub fn load_texture(
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

pub fn create_index_buffer(device: &wgpu::Device, chunk_indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(chunk_indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}
