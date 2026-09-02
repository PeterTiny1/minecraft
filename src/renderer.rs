use core::fmt;
use std::sync::Arc;

use crate::{camera, texture, ui, world::ChunkRenderer};
use vek::{Aabb, Mat4, Vec4};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

const BLOCK_TEXTURES: &[&[u8]] = &[
    include_bytes!("textures/stone.png"),
    include_bytes!("textures/dirt.png"),
    include_bytes!("textures/grass_top0.png"),
    include_bytes!("textures/grass_side0.png"),
    include_bytes!("textures/grass_top1.png"),
    include_bytes!("textures/grass_side1.png"),
    include_bytes!("textures/grass_top2.png"),
    include_bytes!("textures/grass_side2.png"),
    include_bytes!("textures/birch_top.png"),
    include_bytes!("textures/birch_side.png"),
    include_bytes!("textures/wood_top.png"),
    include_bytes!("textures/wood_side.png"),
    include_bytes!("textures/dark_wood_top.png"),
    include_bytes!("textures/dark_wood_side.png"),
    include_bytes!("textures/birch_leaves.png"),
    include_bytes!("textures/leaves.png"),
    include_bytes!("textures/dark_leaves.png"),
    include_bytes!("textures/grass0.png"),
    include_bytes!("textures/grass1.png"),
    include_bytes!("textures/grass2.png"),
    include_bytes!("textures/flower0.png"),
    include_bytes!("textures/flower1.png"),
    include_bytes!("textures/flower2.png"),
    include_bytes!("textures/sand.png"),
    include_bytes!("textures/water_top.png"),
    include_bytes!("textures/water_side.png"),
];
#[must_use]
pub fn cuboid_intersects_frustum(cuboid: &Aabb<f32>, camera: &camera::Camera) -> bool {
    let m = camera.get_transformation();

    // Extract rows directly from column-major matrix (m.cols[col][row])
    let r0 = Vec4::new(m.cols[0].x, m.cols[1].x, m.cols[2].x, m.cols[3].x);
    let r1 = Vec4::new(m.cols[0].y, m.cols[1].y, m.cols[2].y, m.cols[3].y);
    let r2 = Vec4::new(m.cols[0].z, m.cols[1].z, m.cols[2].z, m.cols[3].z);
    let r3 = Vec4::new(m.cols[0].w, m.cols[1].w, m.cols[2].w, m.cols[3].w);

    let planes = [
        r3 + r0, // Left
        r3 - r0, // Right
        r3 + r1, // Bottom
        r3 - r1, // Top
        r2,      // Near
        r3 - r2, // Far
    ];

    let center = (cuboid.min + cuboid.max) * 0.5;
    let extents = cuboid.max - center;

    for plane in planes {
        let radius =
            extents.x * plane.x.abs() + extents.y * plane.y.abs() + extents.z * plane.z.abs();

        let distance = center.x * plane.x + center.y * plane.y + center.z * plane.z + plane.w;

        if distance < -radius {
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
    pub data: [u8; 4],
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
                    format: wgpu::VertexFormat::Uint8x4,
                },
            ],
        }
    }
}

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    view_proj: Mat4<f32>,
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Mat4::<f32>::identity(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &camera::Camera) {
        self.view_proj = camera.get_transformation();
    }
}

pub struct PipelineConfig<'a> {
    pub label: &'a str,
    pub vs_entry: &'a str,
    pub fs_entry: &'a str,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_write_enabled: bool,
    pub blend: Option<wgpu::BlendState>,
}

#[must_use]
pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[Option<wgpu::VertexBufferLayout>],
    shader: &wgpu::ShaderModule,
    config: PipelineConfig,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(config.label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(config.vs_entry), // Custom vertex entry
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(config.fs_entry), // Custom fragment entry!
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: config.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: config.cull_mode,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
            unclipped_depth: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(config.depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    })
}

#[derive(Debug, Clone, Copy)]
pub enum SurfaceCapabilityKind {
    Format,
    PresentMode,
    AlphaMode,
}

impl fmt::Display for SurfaceCapabilityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format => write!(f, "texture format"),
            Self::PresentMode => write!(f, "present mode"),
            Self::AlphaMode => write!(f, "alpha mode"),
        }
    }
}

#[derive(Debug)]
pub enum RenderContextError {
    CreateSurface(wgpu::CreateSurfaceError),
    RequestAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    MissingCapability(SurfaceCapabilityKind),
    TextureLoad(image::ImageError),
}

impl fmt::Display for RenderContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateSurface(e) => write!(f, "Failed to create surface: {e}"),
            Self::RequestAdapter(e) => write!(f, "Failed to find suitable GPU adapter: {e}"),
            Self::RequestDevice(e) => write!(f, "Failed to request graphics device: {e}"),
            Self::MissingCapability(kind) => write!(f, "Surface supports no compatible {kind}!"),
            Self::TextureLoad(e) => write!(f, "Failed to load image: {e}"),
        }
    }
}

impl std::error::Error for RenderContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateSurface(e) => Some(e),
            Self::RequestAdapter(e) => Some(e),
            Self::RequestDevice(e) => Some(e),
            Self::TextureLoad(e) => Some(e),
            Self::MissingCapability(_) => None,
        }
    }
}

impl From<wgpu::CreateSurfaceError> for RenderContextError {
    fn from(e: wgpu::CreateSurfaceError) -> Self {
        Self::CreateSurface(e)
    }
}

impl From<wgpu::RequestDeviceError> for RenderContextError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(e)
    }
}

impl From<wgpu::RequestAdapterError> for RenderContextError {
    fn from(e: wgpu::RequestAdapterError) -> Self {
        Self::RequestAdapter(e)
    }
}

impl From<image::ImageError> for RenderContextError {
    fn from(e: image::ImageError) -> Self {
        Self::TextureLoad(e)
    }
}

#[derive(PartialEq, Eq)]
pub enum RenderOutcome {
    Success,
    NeedsResize,
}

pub struct RenderContext {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    opaque_pipeline: wgpu::RenderPipeline,
    cutout_nocull_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    diffuse_bind_group: wgpu::BindGroup,
    pub uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    depth_texture: texture::Texture,
}

impl RenderContext {
    pub async fn new(
        window: Arc<Window>,
        size: PhysicalSize<u32>,
    ) -> Result<Self, RenderContextError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                apply_limit_buckets: true,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::default(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            })
            .await?;
        let surface_caps = surface.get_capabilities(&adapter);
        let texture_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or(RenderContextError::MissingCapability(
                SurfaceCapabilityKind::Format,
            ))?;
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|&mode| mode == wgpu::PresentMode::Fifo)
            .or_else(|| surface_caps.present_modes.first().copied())
            .ok_or(RenderContextError::MissingCapability(
                SurfaceCapabilityKind::PresentMode,
            ))?;

        let alpha_mode = surface_caps.alpha_modes.first().copied().ok_or(
            RenderContextError::MissingCapability(SurfaceCapabilityKind::AlphaMode),
        )?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: texture_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let diffuse_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
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
                label: Some("diffuse_bind_group_layout"),
            });
        let texture_atlas =
            texture::Texture::from_bytes_mip_array(&device, &queue, BLOCK_TEXTURES, "atlas")?;
        let diffuse_bind_group = load_texture(
            &device,
            &diffuse_bind_group_layout,
            &texture_atlas,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Chunk Pipeline Layout"),
            bind_group_layouts: &[
                Some(&diffuse_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let opaque_pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Some(Vertex::desc())],
            &shader,
            PipelineConfig {
                label: "Opaque Pipeline",
                vs_entry: "vs_main",
                fs_entry: "fs_opaque",
                cull_mode: Some(wgpu::Face::Back),
                depth_write_enabled: true,
                blend: None,
            },
        );
        let cutout_nocull_pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Some(Vertex::desc())],
            &shader,
            PipelineConfig {
                label: "Cutout Double-Sided Pipeline",
                vs_entry: "vs_main",
                fs_entry: "fs_cutout",
                cull_mode: None, // No culling so plants are visible from both sides
                depth_write_enabled: true,
                blend: None,
            },
        );
        let translucent_pipeline = create_render_pipeline(
            &device,
            &pipeline_layout,
            config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Some(Vertex::desc())],
            &shader,
            PipelineConfig {
                label: "Translucent Pipeline",
                vs_entry: "vs_main",
                fs_entry: "fs_translucent",
                cull_mode: None,
                depth_write_enabled: false,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            },
        );
        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            opaque_pipeline,
            cutout_nocull_pipeline,
            translucent_pipeline,
            diffuse_bind_group,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            depth_texture,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth_texture =
            texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
    }

    /// Errors
    ///
    /// `self.surface.get_current_texture` fails
    pub fn render(
        &self,
        chunk_renderer: &ChunkRenderer,
        camera: &camera::Camera,
        ui: &ui::State,
    ) -> RenderOutcome {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return RenderOutcome::NeedsResize;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Frame cannot or should not be presented right now
                return RenderOutcome::Success;
            }
        };

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
                multiview_mask: None,
            });

            render_pass.set_pipeline(&self.opaque_pipeline);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
            chunk_renderer.render_pass(&mut render_pass, camera, |b| &b.opaque);
            render_pass.set_pipeline(&self.cutout_nocull_pipeline);
            chunk_renderer.render_pass(&mut render_pass, camera, |b| &b.cutout_nocull);
            render_pass.set_pipeline(&self.translucent_pipeline);
            chunk_renderer.render_pass(&mut render_pass, camera, |b| &b.translucent);

            render_pass.set_pipeline(&ui.pipeline);
            render_pass.set_bind_group(0, &ui.crosshair_bind_group, &[]);
            render_pass.set_bind_group(1, &ui.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, ui.crosshair.0.slice(..));
            render_pass.set_index_buffer(ui.crosshair.1.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(output);

        RenderOutcome::Success
    }
    pub fn write_uniforms(&self) {
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }
}

#[must_use]
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

#[must_use]
pub fn create_index_buffer(device: &wgpu::Device, chunk_indices: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(chunk_indices),
        usage: wgpu::BufferUsages::INDEX,
    })
}
