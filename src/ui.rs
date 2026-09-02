use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{
    renderer::{
        PipelineConfig, RenderContext, create_index_buffer, create_render_pipeline, load_texture,
    },
    texture,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
    pub aspect: f32,
    _padding: [f32; 3], // Padded to 16 bytes for WGSL uniform alignment rules
}

impl Uniform {
    pub fn from_size(size: PhysicalSize<u32>) -> Self {
        #[allow(clippy::cast_precision_loss)]
        let aspect = if size.height == 0 {
            1.0
        } else {
            size.width as f32 / size.height as f32
        };

        Self {
            aspect,
            _padding: [0.0; 3],
        }
    }
}

pub struct State {
    pub pipeline: wgpu::RenderPipeline,
    pub crosshair: (wgpu::Buffer, wgpu::Buffer),
    pub crosshair_bind_group: wgpu::BindGroup,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform: Uniform,
    pub uniform_buffer: wgpu::Buffer,
}

impl State {
    pub fn resize(&mut self, queue: &wgpu::Queue, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }

        self.uniform = Uniform::from_size(size);

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex([f32; 2], [f32; 2]);

impl Vertex {
    const fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const CROSSHAIR: [Vertex; 4] = [
    Vertex([-0.03125, -0.03125], [0.0, 0.0]),
    Vertex([0.03125, -0.03125], [1.0, 0.0]),
    Vertex([0.03125, 0.03125], [1.0, 1.0]),
    Vertex([-0.03125, 0.03125], [0.0, 1.0]),
];

/// # Panics
///
/// If crosshair.png cannot be loaded
pub fn init_state(render_context: &RenderContext, size: PhysicalSize<u32>) -> State {
    let ui_bind_group_layout =
        render_context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("ui_bind_group_layout"),
            });

    let uniform_bind_group_layout =
        render_context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some("ui_uniform_bind_group_layout"),
            });

    let crosshair_bind_group = load_texture(
        &render_context.device,
        &ui_bind_group_layout,
        &texture::Texture::from_bytes(
            &render_context.device,
            &render_context.queue,
            include_bytes!("textures/crosshair.png"),
            "crosshair.png",
        )
        .expect("failed to parse crosshair texture"),
        Some("crosshair_bind_group"),
    );

    let crosshair = (
        render_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&CROSSHAIR),
                usage: wgpu::BufferUsages::VERTEX,
            }),
        create_index_buffer(&render_context.device, &[0, 1, 2, 0, 2, 3]),
    );

    let uniform = Uniform::from_size(size);

    let uniform_buffer =
        render_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

    let uniform_bind_group = render_context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("ui_uniform_bind_group"),
        });

    // 1. Create the pipeline layout
    let ui_pipeline_layout =
        render_context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("UI Pipeline Layout"),
                bind_group_layouts: &[
                    Some(&ui_bind_group_layout),
                    Some(&uniform_bind_group_layout),
                ],
                immediate_size: 0,
            });

    // 2. Compile the UI shader module
    let ui_shader = render_context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
        });

    // 3. Create the UI pipeline using the updated helper
    let ui_pipeline = create_render_pipeline(
        &render_context.device,
        &ui_pipeline_layout,
        render_context.config.format,
        Some(texture::Texture::DEPTH_FORMAT),
        &[Some(Vertex::desc())],
        &ui_shader,
        PipelineConfig {
            label: "UI Render Pipeline",
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            cull_mode: None, // Disable culling so UI elements render regardless of winding order
            depth_write_enabled: false, // UI shouldn't overwrite the depth buffer over 3D world geometry
            blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Enable transparency for UI overlays
        },
    );

    State {
        pipeline: ui_pipeline,
        crosshair,
        crosshair_bind_group,
        uniform_bind_group,
        uniform,
        uniform_buffer,
    }
}
