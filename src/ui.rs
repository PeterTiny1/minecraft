use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{create_index_buffer, create_render_pipeline, load_texture, texture, RenderContext};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
    pub aspect: f32,
}
pub struct State {
    pub pipeline: wgpu::RenderPipeline,
    pub crosshair: (wgpu::Buffer, wgpu::Buffer),
    pub crosshair_bind_group: wgpu::BindGroup,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform: Uniform,
    pub uniform_buffer: wgpu::Buffer,
}

// location, uv
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

#[allow(clippy::cast_precision_loss)]
#[inline]
pub fn init_state(render_context: &RenderContext, size: PhysicalSize<u32>) -> State {
    let crosshair_bind_group = load_texture(
        &render_context.device,
        &render_context.texture_bind_group_layout,
        &texture::Texture::from_bytes(
            &render_context.device,
            &render_context.queue,
            include_bytes!("crosshair.png"),
            "crosshair.png",
        )
        .unwrap(),
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
    let aspect = size.width as f32 / size.height as f32;
    State {
        pipeline: create_render_pipeline(
            &render_context.device,
            &render_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[
                        &render_context.texture_bind_group_layout,
                        &render_context.device.create_bind_group_layout(
                            &wgpu::BindGroupLayoutDescriptor {
                                entries: &[wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::VERTEX
                                        | wgpu::ShaderStages::FRAGMENT,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                }],
                                label: Some("ui_uniform_bind_group_layout"),
                            },
                        ),
                    ],
                    push_constant_ranges: &[],
                }),
            render_context.config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Vertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
            },
        ),
        uniform: Uniform { aspect },
        crosshair,
        crosshair_bind_group,
        uniform_bind_group: render_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &render_context.device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
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
                    },
                ),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_context
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Uniform Buffer"),
                            contents: bytemuck::cast_slice(&[Uniform { aspect }]),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        })
                        .as_entire_binding(),
                }],
                label: Some("ui_uniform_bind_group"),
            }),
        uniform_buffer: render_context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[Uniform { aspect }]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ),
    }
}
