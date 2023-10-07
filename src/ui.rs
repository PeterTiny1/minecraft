use wgpu::util::DeviceExt;

use crate::{create_index_buffer, create_render_pipeline, load_texture, texture};

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

#[inline]
pub fn init_state(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    config: &wgpu::SurfaceConfiguration,
    size: (u32, u32),
) -> State {
    let crosshair_bind_group = load_texture(
        device,
        texture_bind_group_layout,
        &texture::Texture::from_bytes(
            device,
            queue,
            include_bytes!("crosshair.png"),
            "crosshair.png",
        )
        .unwrap(),
        Some("crosshair_bind_group"),
    );
    let crosshair = (
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&CROSSHAIR),
            usage: wgpu::BufferUsages::VERTEX,
        }),
        create_index_buffer(device, &[0, 1, 2, 0, 2, 3]),
    );
    State {
        pipeline: create_render_pipeline(
            device,
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    }),
                ],
                push_constant_ranges: &[],
            }),
            config.format,
            Some(texture::Texture::DEPTH_FORMAT),
            &[Vertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
            },
        ),
        uniform: Uniform {
            aspect: size.0 as f32 / size.1 as f32,
        },
        crosshair,
        crosshair_bind_group,
        uniform_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            }),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Uniform Buffer"),
                        contents: bytemuck::cast_slice(&[Uniform {
                            aspect: size.0 as f32 / size.1 as f32,
                        }]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    })
                    .as_entire_binding(),
            }],
            label: Some("ui_uniform_bind_group"),
        }),
        uniform_buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniform {
                aspect: size.0 as f32 / size.1 as f32,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }),
    }
}
