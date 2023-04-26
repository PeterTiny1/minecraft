#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiUniform {
    pub aspect: f32,
}
pub struct UiState {
    pub pipeline: wgpu::RenderPipeline,
    pub crosshair: (wgpu::Buffer, wgpu::Buffer),
    pub crosshair_bind_group: wgpu::BindGroup,
    pub uniform_bind_group: wgpu::BindGroup,
    pub uniform: UiUniform,
    pub uniform_buffer: wgpu::Buffer,
}

// location, uv
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex([f32; 2], [f32; 2]);

impl UiVertex {
    pub const fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
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

pub const CROSSHAIR: [UiVertex; 4] = [
    UiVertex([-0.03125, -0.03125], [0.0, 0.0]),
    UiVertex([0.03125, -0.03125], [1.0, 0.0]),
    UiVertex([0.03125, 0.03125], [1.0, 1.0]),
    UiVertex([-0.03125, 0.03125], [0.0, 1.0]),
];
