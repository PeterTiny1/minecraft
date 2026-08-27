use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera,
    mesh::CompletedMesh,
    renderer::cuboid_intersects_frustum,
    world::math::chunkcoord_to_aabb,
};

pub struct ChunkBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

#[derive(Default)]
pub struct ChunkRenderer {
    pub generated_buffers: HashMap<[i32; 2], ChunkBuffers>,
}

impl ChunkRenderer {
    pub fn new() -> Self {
        Self {
            generated_buffers: HashMap::new(),
        }
    }

    /// Uploads a completed mesh to GPU memory and stores its buffers.
    pub fn insert_mesh(&mut self, device: &wgpu::Device, mesh: CompletedMesh) {
        if mesh.indices.is_empty() {
            self.generated_buffers.remove(&mesh.loc);
            return;
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Chunk {},{} Vertex Buffer", mesh.loc[0], mesh.loc[1])),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("Chunk {},{} Index Buffer", mesh.loc[0], mesh.loc[1])),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        self.generated_buffers.insert(
            mesh.loc,
            ChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
            },
        );
    }

    /// Removes GPU buffers when a chunk is unloaded.
    pub fn remove_mesh(&mut self, loc: &[i32; 2]) {
        self.generated_buffers.remove(loc);
    }

    /// Draws all visible chunk meshes.
    pub fn render_chunks<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
    ) {
        for (loc, buffers) in &self.generated_buffers {
            if cuboid_intersects_frustum(&chunkcoord_to_aabb(*loc), camera) {
                render_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                render_pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..buffers.index_count, 0, 0..1);
            }
        }
    }
}
