use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera, mesh::Completed, renderer::cuboid_intersects_frustum,
    world::math::chunkcoord_to_aabb,
};

/// Represents index ranges for the different draw passes inside a chunk's unified buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct SubMeshDraw {
    pub start: u32,
    pub count: u32,
}

pub struct ChunkBuffers {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub opaque: SubMeshDraw,
    pub cutout_nocull: SubMeshDraw,
    pub translucent: SubMeshDraw,
}

impl ChunkBuffers {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.opaque.count == 0 && self.translucent.count == 0
    }
}

#[derive(Default)]
pub struct ChunkRenderer {
    pub chunks: HashMap<[i32; 2], ChunkBuffers>,
}

impl ChunkRenderer {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    /// Packs all sub-meshes into unified chunk vertex/index buffers.
    pub fn insert_mesh(&mut self, device: &wgpu::Device, mesh: Completed) {
        let data = &mesh.data;

        // Check if chunk is completely empty across all pipelines
        if data.opaque_indices.is_empty() && data.translucent_indices.is_empty() {
            self.chunks.remove(&mesh.loc);
            return;
        }

        // 1. Calculate offsets and merge index data
        let opaque_count = data.opaque_indices.len() as u32;
        let cutout_nocull_count = data.cutout_nocull_indices.len() as u32;
        let translucent_count = data.translucent_indices.len() as u32;

        let opaque_v_len = data.opaque_vertices.len() as u32;
        let cutout_nocull_v_len = data.cutout_nocull_vertices.iter().len() as u32;

        // Re-index cutout and translucent indices relative to the combined vertex buffer
        let mut combined_indices =
            Vec::with_capacity((opaque_count + cutout_nocull_count + translucent_count) as usize);

        combined_indices.extend_from_slice(&data.opaque_indices);

        for &i in &data.cutout_nocull_indices {
            combined_indices.push(i + opaque_v_len);
        }

        for &i in &data.translucent_indices {
            combined_indices.push(i + opaque_v_len + cutout_nocull_v_len);
        }

        // 2. Merge vertex data
        let total_vertices = data.opaque_vertices.len()
            + data.cutout_nocull_vertices.len()
            + data.translucent_vertices.len();

        let mut combined_vertices = Vec::with_capacity(total_vertices);
        combined_vertices.extend_from_slice(&data.opaque_vertices);
        combined_vertices.extend_from_slice(&data.cutout_nocull_vertices);
        combined_vertices.extend_from_slice(&data.translucent_vertices);

        // 3. Create single GPU Vertex and Index Buffer per Chunk
        let label_prefix = format!("Chunk {},{}", mesh.loc[0], mesh.loc[1]);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label_prefix} Vertex Buffer")),
            contents: bytemuck::cast_slice(&combined_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label_prefix} Index Buffer")),
            contents: bytemuck::cast_slice(&combined_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // 4. Construct SubMesh slice ranges
        let opaque_draw = SubMeshDraw {
            start: 0,
            count: opaque_count,
        };
        let cutout_nocull_draw = SubMeshDraw {
            start: opaque_count,
            count: cutout_nocull_count,
        };
        let translucent_draw = SubMeshDraw {
            start: opaque_count + cutout_nocull_count,
            count: translucent_count,
        };

        self.chunks.insert(
            mesh.loc,
            ChunkBuffers {
                vertex_buffer,
                index_buffer,
                opaque: opaque_draw,
                cutout_nocull: cutout_nocull_draw,
                translucent: translucent_draw,
            },
        );
    }

    pub fn remove_mesh(&mut self, loc: &[i32; 2]) {
        self.chunks.remove(loc);
    }

    /// Renders a specific pass (Opaque, Cutout, or Translucent) across all visible chunks.
    pub fn render_pass<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera: &Camera,
        get_draw: impl Fn(&'a ChunkBuffers) -> &SubMeshDraw,
    ) {
        for (loc, buffers) in &self.chunks {
            let draw = get_draw(buffers);
            if draw.count == 0 {
                continue;
            }

            if cuboid_intersects_frustum(&chunkcoord_to_aabb(*loc), camera) {
                render_pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                let end = draw.start + draw.count;
                render_pass.draw_indexed(draw.start..end, 0, 0..1);
            }
        }
    }
}
