use std::collections::HashMap;
use wgpu::util::DeviceExt;

use crate::{
    camera::Camera, mesh::Completed, renderer::cuboid_intersects_frustum,
    world::math::chunkcoord_to_aabb,
};

/// Represents index and byte ranges for a draw pass inside a unified buffer.
#[derive(Debug, Default, Clone, Copy)]
pub struct SubMeshDraw {
    pub index_start: u32,
    pub index_count: u32,
    pub vertex_byte_offset: wgpu::BufferAddress,
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
        self.opaque.index_count == 0
            && self.cutout_nocull.index_count == 0
            && self.translucent.index_count == 0
    }
}

pub struct VisibleChunk<'a> {
    pub loc: [i32; 2],
    pub buffers: &'a ChunkBuffers,
    pub relative_offset: [f32; 3],
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

    /// Packs all sub-meshes (with mixed vertex types) into unified GPU buffers.
    pub fn insert_mesh(&mut self, device: &wgpu::Device, mesh: Completed) {
        let data = &mesh.data;

        // Check if chunk is completely empty
        if data.opaque_indices.is_empty()
            && data.cutout_nocull_indices.is_empty()
            && data.translucent_indices.is_empty()
        {
            self.chunks.remove(&mesh.loc);
            return;
        }

        let opaque_index_count = data.opaque_indices.len() as u32;
        let cutout_index_count = data.cutout_nocull_indices.len() as u32;
        let translucent_index_count = data.translucent_indices.len() as u32;

        // -------------------------------------------------------------
        // 1. Pack Heterogeneous Vertex Data into Raw Byte Stream
        // -------------------------------------------------------------
        let mut combined_vertex_bytes = Vec::new();

        // Opaque vertices (e.g. CompactVertex, 4 bytes)
        let opaque_byte_offset = 0;
        let opaque_bytes = bytemuck::cast_slice(&data.opaque_vertices);
        combined_vertex_bytes.extend_from_slice(opaque_bytes);

        // Cutout vertices (e.g. Standard Vertex, 24 bytes)
        let cutout_byte_offset = combined_vertex_bytes.len() as wgpu::BufferAddress;
        let cutout_bytes = bytemuck::cast_slice(&data.cutout_nocull_vertices);
        combined_vertex_bytes.extend_from_slice(cutout_bytes);

        // Translucent vertices (e.g. Standard Vertex, 24 bytes)
        let translucent_byte_offset = combined_vertex_bytes.len() as wgpu::BufferAddress;
        let translucent_bytes = bytemuck::cast_slice(&data.translucent_vertices);
        combined_vertex_bytes.extend_from_slice(translucent_bytes);

        // -------------------------------------------------------------
        // 2. Concatenate Index Data (Keep indices 0-based for each pass!)
        // -------------------------------------------------------------
        let total_indices =
            (opaque_index_count + cutout_index_count + translucent_index_count) as usize;
        let mut combined_indices = Vec::with_capacity(total_indices);

        combined_indices.extend_from_slice(&data.opaque_indices);
        combined_indices.extend_from_slice(&data.cutout_nocull_indices);
        combined_indices.extend_from_slice(&data.translucent_indices);

        // -------------------------------------------------------------
        // 3. Create GPU Buffers
        // -------------------------------------------------------------
        let label_prefix = format!("Chunk {},{}", mesh.loc[0], mesh.loc[1]);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label_prefix} Vertex Buffer")),
            contents: &combined_vertex_bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label_prefix} Index Buffer")),
            contents: bytemuck::cast_slice(&combined_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // -------------------------------------------------------------
        // 4. Construct SubMesh slice ranges
        // -------------------------------------------------------------
        let opaque_draw = SubMeshDraw {
            index_start: 0,
            index_count: opaque_index_count,
            vertex_byte_offset: opaque_byte_offset,
        };

        let cutout_nocull_draw = SubMeshDraw {
            index_start: opaque_index_count,
            index_count: cutout_index_count,
            vertex_byte_offset: cutout_byte_offset,
        };

        let translucent_draw = SubMeshDraw {
            index_start: opaque_index_count + cutout_index_count,
            index_count: translucent_index_count,
            vertex_byte_offset: translucent_byte_offset,
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

    pub fn cull_visible_chunks<'a>(&'a self, camera: &Camera) -> Vec<VisibleChunk<'a>> {
        self.chunks
            .iter()
            .filter_map(|(loc, buffers)| {
                if buffers.is_empty() {
                    return None;
                }

                if cuboid_intersects_frustum(&chunkcoord_to_aabb(*loc), camera) {
                    let rel_x = (loc[0] * 32) as f32;
                    let rel_y = 0.0;
                    let rel_z = (loc[1] * 32) as f32;

                    Some(VisibleChunk {
                        loc: *loc,
                        buffers,
                        relative_offset: [rel_x, rel_y, rel_z],
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Renders a specific pass (Opaque, Cutout, or Translucent) across pre-culled visible chunks.
    pub fn render_pass<'a>(
        visible_chunks: &'a [VisibleChunk<'a>],
        render_pass: &mut wgpu::RenderPass<'a>,
        get_draw: impl Fn(&'a ChunkBuffers) -> &SubMeshDraw,
    ) {
        for (i, chunk) in visible_chunks.iter().enumerate() {
            let draw = get_draw(chunk.buffers);
            if draw.index_count == 0 {
                continue;
            }

            // Slice vertex buffer to start precisely at this vertex type's byte offset!
            render_pass.set_vertex_buffer(
                0,
                chunk.buffers.vertex_buffer.slice(draw.vertex_byte_offset..),
            );

            render_pass.set_index_buffer(
                chunk.buffers.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );

            let end = draw.index_start + draw.index_count;
            let instance_idx = i as u32;

            // Note: base_vertex is 0 because vertex_buffer is sliced to the submesh start!
            render_pass.draw_indexed(draw.index_start..end, 0, instance_idx..instance_idx + 1);
        }
    }
}
