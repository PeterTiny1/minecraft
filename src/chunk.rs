use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, mpsc},
};

use noise::OpenSimplex;
use rkyv::{Archive, Deserialize, Serialize, access, deserialize};
use vek::Vec3;
use wgpu::util::DeviceExt;

use crate::{
    SEED,
    block::BlockType,
    camera::{self, Camera},
    mesh::{self, CompletedMesh, LocatedChunk, MeshJob},
    renderer::{RenderContext, cuboid_intersects_frustum},
    world::{block_index, chunkcoord_to_aabb, nearest_visible_unloaded, world_to_chunk_pos},
    world_gen::generate,
};
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_HEIGHT_I32: i32 = CHUNK_HEIGHT as i32;
pub const CHUNK_DEPTH: usize = 32;
pub const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

pub const CHUNK_SIZE: usize = CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH;
pub type Chunk = Box<[BlockType; CHUNK_SIZE]>;
#[derive(Debug, Clone, Deserialize, Serialize, Archive)]
pub struct ChunkData {
    pub contents: Chunk,
}
pub type ChunkDataStorage = HashMap<[i32; 2], Arc<ChunkData>>;
const NEIGHBOUR_OFFSETS: [[i32; 2]; 8] = [
    [1, 0],   // 0: [x + 1, y]
    [1, 1],   // 1: [x + 1, y + 1]
    [0, 1],   // 2: [x, y + 1]
    [-1, 1],  // 3: [x - 1, y + 1]
    [-1, 0],  // 4: [x - 1, y]
    [-1, -1], // 5: [x - 1, y - 1]
    [0, -1],  // 6: [x, y - 1]
    [1, -1],  // 7: [x + 1, y - 1]
];

pub trait BlockProvider {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType>;
}

impl BlockProvider for ChunkDataStorage {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        if y < 0 || y as usize >= CHUNK_HEIGHT {
            return None;
        }
        let (chunk_loc, [local_x, local_y, local_z]) = world_to_chunk_pos(x, y, z);
        let chunk = self.get(&chunk_loc)?;
        Some(chunk.contents[block_index(local_x, local_y, local_z)])
    }
}
#[derive(Debug)]
struct ChunkBuffers {
    index: wgpu::Buffer,
    vertex: wgpu::Buffer,
    num_indices: u32,
}

pub struct ChunkManager {
    generated_buffers: HashMap<[i32; 2], ChunkBuffers>,
    pub generated_data: ChunkDataStorage,
    noise: OpenSimplex,

    pub sender: mpsc::SyncSender<MeshJob>,
    pub receiver: mpsc::Receiver<CompletedMesh>,
}

impl ChunkManager {
    /// Attempts to read and deserialize a chunk from disk; falls back to procedural generation.
    fn get_or_generate_chunk(&self, path: &Path, chunk_location: [i32; 2]) -> ChunkData {
        let loaded = std::fs::read(path).ok().and_then(|buffer| {
            // buffer lives inside this entire block
            let archived = access::<ArchivedChunkData, rkyv::rancor::Error>(&buffer).ok()?;
            deserialize::<ChunkData, rkyv::rancor::Error>(archived).ok()
            // ChunkData is owned, so buffer can be safely dropped here
        });

        loaded.unwrap_or_else(|| ChunkData {
            contents: generate(&self.noise, chunk_location),
        })
    }

    pub fn load_or_generate_chunk_arc(
        &mut self,
        path: &Path,
        chunk_location: [i32; 2],
    ) -> Arc<ChunkData> {
        if let Some(existing) = self.generated_data.get(&chunk_location) {
            return existing.clone();
        }

        let new_chunk = Arc::new(self.get_or_generate_chunk(path, chunk_location));
        self.generated_data
            .insert(chunk_location, new_chunk.clone());
        new_chunk
    }

    /// Panics
    ///
    /// If the number of indices exceeds the 32 bit integer limit
    pub fn insert_chunk(&mut self, render_context: &RenderContext) {
        // Destructure CompletedMesh fields directly in the match pattern
        while let Ok(CompletedMesh {
            vertices,
            indices,
            loc,
        }) = self.receiver.try_recv()
        {
            self.generated_buffers.insert(
                loc,
                ChunkBuffers {
                    vertex: render_context.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ),
                    index: render_context.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ),
                    num_indices: u32::try_from(indices.len())
                        .expect("mesh index count exceeded u32 limit"),
                },
            );
        }
    }
    pub fn render_chunks(&self, render_pass: &mut wgpu::RenderPass, camera: &camera::Camera) {
        self.generated_buffers
            .iter() // Iterates over (&chunk_location, &chunk_data) pairs
            .filter(|(location, _chunk)| {
                // Dereference location explicitly depending on map type (or use **location)
                cuboid_intersects_frustum(&chunkcoord_to_aabb(**location), camera)
            })
            .for_each(|(_location, chunk)| {
                // We already have a direct reference to `chunk` here—no map lookup needed!
                render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
                render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
            });
    }

    pub fn queue_mesh_job(&self, loc: [i32; 2]) {
        if let Some(center_arc) = self.generated_data.get(&loc) {
            let mut neighbours = Vec::with_capacity(8);
            for offset in NEIGHBOUR_OFFSETS {
                let n_loc = [loc[0] + offset[0], loc[1] + offset[1]];
                if let Some(neighbor_arc) = self.generated_data.get(&n_loc) {
                    neighbours.push(LocatedChunk {
                        loc: n_loc,
                        data: neighbor_arc.clone(),
                    });
                }
            }

            let job = MeshJob {
                chunk: LocatedChunk {
                    loc,
                    data: center_arc.clone(),
                },
                neighbours,
            };
            let _ = self.sender.try_send(job);
        }
    }

    pub fn set_block(&mut self, target_pos: vek::Vec3<i32>, block: BlockType) -> bool {
        if target_pos.y < 0 || target_pos.y >= CHUNK_HEIGHT_I32 {
            return false;
        }

        let (chunk_loc, [local_x, local_y, local_z]) =
            world_to_chunk_pos(target_pos.x, target_pos.y, target_pos.z);

        let Some(chunk_arc) = self.generated_data.get_mut(&chunk_loc) else {
            return false; // Chunk isn't loaded/generated yet
        };

        let idx = block_index(local_x, local_y, local_z);
        let chunk = Arc::make_mut(chunk_arc);

        // Don't re-mesh if the block didn't actually change
        if chunk.contents[idx] == block {
            return false;
        }

        chunk.contents[idx] = block;

        self.queue_mesh_with_neighbors(chunk_loc);

        true
    }

    /// Dispatches loading/generation and neighbor remeshing for the next visible chunk.
    pub fn update_visible_chunks(&mut self, camera: &Camera) {
        let Some(chunk_loc) = nearest_visible_unloaded(&self.generated_data, camera) else {
            return;
        };

        tracing::trace!(chunk_loc = ?chunk_loc, "Queueing visible chunk");

        // Let ChunkManager handle file paths internally
        let path_str = format!("{},{}.bin", chunk_loc[0], chunk_loc[1]);
        let _ = self.load_or_generate_chunk_arc(Path::new(&path_str), chunk_loc);

        // Batch meshing for chunk + 8 surrounding neighbors
        self.queue_mesh_with_neighbors(chunk_loc);
    }

    /// Queues mesh updates for a target chunk and all 8 surrounding neighbors.
    fn queue_mesh_with_neighbors(&self, [x, z]: [i32; 2]) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                self.queue_mesh_job([x + dx, z + dz]);
            }
        }
    }

    #[must_use]
    pub fn get_block(&self, pos: Vec3<i32>) -> Option<BlockType> {
        self.generated_data.get_block(pos.x, pos.y, pos.z)
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        let generated_chunk_buffers = HashMap::new();
        let (send_generate, recv_generate) = mpsc::sync_channel(10);
        let (send_chunk, recv_chunk) = mpsc::sync_channel(10);
        let generated_chunkdata = HashMap::new();
        mesh::start_meshgen(recv_generate, send_chunk);
        let noise = OpenSimplex::new(SEED);
        Self {
            generated_buffers: generated_chunk_buffers,
            generated_data: generated_chunkdata,
            noise,
            sender: send_generate,
            receiver: recv_chunk,
        }
    }
}
