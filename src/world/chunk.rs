use std::{collections::HashMap, path::Path, sync::Arc};

use noise::OpenSimplex;
use rkyv::{Archive, Deserialize, Serialize, access, deserialize};
use vek::Vec3;

use crate::{
    SEED,
    block::BlockType,
    camera::Camera,
    mesh::{CompletedMesh, LocatedChunk, MeshJob, MeshWorker},
    worker::GenericWorker,
    world::{block_index, nearest_visible_unloaded, world_to_chunk_pos},
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

pub struct ChunkJob {
    pub location: [i32; 2],
}

pub struct CompletedChunk {
    pub location: [i32; 2],
    pub data: Arc<ChunkData>,
}

pub type ChunkWorker = GenericWorker<ChunkJob, CompletedChunk>;

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

pub struct ChunkManager {
    pub generated_data: ChunkDataStorage,
    mesh_worker: MeshWorker,
    chunk_worker: ChunkWorker,
}

impl ChunkManager {
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
            let _ = self.mesh_worker.sender.try_send(job);
        }
    }

    pub fn set_block(&mut self, target_pos: Vec3<i32>, block: BlockType) -> bool {
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

    /// Dispatches generation/loading of the next visible chunk to the background worker.
    pub fn update_visible_chunks(&mut self, camera: &Camera) {
        let Some(chunk_loc) = nearest_visible_unloaded(&self.generated_data, camera) else {
            return;
        };

        tracing::trace!(chunk_loc = ?chunk_loc, "Queueing visible chunk");

        let _ = self.chunk_worker.sender.try_send(ChunkJob {
            location: chunk_loc,
        });
    }

    /// Non-blocking check for any chunks generated/loaded by background workers.
    /// Inserts completed chunk data into storage and triggers meshing for its neighborhood.
    pub fn poll_completed_chunk(&mut self) -> Option<[i32; 2]> {
        let completed = self.chunk_worker.receiver.try_recv().ok()?;
        let loc = completed.location;

        self.generated_data.insert(loc, completed.data);
        self.queue_mesh_with_neighbors(loc);

        Some(loc)
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

    /// Non-blocking check for any meshes completed by background workers.
    pub fn poll_completed_mesh(&self) -> Option<CompletedMesh> {
        self.mesh_worker.receiver.try_recv().ok()
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        let noise = OpenSimplex::new(SEED);

        // Background worker handles disk I/O and terrain generation
        let chunk_worker = ChunkWorker::spawn(10, move |job: ChunkJob| {
            let path_str = format!("{},{}.bin", job.location[0], job.location[1]);
            let path = Path::new(&path_str);

            let loaded = std::fs::read(path).ok().and_then(|buffer| {
                let archived = access::<ArchivedChunkData, rkyv::rancor::Error>(&buffer).ok()?;
                deserialize::<ChunkData, rkyv::rancor::Error>(archived).ok()
            });

            let chunk_data = loaded.unwrap_or_else(|| ChunkData {
                contents: generate(&noise, job.location),
            });

            Some(CompletedChunk {
                location: job.location,
                data: Arc::new(chunk_data),
            })
        });

        let mesh_worker = MeshWorker::spawn(10, |job: MeshJob| {
            let (vertices, indices) = crate::mesh::generate(&job.chunk, &job.neighbours);

            Some(CompletedMesh {
                vertices,
                indices,
                loc: job.chunk.loc,
            })
        });

        Self {
            generated_data: HashMap::new(),
            mesh_worker,
            chunk_worker,
        }
    }
}
