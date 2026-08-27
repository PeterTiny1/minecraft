use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use noise::OpenSimplex;
use rkyv::{Archive, Deserialize, Serialize, access, deserialize};
use vek::Vec3;

use crate::{
    SEED,
    block::BlockType,
    camera::Camera,
    mesh::{ChunkMeshBuilder, CompletedMesh, LocatedChunk, MeshJob, MeshWorker},
    worker::GenericWorker,
    world::{block_index, generate, nearest_unloaded_chunks, world_to_chunk_pos},
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

const CHUNK_WORKER_CAPACITY: usize = 10;

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
    pending_chunks: HashSet<[i32; 2]>,

    // Backlog of chunks waiting for mesh generation/re-mesh
    mesh_queue: HashSet<[i32; 2]>,

    mesh_worker: MeshWorker,
    chunk_worker: ChunkWorker,
}

impl ChunkManager {
    /// Schedule a chunk and its 8 neighbors for re-meshing
    pub fn mark_dirty_with_neighbors(&mut self, [x, z]: [i32; 2]) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                let loc = [x + dx, z + dz];
                // Only queue meshing if the target chunk is actually loaded
                if self.generated_data.contains_key(&loc) {
                    self.mesh_queue.insert(loc);
                }
            }
        }
    }

    /// Processes the main thread mesh backlog, prioritizing chunks closest to the camera,
    /// and dispatches jobs to the worker as channel capacity allows.
    pub fn update_mesh_queue(&mut self, camera_pos: vek::Vec3<f32>) {
        if self.mesh_queue.is_empty() {
            return;
        }

        let cam_x = (camera_pos.x as i32).div_euclid(CHUNK_WIDTH_I32);
        let cam_z = (camera_pos.z as i32).div_euclid(CHUNK_DEPTH_I32);

        // Sort candidates so the closest chunks are at the END of the vector
        // (allowing efficient `pop()`ing from nearest to furthest).
        let mut sorted_queue: Vec<[i32; 2]> = self.mesh_queue.iter().copied().collect();
        sorted_queue.sort_unstable_by_key(|&[x, z]| {
            let dx = (x - cam_x) as i64;
            let dz = (z - cam_z) as i64;
            -(dx * dx + dz * dz)
        });

        while let Some(loc) = sorted_queue.pop() {
            let Some(center_arc) = self.generated_data.get(&loc) else {
                // If the chunk is no longer in storage, remove it from the backlog
                self.mesh_queue.remove(&loc);
                continue;
            };

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

            // Non-blocking send to mesh worker
            match self.mesh_worker.sender.try_send(job) {
                Ok(()) => {
                    self.mesh_queue.remove(&loc);
                }
                Err(_) => {
                    // Channel full; leave remaining items (including `loc`) in `self.mesh_queue`
                    break;
                }
            }
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

        self.mark_dirty_with_neighbors(chunk_loc);

        true
    }

    /// Dispatches missing visible chunks to the background worker until channel capacity is saturated.
    pub fn update_visible_chunks(&mut self, camera: &Camera) {
        let free_capacity = CHUNK_WORKER_CAPACITY.saturating_sub(self.pending_chunks.len());
        if free_capacity == 0 {
            return;
        }

        let to_queue = nearest_unloaded_chunks(
            &self.generated_data,
            &self.pending_chunks,
            camera,
            free_capacity,
        );

        for loc in to_queue {
            if self
                .chunk_worker
                .sender
                .try_send(ChunkJob { location: loc })
                .is_ok()
            {
                self.pending_chunks.insert(loc);
            }
        }
    }

    /// Non-blocking check for any chunks generated/loaded by background workers.
    /// Inserts completed chunk data into storage and flags its neighborhood as dirty for meshing.
    pub fn poll_completed_chunk(&mut self) -> Option<[i32; 2]> {
        let completed = self.chunk_worker.receiver.try_recv().ok()?;
        let loc = completed.location;

        self.pending_chunks.remove(&loc);
        self.generated_data.insert(loc, completed.data);
        self.mark_dirty_with_neighbors(loc);

        Some(loc)
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
        let chunk_worker = ChunkWorker::spawn(CHUNK_WORKER_CAPACITY, move |job: ChunkJob| {
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
            let (vertices, indices) = ChunkMeshBuilder::build(&job.chunk, &job.neighbours);

            Some(CompletedMesh {
                vertices,
                indices,
                loc: job.chunk.loc,
            })
        });

        Self {
            generated_data: HashMap::new(),
            pending_chunks: HashSet::new(),
            mesh_queue: HashSet::new(),
            mesh_worker,
            chunk_worker,
        }
    }
}
