use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use vek::Vec3;

use crate::{
    block::BlockType,
    camera::Camera,
    mesh::{CompletedMesh, LocatedChunk, MeshJob, MeshWorker},
    world::{
        factory::{spawn_chunk_worker, spawn_mesh_worker},
        math::resolve_block_target,
        nearest_unloaded_chunks,
        types::{
            BlockProvider, CHUNK_DEPTH_I32, CHUNK_WIDTH_I32, ChunkDataStorage, ChunkJob,
            ChunkWorker, NEIGHBOUR_OFFSETS,
        },
    },
};

const CHUNK_WORKER_CAPACITY: usize = 10;

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
        let Some((chunk_loc, idx, neighbor_offsets)) = resolve_block_target(target_pos) else {
            return false;
        };

        let Some(chunk_arc) = self.generated_data.get_mut(&chunk_loc) else {
            return false;
        };

        let chunk = Arc::make_mut(chunk_arc);
        if chunk.contents[idx] == block {
            return false;
        }

        chunk.contents[idx] = block;

        // Self always re-meshes
        self.mesh_queue.insert(chunk_loc);

        // Re-mesh direct AND diagonal neighbor chunks affected by corner AO
        for [offset_x, offset_z] in neighbor_offsets.into_iter().flatten() {
            let n_loc = [chunk_loc[0] + offset_x, chunk_loc[1] + offset_z];
            if self.generated_data.contains_key(&n_loc) {
                self.mesh_queue.insert(n_loc);
            }
        }

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
        Self {
            generated_data: HashMap::new(),
            pending_chunks: HashSet::new(),
            chunk_worker: spawn_chunk_worker(CHUNK_WORKER_CAPACITY),
            mesh_worker: spawn_mesh_worker(CHUNK_WORKER_CAPACITY),
            mesh_queue: HashSet::new(),
        }
    }
}
