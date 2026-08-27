use std::collections::HashSet;

use vek::Vec3;

use crate::{
    mesh::{CompletedMesh, LocatedChunk, MeshJob, MeshWorker},
    world::{
        factory::spawn_mesh_worker,
        storage::WorldStorage,
        types::{CHUNK_DEPTH_I32, CHUNK_WIDTH_I32, NEIGHBOUR_OFFSETS},
    },
};

pub struct MeshPipeline {
    queue: HashSet<[i32; 2]>,
    worker: MeshWorker,
}

impl MeshPipeline {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: HashSet::new(),
            worker: spawn_mesh_worker(capacity),
        }
    }

    pub fn mark_dirty(&mut self, loc: [i32; 2], storage: &WorldStorage) {
        if storage.data.contains_key(&loc) {
            self.queue.insert(loc);
        }
    }

    /// Schedule a chunk and its 8 neighbors for re-meshing
    pub fn mark_dirty_with_neighbors(&mut self, [x, z]: [i32; 2], storage: &WorldStorage) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                let loc = [x + dx, z + dz];
                self.mark_dirty(loc, storage);
            }
        }
    }

    /// Processes the main thread mesh backlog, prioritizing chunks closest to the camera,
    /// and dispatches jobs to the worker as channel capacity allows.
    pub fn update_queue(&mut self, camera_pos: Vec3<f32>, storage: &WorldStorage) {
        if self.queue.is_empty() {
            return;
        }

        let cam_x = (camera_pos.x as i32).div_euclid(CHUNK_WIDTH_I32);
        let cam_z = (camera_pos.z as i32).div_euclid(CHUNK_DEPTH_I32);

        // Sort candidates so the closest chunks are at the END of the vector
        // (allowing efficient `pop()`ing from nearest to furthest).
        let mut sorted_queue: Vec<[i32; 2]> = self.queue.iter().copied().collect();
        sorted_queue.sort_unstable_by_key(|&[x, z]| {
            let dx = (x - cam_x) as i64;
            let dz = (z - cam_z) as i64;
            -(dx * dx + dz * dz)
        });

        while let Some(loc) = sorted_queue.pop() {
            let Some(center_arc) = storage.data.get(&loc) else {
                // If the chunk is no longer in storage, remove it from the backlog
                self.queue.remove(&loc);
                continue;
            };

            let mut neighbours = Vec::with_capacity(8);
            for offset in NEIGHBOUR_OFFSETS {
                let n_loc = [loc[0] + offset[0], loc[1] + offset[1]];
                if let Some(neighbor_arc) = storage.data.get(&n_loc) {
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
            match self.worker.sender.try_send(job) {
                Ok(()) => {
                    self.queue.remove(&loc);
                }
                Err(_) => {
                    // Channel full; leave remaining items in `self.queue`
                    break;
                }
            }
        }
    }

    /// Non-blocking check for any meshes completed by background workers.
    pub fn poll_completed(&self) -> Option<CompletedMesh> {
        self.worker.receiver.try_recv().ok()
    }
}
