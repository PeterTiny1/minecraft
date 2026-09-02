use std::{collections::HashSet, sync::mpsc::TrySendError};

use vek::Vec3;

use crate::{
    mesh::{Completed, Job, LocatedChunk, Worker},
    world::{
        factory::spawn_mesh_worker,
        storage::WorldStorage,
        types::{CHUNK_DEPTH_I32, CHUNK_WIDTH_I32, NEIGHBOUR_OFFSETS},
    },
};

pub struct MeshPipeline {
    queue: HashSet<[i32; 2]>,
    worker: Worker,
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

        let mut sorted_queue: Vec<[i32; 2]> = self.queue.iter().copied().collect();
        sorted_queue.sort_unstable_by_key(|&[x, z]| {
            let dx = (x - cam_x) as i64;
            let dz = (z - cam_z) as i64;
            -(dx * dx + dz * dz)
        });

        while let Some(loc) = sorted_queue.pop() {
            let Some(center_arc) = storage.data.get(&loc) else {
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

            let job = Job {
                chunk: LocatedChunk {
                    loc,
                    data: center_arc.clone(),
                },
                neighbours,
            };

            match self.worker.sender.try_send(job) {
                Ok(()) => {
                    self.queue.remove(&loc);
                }
                Err(TrySendError::Full(_)) => {
                    // Channel is saturated; stop popping so remaining items remain in `self.queue` for next frame
                    break;
                }
                Err(TrySendError::Disconnected(_)) => {
                    // Worker thread died, clean up queue to prevent infinite spinning
                    self.queue.remove(&loc);
                    break;
                }
            }
        }
    }

    /// Non-blocking check for any meshes completed by background workers.
    pub fn poll_completed(&self) -> Option<Completed> {
        self.worker.receiver.try_recv().ok()
    }

    pub fn remove_mesh(&mut self, loc: [i32; 2]) {
        self.queue.remove(&loc);
    }
}
