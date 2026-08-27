use std::collections::HashSet;

use vek::Vec3;

use crate::{
    block::BlockType,
    camera::Camera,
    mesh::CompletedMesh,
    world::{
        factory::spawn_chunk_worker,
        mesh_pipeline::MeshPipeline,
        nearest_unloaded_chunks,
        storage::WorldStorage,
        types::{ChunkJob, ChunkWorker},
    },
};

const CHUNK_WORKER_CAPACITY: usize = 10;

pub struct ChunkManager {
    pub storage: WorldStorage,
    pub mesh_pipeline: MeshPipeline,
    pending_chunks: HashSet<[i32; 2]>,
    chunk_worker: ChunkWorker,
}

impl ChunkManager {
    pub fn mark_dirty_with_neighbors(&mut self, loc: [i32; 2]) {
        self.mesh_pipeline
            .mark_dirty_with_neighbors(loc, &self.storage);
    }

    pub fn update_mesh_queue(&mut self, camera_pos: vek::Vec3<f32>) {
        self.mesh_pipeline.update_queue(camera_pos, &self.storage);
    }

    pub fn set_block(&mut self, target_pos: Vec3<i32>, block: BlockType) -> bool {
        if let Some(outcome) = self.storage.set_block(target_pos, block) {
            for loc in outcome.dirty_chunks {
                self.mesh_pipeline.mark_dirty(loc, &self.storage);
            }
            true
        } else {
            false
        }
    }

    pub fn update_visible_chunks(&mut self, camera: &Camera) {
        let free_capacity = CHUNK_WORKER_CAPACITY.saturating_sub(self.pending_chunks.len());
        if free_capacity == 0 {
            return;
        }

        let to_queue = nearest_unloaded_chunks(
            &self.storage.data,
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

    pub fn poll_completed_chunk(&mut self) -> Option<[i32; 2]> {
        let completed = self.chunk_worker.receiver.try_recv().ok()?;
        let loc = completed.location;

        self.pending_chunks.remove(&loc);
        self.storage.data.insert(loc, completed.data);
        self.mark_dirty_with_neighbors(loc);

        Some(loc)
    }

    #[must_use]
    pub fn get_block(&self, pos: Vec3<i32>) -> Option<BlockType> {
        self.storage.get_block(pos)
    }
    /// Non-blocking check for any meshes completed by background workers.
    pub fn poll_completed_mesh(&self) -> Option<CompletedMesh> {
        self.mesh_pipeline.poll_completed()
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self {
            storage: WorldStorage::new(),
            pending_chunks: HashSet::new(),
            chunk_worker: spawn_chunk_worker(CHUNK_WORKER_CAPACITY),
            mesh_pipeline: MeshPipeline::new(CHUNK_WORKER_CAPACITY),
        }
    }
}
