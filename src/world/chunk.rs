use vek::Vec3;

use crate::{
    block::BlockType,
    camera::Camera,
    mesh::CompletedMesh,
    world::{loader::ChunkLoader, mesh_pipeline::MeshPipeline, storage::WorldStorage},
};

const CHUNK_WORKER_CAPACITY: usize = 10;

pub struct ChunkManager {
    pub storage: WorldStorage,
    pub mesh_pipeline: MeshPipeline,
    pub loader: ChunkLoader,
}

impl ChunkManager {
    pub fn mark_dirty_with_neighbors(&mut self, loc: [i32; 2]) {
        self.mesh_pipeline
            .mark_dirty_with_neighbors(loc, &self.storage);
    }

    pub fn update_mesh_queue(&mut self, camera_pos: Vec3<f32>) {
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
        self.loader.update_visible_chunks(camera, &self.storage);
    }

    pub fn poll_completed_chunk(&mut self) -> Option<[i32; 2]> {
        let completed = self.loader.poll_completed_chunk()?;
        let loc = completed.location;

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
            mesh_pipeline: MeshPipeline::new(CHUNK_WORKER_CAPACITY),
            loader: ChunkLoader::new(CHUNK_WORKER_CAPACITY),
        }
    }
}
