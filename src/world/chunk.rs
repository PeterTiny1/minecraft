use vek::Vec3;

use crate::{
    RENDER_DISTANCE,
    block::BlockType,
    camera::Camera,
    mesh::Completed,
    world::{
        CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_WIDTH, CHUNK_WIDTH_I32, ChunkRenderer,
        loader::ChunkLoader, mesh_pipeline::MeshPipeline, save_chunks, storage::WorldStorage,
    },
};

const CHUNK_WORKER_CAPACITY: usize = 10;
const UNLOAD_MARGIN: f32 = 64.0; // Extra buffer to prevent chunk loading thrashing at borders
const UNLOAD_DISTANCE: f32 = RENDER_DISTANCE + UNLOAD_MARGIN;

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
    pub fn poll_completed_mesh(&self) -> Option<Completed> {
        self.mesh_pipeline.poll_completed()
    }

    pub fn save_all(&self) {
        let save_dir = std::path::Path::new("saves");
        save_chunks(&self.storage.data, save_dir);
    }
    pub fn unload_far_chunks(&mut self, camera_pos: Vec3<f32>, chunk_renderer: &mut ChunkRenderer) {
        let player_chunk_x = (camera_pos.x / CHUNK_WIDTH as f32).floor() as i32;
        let player_chunk_z = (camera_pos.z / CHUNK_DEPTH as f32).floor() as i32;

        let unload_dist_sq = UNLOAD_DISTANCE * UNLOAD_DISTANCE;

        // Collect keys to evict (prevents mutating storage while iterating)
        let to_unload: Vec<[i32; 2]> = self
            .storage
            .data
            .keys()
            .copied()
            .filter(|&[cx, cz]| {
                let dx = (cx - player_chunk_x) * CHUNK_WIDTH_I32;
                let dz = (cz - player_chunk_z) * CHUNK_DEPTH_I32;
                (dx * dx + dz * dz) as f32 > unload_dist_sq
            })
            .collect();

        for loc in to_unload {
            // Remove from memory
            if let Some(chunk_data) = self.storage.data.remove(&loc) {
                // Offload save to background worker thread
                self.loader.unload_chunk(loc, chunk_data);
            }

            // 2. Cancel any pending CPU meshing jobs
            self.mesh_pipeline.remove_mesh(loc);

            // 3. Drop GPU vertex & index buffers
            chunk_renderer.remove_mesh(&loc);
        }
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
