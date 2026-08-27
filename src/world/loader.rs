use std::collections::HashSet;

use crate::{
    camera::Camera,
    world::{
        factory::spawn_chunk_worker,
        nearest_unloaded_chunks,
        storage::WorldStorage,
        types::{ChunkJob, ChunkWorker, CompletedChunk},
    },
};

pub struct ChunkLoader {
    pending_chunks: HashSet<[i32; 2]>,
    worker: ChunkWorker,
    capacity: usize,
}

impl ChunkLoader {
    pub fn new(capacity: usize) -> Self {
        Self {
            pending_chunks: HashSet::new(),
            worker: spawn_chunk_worker(capacity),
            capacity,
        }
    }

    /// Dispatches missing visible chunks to the background worker until channel capacity is saturated.
    pub fn update_visible_chunks(&mut self, camera: &Camera, storage: &WorldStorage) {
        let free_capacity = self.capacity.saturating_sub(self.pending_chunks.len());
        if free_capacity == 0 {
            return;
        }

        let to_queue =
            nearest_unloaded_chunks(&storage.data, &self.pending_chunks, camera, free_capacity);

        for loc in to_queue {
            if self
                .worker
                .sender
                .try_send(ChunkJob { location: loc })
                .is_ok()
            {
                self.pending_chunks.insert(loc);
            }
        }
    }

    /// Non-blocking check for any chunks generated/loaded by background workers.
    /// Removes the chunk from pending state and returns the completed data.
    pub fn poll_completed_chunk(&mut self) -> Option<CompletedChunk> {
        let completed = self.worker.receiver.try_recv().ok()?;
        self.pending_chunks.remove(&completed.location);
        Some(completed)
    }
}
