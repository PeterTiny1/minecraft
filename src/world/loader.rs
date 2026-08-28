use std::{
    collections::{HashSet, VecDeque},
    sync::Arc,
};

use crate::{
    camera::Camera,
    world::{
        factory::spawn_chunk_worker,
        nearest_unloaded_chunks,
        storage::WorldStorage,
        types::{ChunkData, ChunkJob, ChunkJobKind, ChunkWorker, CompletedChunk},
    },
};

pub struct ChunkLoader {
    pending_chunks: HashSet<[i32; 2]>,
    /// Unbounded backlog for save jobs so no save is ever dropped when channel saturates
    save_backlog: VecDeque<ChunkJob>,
    worker: ChunkWorker,
    capacity: usize,
}

impl ChunkLoader {
    pub fn new(capacity: usize) -> Self {
        Self {
            pending_chunks: HashSet::new(),
            save_backlog: VecDeque::new(),
            worker: spawn_chunk_worker(capacity),
            capacity,
        }
    }

    /// Dispatches missing visible chunks and flushes pending save jobs to workers.
    pub fn update_visible_chunks(&mut self, camera: &Camera, storage: &WorldStorage) {
        // 1. Drain save backlog into worker channel first (prioritize saving memory)
        self.flush_save_backlog();

        // 2. Schedule missing chunk loads up to available channel capacity
        let free_capacity = self.capacity.saturating_sub(self.pending_chunks.len());
        if free_capacity == 0 {
            return;
        }

        let to_queue =
            nearest_unloaded_chunks(&storage.data, &self.pending_chunks, camera, free_capacity);

        for loc in to_queue {
            let job = ChunkJob {
                location: loc,
                kind: ChunkJobKind::LoadOrGenerate,
            };

            if self.worker.sender.try_send(job).is_ok() {
                self.pending_chunks.insert(loc);
            }
        }
    }

    /// Dispatches an unloaded chunk's data to be serialized and saved to disk.
    /// If the worker channel is full, queues it in an internal backlog to guarantee zero lost saves.
    pub fn unload_chunk(&mut self, location: [i32; 2], data: Arc<ChunkData>) {
        self.pending_chunks.remove(&location);

        let job = ChunkJob {
            location,
            kind: ChunkJobKind::Save(data),
        };

        // Try immediate send; if worker channel is full, push to local backlog
        if let Err(std::sync::mpsc::TrySendError::Full(overflow_job)) =
            self.worker.sender.try_send(job)
        {
            self.save_backlog.push_back(overflow_job);
        }
    }

    /// Attempts to push queued save tasks from the backlog to the worker pool.
    fn flush_save_backlog(&mut self) {
        while let Some(job) = self.save_backlog.pop_front() {
            if let Err(std::sync::mpsc::TrySendError::Full(returned_job)) =
                self.worker.sender.try_send(job)
            {
                // Channel is saturated again; put the job back and try again next frame
                self.save_backlog.push_front(returned_job);
                break;
            }
        }
    }

    /// Non-blocking check for any chunks generated/loaded by background workers.
    pub fn poll_completed_chunk(&mut self) -> Option<CompletedChunk> {
        let completed = self.worker.receiver.try_recv().ok()?;
        self.pending_chunks.remove(&completed.location);
        Some(completed)
    }
}
