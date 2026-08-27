use std::{
    sync::{Arc, mpsc},
    thread,
};

use crate::{ChunkData, mesh::builder::generate, renderer::Vertex};

pub struct LocatedChunk {
    pub loc: [i32; 2],
    pub data: Arc<ChunkData>,
}

pub struct MeshJob {
    pub chunk: LocatedChunk,
    pub neighbours: Vec<LocatedChunk>,
}

pub struct CompletedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub loc: [i32; 2],
}

pub fn start_meshgen(
    recv_generate: mpsc::Receiver<MeshJob>,
    send_chunk: mpsc::SyncSender<CompletedMesh>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        // Simple blocking loop: waits for a job, generates mesh,
        // and blocks on send if the queue is full.
        while let Ok(job) = recv_generate.recv() {
            let (vertices, indices) = generate(&job.chunk, &job.neighbours);

            let result = CompletedMesh {
                vertices,
                indices,
                loc: job.chunk.loc,
            };

            // Will block naturally if main thread hasn't consumed previous meshes,
            // providing clean backpressure without manual sleep calls.
            if send_chunk.send(result).is_err() {
                break; // Main thread disconnected / dropped receiver
            }
        }
    })
}
