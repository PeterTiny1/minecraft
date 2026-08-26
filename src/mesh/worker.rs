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
    // NORTH CLOCKWISE
    pub neighbours: Vec<LocatedChunk>,
}

pub fn start_meshgen(
    recv_generate: mpsc::Receiver<MeshJob>,
    send_chunk: mpsc::SyncSender<(Vec<Vertex>, Vec<u32>, [i32; 2])>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || manage_meshgen(&recv_generate, &send_chunk))
}

fn manage_meshgen(
    recv_generate: &mpsc::Receiver<MeshJob>,
    send_chunk: &mpsc::SyncSender<(Vec<Vertex>, Vec<u32>, [i32; 2])>,
) {
    let mut waiting: Vec<(Vec<Vertex>, Vec<u32>, [i32; 2])> = Vec::new();

    loop {
        // 1. Drain pending waiting items without cloning heavy vertex arrays
        let mut still_waiting = Vec::new();
        for item in waiting.drain(..) {
            if let Err(mpsc::TrySendError::Full(rejected)) = send_chunk.try_send(item) {
                still_waiting.push(rejected);
            }
        }
        waiting = still_waiting;

        // 2. Fetch next job
        let job = if waiting.is_empty() {
            match recv_generate.recv() {
                Ok(j) => j,
                Err(_) => break, // Channel closed
            }
        } else {
            match recv_generate.try_recv() {
                Ok(j) => j,
                Err(mpsc::TryRecvError::Empty) => {
                    thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        };

        // 3. Process mesh job
        let (mesh, indices) = generate(&job.chunk, &job.neighbours);
        let result = (mesh, indices, job.chunk.loc);

        if let Err(mpsc::TrySendError::Full(item)) = send_chunk.try_send(result) {
            waiting.push(item);
        }
    }
}
