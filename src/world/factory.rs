use noise::OpenSimplex;
use rkyv::{access, api::low::deserialize};
use std::sync::Arc;

use crate::{
    SEED,
    mesh::{ChunkMeshBuilder, CompletedMesh, MeshJob, MeshWorker},
    world::types::{ArchivedChunkData, ChunkData, ChunkJob, ChunkWorker, CompletedChunk},
};

/// Spawns the worker thread pool responsible for loading/generating terrain chunk data.
pub fn spawn_chunk_worker(capacity: usize) -> ChunkWorker {
    let noise = OpenSimplex::new(SEED);
    ChunkWorker::spawn(capacity, move |job: ChunkJob| {
        let file_name = format!("{},{}.bin", job.location[0], job.location[1]);

        let chunk_data = if let Ok(bytes) = std::fs::read(&file_name) {
            let archived = access::<ArchivedChunkData, rkyv::rancor::Error>(&bytes).ok()?;
            let data: ChunkData = deserialize::<_, rkyv::rancor::Error>(archived).ok()?;
            Arc::new(data)
        } else {
            let chunk = crate::world::generate(&noise, job.location);
            Arc::new(ChunkData { contents: chunk })
        };

        Some(CompletedChunk {
            location: job.location,
            data: chunk_data,
        })
    })
}

/// Spawns the worker thread pool responsible for building chunk render meshes.
pub fn spawn_mesh_worker(capacity: usize) -> MeshWorker {
    MeshWorker::spawn(capacity, move |job: MeshJob| {
        let loc = job.chunk.loc;
        let (vertices, indices) = ChunkMeshBuilder::build(&job.chunk, &job.neighbours);

        Some(CompletedMesh {
            vertices,
            indices,
            loc,
        })
    })
}
