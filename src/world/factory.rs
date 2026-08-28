use noise::OpenSimplex;
use rkyv::{access, api::low::deserialize};
use std::sync::Arc;

use crate::{
    SEED,
    mesh::{ChunkMeshBuilder, CompletedMesh, MeshJob, MeshWorker},
    world::{
        save::save_single_chunk,
        types::{
            ArchivedChunkData, ChunkData, ChunkJob, ChunkJobKind, ChunkWorker, CompletedChunk,
        },
    },
};

pub fn spawn_chunk_worker(capacity: usize) -> ChunkWorker {
    let noise = OpenSimplex::new(SEED);
    ChunkWorker::spawn(capacity, move |job: ChunkJob| match job.kind {
        ChunkJobKind::Save(data) => {
            let save_dir = std::path::Path::new("saves");
            save_single_chunk(job.location, &data, save_dir);
            // Fire-and-forget: No completed chunk needs to return to the main thread
            None
        }
        ChunkJobKind::LoadOrGenerate => {
            let save_dir = std::path::Path::new("saves");
            let file_path = save_dir.join(format!("{},{}.bin", job.location[0], job.location[1]));

            let chunk_data = if let Ok(bytes) = std::fs::read(&file_path) {
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
        }
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
