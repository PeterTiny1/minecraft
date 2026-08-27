use crate::{ChunkData, renderer::Vertex};
use std::sync::Arc;

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

pub type MeshWorker = crate::worker::GenericWorker<MeshJob, CompletedMesh>;
