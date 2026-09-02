mod builder;
mod textures;

use std::sync::Arc;

pub use builder::ChunkMeshBuilder;

use crate::{ChunkData, renderer::Vertex};

#[derive(Debug, Clone, Default)]
pub struct Data {
    pub opaque_vertices: Vec<Vertex>,
    pub opaque_indices: Vec<u32>,
    pub cutout_nocull_vertices: Vec<Vertex>,
    pub cutout_nocull_indices: Vec<u32>,
    pub translucent_vertices: Vec<Vertex>,
    pub translucent_indices: Vec<u32>,
}

pub struct LocatedChunk {
    pub loc: [i32; 2],
    pub data: Arc<ChunkData>,
}

pub struct Job {
    pub chunk: LocatedChunk,
    pub neighbours: Vec<LocatedChunk>,
}

pub struct Completed {
    pub loc: [i32; 2],
    pub data: Data,
}

pub type Worker = crate::worker::GenericWorker<Job, Completed>;
