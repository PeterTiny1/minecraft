use rkyv::{Archive, Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{block::BlockType, worker::GenericWorker};

// --- World & Chunk Dimensions ---
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_HEIGHT_I32: i32 = CHUNK_HEIGHT as i32;
pub const CHUNK_DEPTH: usize = 32;
pub const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

pub const CHUNK_SIZE: usize = CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_DEPTH;
pub type Chunk = Box<[BlockType; CHUNK_SIZE]>;

// --- Serialized Data Wrappers ---
#[derive(Debug, Clone, Deserialize, Serialize, Archive)]
pub struct ChunkData {
    pub contents: Chunk,
}

pub type ChunkDataStorage = HashMap<[i32; 2], Arc<ChunkData>>;

// --- Worker Job DTOs ---
pub struct ChunkJob {
    pub location: [i32; 2],
}

pub struct CompletedChunk {
    pub location: [i32; 2],
    pub data: Arc<ChunkData>,
}

pub type ChunkWorker = GenericWorker<ChunkJob, CompletedChunk>;

// --- Constants ---
pub const NEIGHBOUR_OFFSETS: [[i32; 2]; 8] = [
    [1, 0],   // 0: [x + 1, y]
    [1, 1],   // 1: [x + 1, y + 1]
    [0, 1],   // 2: [x, y + 1]
    [-1, 1],  // 3: [x - 1, y + 1]
    [-1, 0],  // 4: [x - 1, y]
    [-1, -1], // 5: [x - 1, y - 1]
    [0, -1],  // 6: [x, y - 1]
    [1, -1],  // 7: [x + 1, y - 1]
];
