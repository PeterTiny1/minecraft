use std::{
    collections::HashMap,
    path::Path,
    sync::{mpsc, Arc},
    thread,
};

use bincode::{Decode, Encode};
use noise::OpenSimplex;
use vek::{Aabb, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    block::BlockType,
    camera,
    mesh_gen::{generate_chunk_mesh, Index},
    renderer::{cuboid_intersects_frustum, RenderContext, Vertex},
    world_gen::generate,
    RENDER_DISTANCE, SEED,
};
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 32;
pub const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

pub type Chunk = Box<[[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]>;
#[derive(Debug, Clone, Encode, Decode)]
pub struct ChunkData {
    pub contents: Chunk,
}
pub type ChunkDataStorage = HashMap<[i32; 2], Arc<ChunkData>>;
pub struct LocatedChunk {
    pub loc: [i32; 2],
    pub data: Arc<ChunkData>,
}
pub struct MeshJob {
    pub chunk: LocatedChunk,
    // NORTH CLOCKWISE
    pub neighbours: Vec<LocatedChunk>,
}
const NEIGHBOUR_OFFSETS: [[i32; 2]; 8] = [
    [1, 0],   // 0: [x + 1, y]
    [1, 1],   // 1: [x + 1, y + 1]
    [0, 1],   // 2: [x, y + 1]
    [-1, 1],  // 3: [x - 1, y + 1]
    [-1, 0],  // 4: [x - 1, y]
    [-1, -1], // 5: [x - 1, y - 1]
    [0, -1],  // 6: [x, y - 1]
    [1, -1],  // 7: [x + 1, y - 1]
];
pub trait BlockProvider {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType>;
}
impl BlockProvider for ChunkDataStorage {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        let chunk_x = x.div_euclid(CHUNK_WIDTH_I32);
        let chunk_z = z.div_euclid(CHUNK_DEPTH_I32);
        let chunk = self.get(&[chunk_x, chunk_z])?;
        let x = x.rem_euclid(CHUNK_WIDTH_I32) as usize;
        let z = z.rem_euclid(CHUNK_DEPTH_I32) as usize;
        #[allow(clippy::cast_sign_loss)]
        if y >= 0 && (y as usize) < CHUNK_HEIGHT {
            Some(chunk.contents[x][y as usize][z])
        } else {
            None
        }
    }
}
#[derive(Debug)]
struct ChunkBuffers {
    index: wgpu::Buffer,
    vertex: wgpu::Buffer,
    num_indices: u32,
}

pub struct ChunkManager {
    generated_buffers: HashMap<[i32; 2], ChunkBuffers>,
    pub generated_data: ChunkDataStorage,
    noise: OpenSimplex,

    pub sender: mpsc::SyncSender<MeshJob>,
    pub receiver: mpsc::Receiver<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
}

impl ChunkManager {
    /// # Panics
    ///
    /// If the file at the path cannot be read
    /// If bincode cannot decode the data
    pub fn load_chunk(
        &self,
        path: &Path,
        e: std::collections::hash_map::VacantEntry<'_, [i32; 2], Arc<ChunkData>>,
        chunk_location: [i32; 2],
    ) -> Arc<ChunkData> {
        let chunk_contents = if path.exists() {
            let buffer = std::fs::read(path).unwrap();
            bincode::decode_from_slice(&buffer, bincode::config::standard())
                .unwrap()
                .0
        } else {
            generate(&self.noise, chunk_location)
        };

        let center_arc = Arc::new(ChunkData {
            contents: chunk_contents,
        });

        // Insert and return a clone of the Arc
        e.insert(center_arc.clone());

        center_arc
    }
    pub fn load_and_insert_chunk(
        &mut self,
        path: &Path,
        chunk_location: [i32; 2],
    ) -> Arc<ChunkData> {
        let chunk_contents = if path.exists() {
            let buffer = std::fs::read(path).unwrap();
            bincode::decode_from_slice(&buffer, bincode::config::standard())
                .unwrap()
                .0
        } else {
            generate(&self.noise, chunk_location)
        };

        let center_arc = Arc::new(ChunkData {
            contents: chunk_contents,
        });

        // Grab the entry and insert it completely internally where it won't conflict
        self.generated_data
            .insert(chunk_location, center_arc.clone());

        center_arc
    }
    /// Panics
    ///
    /// If the number of indices exceeds the 32 bit integer limit
    pub fn insert_chunk(&mut self, render_context: &RenderContext) {
        while let Ok((mesh, indices, index)) = self.receiver.try_recv() {
            self.generated_buffers.insert(
                index,
                ChunkBuffers {
                    vertex: render_context.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Vertex Buffer"),
                            contents: bytemuck::cast_slice(&mesh),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    ),
                    index: render_context.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("Index Buffer"),
                            contents: bytemuck::cast_slice(&indices),
                            usage: wgpu::BufferUsages::INDEX,
                        },
                    ),
                    num_indices: u32::try_from(indices.len()).unwrap(),
                },
            );
            // let (vertsize, indexsize) = self
            //     .generated_chunk_buffers
            //     .iter()
            //     .fold((0, 0), |acc, (_, item)| {
            //         (acc.0 + item.vertex.size(), acc.1 + item.index.size())
            //     });

            // println!("Index space: {indexsize}");
            // println!("Vertex space: {vertsize}");
        }
    }
    pub fn render_chunks(&self, render_pass: &mut wgpu::RenderPass, camera: &camera::Camera) {
        let mut keys: Vec<_> = self
            .generated_buffers
            .keys()
            .filter(|c| cuboid_intersects_frustum(&chunkcoord_to_aabb(**c), camera))
            .collect();

        keys.sort_by(|&a, &b| {
            ((b[0] * CHUNK_WIDTH_I32 - camera.get_position().x as i32).pow(2)
                + (b[1] * CHUNK_DEPTH_I32 - camera.get_position().z as i32).pow(2))
            .cmp(
                &((a[0] * CHUNK_WIDTH_I32 - camera.get_position().x as i32).pow(2)
                    + (a[1] * CHUNK_DEPTH_I32 - camera.get_position().z as i32).pow(2)),
            )
        });
        for chunk_location in keys {
            let chunk = &self.generated_buffers[chunk_location];
            render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
            render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
        }
    }
    pub fn queue_mesh_job(&self, world_map: &HashMap<[i32; 2], Arc<ChunkData>>, loc: [i32; 2]) {
        if let Some(center_arc) = world_map.get(&loc) {
            let mut neighbours = Vec::new();
            for offset in NEIGHBOUR_OFFSETS {
                let n_loc = [loc[0] + offset[0], loc[1] + offset[1]];
                if let Some(neighbor_arc) = world_map.get(&n_loc) {
                    neighbours.push(LocatedChunk {
                        loc: n_loc,
                        data: neighbor_arc.clone(),
                    });
                }
            }

            let job = MeshJob {
                chunk: LocatedChunk {
                    loc,
                    data: center_arc.clone(),
                },
                neighbours,
            };
            let _ = self.sender.try_send(job); // Handle or log error if needed
        }
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        let generated_chunk_buffers = HashMap::new();
        let (send_generate, recv_generate) = mpsc::sync_channel(10);
        let (send_chunk, recv_chunk) = mpsc::sync_channel(10);
        let generated_chunkdata = HashMap::new();
        start_meshgen(recv_generate, send_chunk);
        let noise = OpenSimplex::new(SEED);
        Self {
            generated_buffers: generated_chunk_buffers,
            generated_data: generated_chunkdata,
            noise,
            sender: send_generate,
            receiver: recv_chunk,
        }
    }
}

const MAX_DISTANCE_X: i32 = RENDER_DISTANCE as i32 / CHUNK_WIDTH_I32 + 1;
const MAX_DISTANCE_Y: i32 = RENDER_DISTANCE as i32 / CHUNK_DEPTH_I32 + 1;

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn chunkcoord_to_aabb(coord: [i32; 2]) -> Aabb<f32> {
    let min = Vec3::new(
        (coord[0] * CHUNK_WIDTH_I32) as f32,
        0.0,
        (coord[1] * CHUNK_DEPTH_I32) as f32,
    );
    Aabb {
        min,
        max: min + Vec3::new(CHUNK_WIDTH as f32, CHUNK_HEIGHT as f32, CHUNK_DEPTH as f32),
    }
}

#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn nearest_visible_unloaded(
    generated_chunks: &HashMap<[i32; 2], Arc<ChunkData>>,
    camera: &camera::Camera,
) -> Option<[i32; 2]> {
    let chunk_x = (camera.get_position().x as i32).div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = (camera.get_position().z as i32).div_euclid(CHUNK_WIDTH_I32);
    let length = |a, b| a * a + b * b;
    (-MAX_DISTANCE_X..=MAX_DISTANCE_X)
        .flat_map(|i| {
            (-MAX_DISTANCE_Y..=MAX_DISTANCE_Y).filter_map(move |j| {
                let distance = length(i, j);
                let location = [i + chunk_x, j + chunk_z];
                if distance <= (RENDER_DISTANCE * RENDER_DISTANCE) as i32
                    && !generated_chunks.contains_key(&location)
                    && cuboid_intersects_frustum(&chunkcoord_to_aabb([i, j]), camera)
                {
                    Some((location, distance))
                } else {
                    None
                }
            })
        })
        .reduce(|acc, v| if acc.1 > v.1 { v } else { acc })
        .map(|(loc, _)| loc)
}

pub fn start_meshgen(
    recv_generate: mpsc::Receiver<MeshJob>,
    send_chunk: mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || manage_meshgen(&recv_generate, &send_chunk))
}

fn manage_meshgen(
    recv_generate: &mpsc::Receiver<MeshJob>,
    send_chunk: &mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) {
    let mut waiting = vec![];
    loop {
        let mut new_waiting = vec![];
        for item in waiting {
            match send_chunk.try_send(item) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(v)) => new_waiting.push(v),
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
        }
        waiting = new_waiting;

        // 1. Receive the self-contained MeshJob.
        // We can destructure it right here in the match arm.
        if let Ok(MeshJob { chunk, neighbours }) = recv_generate.recv() {
            // 2. Compute the mesh using only the isolated data given to this job.
            // No global HashMap, no RwLock reading, zero lock contention!
            let (mesh, indices) = generate_chunk_mesh(&chunk, &neighbours);

            // 3. Try to push the completed mesh data up to the main thread
            match send_chunk.try_send((mesh, indices, chunk.loc)) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    break;
                }
                Err(mpsc::TrySendError::Full(v)) => waiting.push(v),
            }
        }
    }
}
