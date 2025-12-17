use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{mpsc, Arc, RwLock},
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
    AppState, RENDER_DISTANCE, SEED,
};
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 32;
pub const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

/// Stores the data of a chunk, 32x256x32 on Linux, 16x256x16 on Windows, accessed in order x, y, z
pub type Chunk = Box<[[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]>;
pub type ChunkDataStorage = HashMap<[i32; 2], ChunkData>;
pub struct MeshRegen {
    pub chunk: [i32; 2],
    // NORTH CLOCKWISE
    pub neighbours: [bool; 8],
}

pub trait BlockProvider {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType>;
}
impl BlockProvider for ChunkDataStorage {
    fn get_block(&self, x: i32, y: i32, z: i32) -> Option<BlockType> {
        let chunk_x = x.div_euclid(CHUNK_WIDTH_I32);
        let chunk_z = z.div_euclid(CHUNK_DEPTH_I32);
        let chunk = self.get(&[chunk_x, chunk_z])?;
        let x = (x - (x.div_euclid(CHUNK_WIDTH_I32) * CHUNK_WIDTH_I32)) as usize;
        let z = (z - (z.div_euclid(CHUNK_DEPTH_I32) * CHUNK_DEPTH_I32)) as usize;
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
    pub generated_data: Arc<RwLock<ChunkDataStorage>>,
    noise: OpenSimplex,

    pub sender: mpsc::SyncSender<MeshRegen>,
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
        e: std::collections::hash_map::VacantEntry<'_, [i32; 2], ChunkData>,
        chunk_location: [i32; 2],
    ) {
        let chunk_contents = if path.exists() {
            let buffer = std::fs::read(path).unwrap();
            bincode::decode_from_slice(&buffer, bincode::config::standard())
                .unwrap()
                .0
        } else {
            generate(&self.noise, chunk_location)
        };
        e.insert(ChunkData {
            contents: chunk_contents,
        });
        match self.sender.try_send(MeshRegen {
            chunk: chunk_location,
            neighbours: [true; 8],
        }) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => panic!("Got disconnected!"),
            Err(mpsc::TrySendError::Full(_)) => todo!(),
        }
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
        for chunk_location in keys.into_iter() {
            let chunk = &self.generated_buffers[chunk_location];
            render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
            render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
        }
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        let generated_chunk_buffers = HashMap::new();
        let (send_generate, recv_generate) = mpsc::sync_channel(10);
        let (send_chunk, recv_chunk) = mpsc::sync_channel(10);
        let generated_chunkdata = Arc::new(RwLock::new(HashMap::new()));
        start_meshgen(recv_generate, Arc::clone(&generated_chunkdata), send_chunk);
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

#[derive(Debug, Clone, Encode, Decode)]
pub struct ChunkData {
    pub contents: Chunk,
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
pub fn nearest_visible_unloaded<S: ::std::hash::BuildHasher>(
    generated_chunks: &HashMap<[i32; 2], ChunkData, S>,
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
    recv_generate: mpsc::Receiver<MeshRegen>,
    chunkdata_arc: Arc<RwLock<ChunkDataStorage>>,
    send_chunk: mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || manage_meshgen(&recv_generate, &chunkdata_arc, &send_chunk))
}

fn manage_meshgen(
    recv_generate: &mpsc::Receiver<MeshRegen>,
    chunkdata_arc: &Arc<RwLock<HashMap<[i32; 2], ChunkData>>>,
    send_chunk: &mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) {
    let mut waiting = vec![];
    loop {
        let mut new_waiting = vec![];
        // TODO: work out why this doesn't work in practice
        for item in waiting {
            match send_chunk.try_send(item) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Full(v)) => new_waiting.push(v),
                Err(mpsc::TrySendError::Disconnected(_)) => break,
            }
        }
        waiting = new_waiting;
        if let Ok(MeshRegen {
            chunk: chunk_location,
            neighbours,
        }) = recv_generate.recv()
        {
            let generated_chunkdata = chunkdata_arc.read().unwrap();
            let [x, y] = chunk_location;
            let mut chunks_to_remesh = vec![[x, y]];
            if neighbours[0] {
                chunks_to_remesh.push([x + 1, y]);
            }
            if neighbours[1] {
                chunks_to_remesh.push([x + 1, y + 1]);
            }
            if neighbours[2] {
                chunks_to_remesh.push([x, y + 1]);
            }
            if neighbours[3] {
                chunks_to_remesh.push([x - 1, y + 1]);
            }
            if neighbours[4] {
                chunks_to_remesh.push([x - 1, y]);
            }
            if neighbours[5] {
                chunks_to_remesh.push([x - 1, y - 1]);
            }
            if neighbours[6] {
                chunks_to_remesh.push([x, y - 1]);
            }
            if neighbours[7] {
                chunks_to_remesh.push([x + 1, y - 1]);
            }
            for loc in chunks_to_remesh {
                if !generated_chunkdata.contains_key(&loc) {
                    continue;
                }
                let (mesh, indices) = generate_chunk_mesh(&*generated_chunkdata, loc[0], loc[1]);
                match send_chunk.try_send((mesh, indices, loc)) {
                    Ok(()) => {}
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        break;
                    }
                    Err(mpsc::TrySendError::Full(v)) => waiting.push(v),
                }
            }
        }
    }
}

/// # Panics
///
/// If the lock cannot be acquired for whatever reason
pub fn save_file(state: &AppState) {
    let generated_chunkdata = state.chunk_manager.generated_data.read().unwrap();
    let iterator = generated_chunkdata.iter();
    for (chunk_location, data) in iterator {
        let location = format!(
            "{}.bin",
            chunk_location
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let path = Path::new(&location);
        if let Ok(mut file) = File::create(path) {
            bincode::encode_into_std_write(data, &mut file, bincode::config::standard()).unwrap();
        }
    }
}
