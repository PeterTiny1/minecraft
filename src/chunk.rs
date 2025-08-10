use std::{
    collections::HashMap,
    f32::consts::FRAC_1_SQRT_2,
    path::Path,
    sync::{mpsc, Arc, Mutex},
    thread,
};

use bincode::{Decode, Encode};
use half::f16;
use itertools::Itertools;
use noise::{NoiseFn, OpenSimplex};
use vek::{Aabb, Vec3};
use wgpu::util::DeviceExt;

use crate::{
    camera, cuboid_intersects_frustum, ChunkDataStorage, RenderContext, Vertex, RENDER_DISTANCE,
    SEED,
};
#[cfg(target_os = "windows")]
pub const CHUNK_WIDTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
#[cfg(target_os = "windows")]
pub const CHUNK_DEPTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_DEPTH: usize = 32;
pub const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

const TEXTURE_WIDTH: f32 = 1.0 / 16.0;
const HALF_TEXTURE_WIDTH: f32 = TEXTURE_WIDTH / 2.0;

/// Stores the data of a chunk, 32x256x32 on Linux, 16x256x16 on Windows, accessed in order x, y, z
type Chunk = [[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];
static EMPTY_CHUNK: Chunk = [[[BlockType::Air; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];
pub type Index = u32;
#[derive(Debug)]
struct ChunkBuffers {
    index: wgpu::Buffer,
    vertex: wgpu::Buffer,
    num_indices: u32,
}

#[derive(Clone, Copy)]
enum Biome {
    BirchFalls,
    GreenGrove,
    DarklogForest,
    // PineHills,
    // SnowDesert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Encode, Decode)]
pub enum BlockType {
    #[default]
    Air,
    Stone,
    GrassBlock0,
    GrassBlock1,
    GrassBlock2,
    Grass0,
    Grass1,
    Grass2,
    Flower0,
    Flower1,
    Flower2,
    Wood,
    BirchWood,
    DarkWood,
    DarkLeaf,
    Leaf,
    BirchLeaf,
    Water,
    Sand,
    Dirt,
}

impl BlockType {
    #[inline]
    const fn get_offset(self) -> [[f32; 2]; 6] {
        const TEXTURE_WIDTH_2: f32 = TEXTURE_WIDTH * 2.;
        const TEXTURE_WIDTH_3: f32 = TEXTURE_WIDTH * 3.;
        const TEXTURE_WIDTH_4: f32 = TEXTURE_WIDTH * 4.;
        const TEXTURE_WIDTH_5: f32 = TEXTURE_WIDTH * 5.;
        const TEXTURE_WIDTH_6: f32 = TEXTURE_WIDTH * 6.;
        const TEXTURE_WIDTH_7: f32 = TEXTURE_WIDTH * 7.;
        const TEXTURE_WIDTH_8: f32 = TEXTURE_WIDTH * 8.;
        const TEXTURE_WIDTH_9: f32 = TEXTURE_WIDTH * 9.;
        match self {
            Self::Stone => [[0., TEXTURE_WIDTH_8]; 6],
            Self::GrassBlock0 => [
                [0., TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [0., 0.],
            ],
            Self::GrassBlock1 => [
                [0., TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
                [0., 0.],
            ],
            Self::GrassBlock2 => [
                [0., TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
                [0., 0.],
            ],
            Self::BirchWood => [
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH],
            ],
            Self::Wood => [
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH_2],
            ],
            Self::DarkWood => [
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
                [TEXTURE_WIDTH_7, TEXTURE_WIDTH_3],
            ],
            Self::BirchLeaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH]; 6],
            Self::Leaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH_2]; 6],
            Self::DarkLeaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH_3]; 6],
            Self::Grass0 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH]; 6],
            Self::Grass1 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH_2]; 6],
            Self::Grass2 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH_3]; 6],
            Self::Flower0 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH]; 6],
            Self::Flower1 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH_2]; 6],
            Self::Flower2 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH_3]; 6],
            Self::Water => [
                [TEXTURE_WIDTH_4, 0.],
                [TEXTURE_WIDTH_5, 0.],
                [TEXTURE_WIDTH_5, 0.],
                [TEXTURE_WIDTH_5, 0.],
                [TEXTURE_WIDTH_5, 0.],
                [TEXTURE_WIDTH_4, 0.],
            ],
            Self::Sand => [[TEXTURE_WIDTH_9, 0.]; 6],
            Self::Dirt => [[0., 0.]; 6],
            Self::Air => [[0.; 2]; 6],
        }
    }

    #[inline]
    pub const fn is_solid(self) -> bool {
        !self.is_transparent() || matches!(self, Self::Leaf | Self::BirchLeaf | Self::DarkLeaf)
    }

    #[inline]
    pub const fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Air | Self::Leaf | Self::BirchLeaf | Self::DarkLeaf
        ) || self.is_liquid()
            || self.is_grasslike()
    }

    #[inline]
    pub const fn is_liquid(self) -> bool {
        matches!(self, Self::Water)
    }

    #[inline]
    pub const fn is_grasslike(self) -> bool {
        matches!(
            self,
            Self::Flower0
                | Self::Flower1
                | Self::Flower2
                | Self::Grass0
                | Self::Grass1
                | Self::Grass2
        )
    }
}

pub struct ChunkManager {
    generated_buffers: HashMap<[i32; 2], ChunkBuffers>,
    pub generated_data: Arc<Mutex<ChunkDataStorage>>,
    noise: OpenSimplex,

    pub sender: mpsc::SyncSender<[i32; 2]>,
    pub receiver: mpsc::Receiver<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
}

impl ChunkManager {
    pub fn new() -> Self {
        let generated_chunk_buffers = HashMap::new();
        let (send_generate, recv_generate) = mpsc::sync_channel(10);
        let (send_chunk, recv_chunk) = mpsc::sync_channel(10);
        let generated_chunkdata = Arc::new(Mutex::new(HashMap::new()));
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
        match self.sender.try_send(chunk_location) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Disconnected(_)) => panic!("Got disconnected!"),
            Err(mpsc::TrySendError::Full(_)) => todo!(),
        }
    }
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
        for chunk_location in self
            .generated_buffers
            .keys()
            .sorted_by(|&a, &b| {
                ((b[0] * CHUNK_WIDTH_I32 - camera.get_position().x as i32).pow(2)
                    + (b[1] * CHUNK_DEPTH_I32 - camera.get_position().z as i32).pow(2))
                .cmp(
                    &((a[0] * CHUNK_WIDTH_I32 - camera.get_position().x as i32).pow(2)
                        + (a[1] * CHUNK_DEPTH_I32 - camera.get_position().z as i32).pow(2)),
                )
            })
            .filter(|c| cuboid_intersects_frustum(&chunkcoord_to_aabb(**c), camera))
        {
            let chunk = &self.generated_buffers[chunk_location];
            render_pass.set_vertex_buffer(0, chunk.vertex.slice(..));
            render_pass.set_index_buffer(chunk.index.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct ChunkData {
    pub contents: Chunk,
}

const MAX_DISTANCE_X: i32 = RENDER_DISTANCE as i32 / CHUNK_WIDTH_I32 + 1;
const MAX_DISTANCE_Y: i32 = RENDER_DISTANCE as i32 / CHUNK_DEPTH_I32 + 1;
const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;

pub fn noise_at(
    noise: &OpenSimplex,
    x: i32,
    z: i32,
    chunk_location: [i32; 2],
    scale: f64,
    offset: f64,
) -> f64 {
    noise.get([
        f64::from(x + (chunk_location[0] * CHUNK_WIDTH_I32)) / scale + offset,
        f64::from(z + (chunk_location[1] * CHUNK_DEPTH_I32)) / scale + offset,
    ])
}

fn chunk_at_block(
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    x: i32,
    z: i32,
) -> Option<&ChunkData> {
    let chunk_x = x.div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = z.div_euclid(CHUNK_DEPTH_I32);
    generated_chunks.get(&[chunk_x, chunk_z])
}

#[allow(clippy::cast_sign_loss)]
pub fn get_block(
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    x: i32,
    y: i32,
    z: i32,
) -> Option<BlockType> {
    let chunk = chunk_at_block(generated_chunks, x, z)?;
    let x = (x - (x.div_euclid(CHUNK_WIDTH_I32) * CHUNK_WIDTH_I32)) as usize;
    let z = (z - (z.div_euclid(CHUNK_DEPTH_I32) * CHUNK_DEPTH_I32)) as usize;
    #[allow(clippy::cast_sign_loss)]
    if y >= 0 && (y as usize) < CHUNK_HEIGHT {
        Some(chunk.contents[x][y as usize][z])
    } else {
        None
    }
}

#[allow(clippy::cast_precision_loss)]
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
pub fn nearest_visible_unloaded(
    x: f32,
    z: f32,
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    camera: &camera::Camera,
) -> Option<[i32; 2]> {
    let chunk_x = (x as i32).div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = (z as i32).div_euclid(CHUNK_WIDTH_I32);
    let length = |a, b| (a * a + b * b);
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

#[test]
fn test_add_arrs() {
    assert_eq!(add_arrs([1.0, 2.0], [3.0, 4.0]), [4.0, 6.0]);
}

#[inline]
fn add_arrs(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

const WATER_HEIGHT: usize = 64;
const BIOME_SCALE: f64 = 250.0;
// const SCALING_FACTOR: f64 = 0.011;

pub fn generate(noise: &OpenSimplex, location: [i32; 2]) -> Chunk {
    let heightmap = generate_worldscale_heightmap(noise, location);
    let biomemap = generate_biomemap(noise, location);
    let mut contents = EMPTY_CHUNK;

    for x in 0..CHUNK_WIDTH {
        for y in (0..CHUNK_HEIGHT).rev() {
            for z in 0..CHUNK_DEPTH {
                let terrain_height = heightmap[x][z];
                let biome = biomemap[x][z];
                contents[x][y][z] = determine_type(terrain_height, x, y, z, biome, noise);
            }
        }
    }
    let trees = generate_trees(noise, location);
    for (x, z) in trees {
        let height = heightmap[x][z];
        if height <= WATER_HEIGHT {
            continue;
        }
        let biome = biomemap[x][z];
        place_tree(biome, &mut contents, x, height, z);
    }
    contents
}

const fn place_tree(biome: Biome, contents: &mut Chunk, x: usize, height: usize, z: usize) {
    let wood_type = match biome {
        Biome::BirchFalls => BlockType::BirchWood,
        Biome::GreenGrove => BlockType::Wood,
        Biome::DarklogForest => BlockType::DarkWood,
    };
    let leaf_type = match biome {
        Biome::BirchFalls => BlockType::BirchLeaf,
        Biome::GreenGrove => BlockType::Leaf,
        Biome::DarklogForest => BlockType::DarkLeaf,
    };
    contents[x][height + 1][z] = wood_type;
    contents[x][height + 2][z] = wood_type;
    contents[x][height + 3][z] = wood_type;
    contents[x][height + 4][z] = wood_type;
    contents[x][height + 5][z] = leaf_type;
}

fn generate_biomemap(
    noise: &OpenSimplex,
    chunk_location: [i32; 2],
) -> [[Biome; CHUNK_DEPTH]; CHUNK_WIDTH] {
    let mut biomemap = [[Biome::BirchFalls; CHUNK_DEPTH]; CHUNK_WIDTH];

    for (x, row) in biomemap.iter_mut().enumerate() {
        for (z, item) in row.iter_mut().enumerate() {
            let v = noise_at(noise, x as i32, z as i32, chunk_location, BIOME_SCALE, 18.9);
            let biome = determine_biome(v);
            *item = biome;
        }
    }

    biomemap
}

fn determine_biome(v: f64) -> Biome {
    if v > 0.2 {
        Biome::DarklogForest
    } else if v > -0.1 {
        Biome::GreenGrove
    } else {
        Biome::BirchFalls
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn generate_worldscale_heightmap(
    noise: &OpenSimplex,
    location: [i32; 2],
) -> [[usize; CHUNK_DEPTH]; CHUNK_WIDTH] {
    generate_heightmap(noise, location).map(|a| a.map(|b| b as usize))
}

const HEIGHT_SCALE: f64 = 96.0;
#[allow(clippy::cast_precision_loss)]
fn generate_heightmap(
    noise: &OpenSimplex,
    location: [i32; 2],
) -> [[f64; CHUNK_DEPTH]; CHUNK_WIDTH] {
    const OCTAVES: usize = 5;
    const PERSISTENCE: f64 = 0.5;
    const LACUNARITY: f64 = 2.0;

    let mut heightmap = [[0.0; CHUNK_DEPTH]; CHUNK_WIDTH];
    // let mut effect = [[0.0; CHUNK_DEPTH]; CHUNK_WIDTH];

    for (x, column) in heightmap.iter_mut().enumerate() {
        for (z, tile) in column.iter_mut().enumerate() {
            let mut amplitude = 1.0;
            let mut frequency = 0.007;
            let mut noise_height = 0.8;
            let mut gradient_x = 0.0;
            let mut gradient_z = 0.0;

            for octave in 0..OCTAVES {
                let octave = octave * 5;
                let sample_x = f64::from(location[0] * CHUNK_WIDTH_I32 + x as i32) * frequency;
                let sample_z = f64::from(location[1] * CHUNK_DEPTH_I32 + z as i32) * frequency;
                gradient_x += {
                    let a = (f64::from(location[0] * CHUNK_WIDTH_I32 + x as i32) - 0.1) * frequency;
                    let sample_a = noise.get([a, sample_z, octave as f64]);
                    let b = (f64::from(location[0] * CHUNK_WIDTH_I32 + x as i32) + 0.1) * frequency;
                    let sample_b = noise.get([b, sample_z, octave as f64]);
                    (sample_a - sample_b) * amplitude
                };
                gradient_z += {
                    let c = (f64::from(location[1] * CHUNK_DEPTH_I32 + z as i32) - 0.1) * frequency;
                    let sample_c = noise.get([sample_x, c, octave as f64]);
                    let d = (f64::from(location[1] * CHUNK_DEPTH_I32 + z as i32) + 0.1) * frequency;
                    let sample_d = noise.get([sample_x, d, octave as f64]);
                    (sample_c - sample_d) * amplitude
                };
                let effect = 1.0 / (1.0 + gradient_x.hypot(gradient_z));
                let octave_noise = noise.get([sample_x, sample_z, octave as f64]);
                noise_height += octave_noise * amplitude * effect;
                amplitude *= PERSISTENCE;
                frequency *= LACUNARITY;
            }

            *tile = noise_height * HEIGHT_SCALE;
        }
    }

    heightmap
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn determine_type(
    terrain_height: usize,
    x: usize,
    y: usize,
    z: usize,
    biome: Biome,
    noise: &OpenSimplex,
) -> BlockType {
    if y < terrain_height - 3 {
        BlockType::Stone
    } else if y < terrain_height {
        BlockType::Dirt
    } else if y == terrain_height {
        if terrain_height > WATER_HEIGHT {
            match biome {
                Biome::BirchFalls => BlockType::GrassBlock0,
                Biome::GreenGrove => BlockType::GrassBlock1,
                Biome::DarklogForest => BlockType::GrassBlock2,
            }
        } else {
            BlockType::Sand
        }
    } else if y < WATER_HEIGHT {
        BlockType::Water
    } else if terrain_height > WATER_HEIGHT
        && y == terrain_height + 1
        && noise.get([x as f64 / 4.0, z as f64 / 4.0, y as f64 / 4.0]) > 0.3
    {
        if noise.get([x as f64, y as f64, z as f64]) > 0.3 {
            match biome {
                Biome::BirchFalls => BlockType::Flower0,
                Biome::GreenGrove => BlockType::Flower1,
                Biome::DarklogForest => BlockType::Flower2,
            }
        } else {
            match biome {
                Biome::BirchFalls => BlockType::Grass0,
                Biome::GreenGrove => BlockType::Grass1,
                Biome::DarklogForest => BlockType::Grass2,
            }
        }
    } else {
        BlockType::Air
    }
}

#[allow(clippy::cast_sign_loss)]
fn generate_trees(noise: &OpenSimplex, location: [i32; 2]) -> Vec<(usize, usize)> {
    let [chunk_x, chunk_z] = location;
    let mut trees = vec![];
    for x in 0..CHUNK_WIDTH_I32 {
        for z in 0..CHUNK_DEPTH_I32 {
            let world_x = chunk_x * CHUNK_WIDTH_I32 + x;
            let world_z = chunk_z * CHUNK_DEPTH_I32 + z;

            let density_threshold =
                noise.get([f64::from(world_x) / 40.0, f64::from(world_z) / 40.0]) / 2.0 + 0.5;
            if noise.get([f64::from(world_x) * 3.0, f64::from(world_z) * 3.0]) > density_threshold {
                trees.push((x as usize, z as usize));
            }
        }
    }

    trees
}

pub fn start_meshgen(
    recv_generate: mpsc::Receiver<[i32; 2]>,
    chunkdata_arc: Arc<Mutex<ChunkDataStorage>>,
    send_chunk: mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || manage_meshgen(&recv_generate, &chunkdata_arc, &send_chunk))
}

fn manage_meshgen(
    recv_generate: &mpsc::Receiver<[i32; 2]>,
    chunkdata_arc: &Arc<Mutex<HashMap<[i32; 2], ChunkData>>>,
    send_chunk: &mpsc::SyncSender<(Vec<Vertex>, Vec<u32>, [i32; 2])>,
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
        if let Ok(chunk_location) = recv_generate.recv() {
            let generated_chunkdata = chunkdata_arc.lock().unwrap();
            let chunk_data = generated_chunkdata[&chunk_location];
            let [x, y]: [i32; 2] = chunk_location;
            let chunk_locations = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]];
            let (mesh, index_buffer) = generate_chunk_mesh(
                chunk_location,
                &chunk_data.contents,
                chunk_locations.map(|chunk| generated_chunkdata.get(&chunk).map(|c| &c.contents)),
            );
            let further_chunks = [
                [[x + 2, y], [x + 1, y + 1], [x + 1, y - 1]],
                [[x - 2, y], [x - 1, y + 1], [x - 1, y - 1]],
                [[x + 1, y + 1], [x - 1, y + 1], [x, y + 2]],
                [[x + 1, y - 1], [x - 1, y - 1], [x, y - 2]],
            ];
            for (index, (chunk_index, surrounding_chunks)) in
                chunk_locations.iter().zip(further_chunks).enumerate()
            {
                let get_chunk = |a, b| {
                    if index == a {
                        Some(&chunk_data)
                    } else {
                        generated_chunkdata.get(&surrounding_chunks[b])
                    }
                };
                if let Some(chunk) = generated_chunkdata.get(chunk_index) {
                    let (mesh, indices) = generate_chunk_mesh(
                        chunk_locations[index],
                        &chunk.contents,
                        [
                            get_chunk(1, 0),
                            get_chunk(0, usize::from(index != 1)),
                            get_chunk(3, if index < 2 { 1 } else { 2 }),
                            get_chunk(2, 2),
                        ]
                        .map(|c| c.map(|chunk| &chunk.contents)),
                    );

                    match send_chunk.try_send((mesh, indices, *chunk_index)) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Disconnected(_)) => {
                            break;
                        }
                        Err(mpsc::TrySendError::Full(v)) => waiting.push(v),
                    }
                }
            }
            match send_chunk.try_send((mesh, index_buffer, chunk_location)) {
                Ok(()) => {}
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    break;
                }
                Err(mpsc::TrySendError::Full(v)) => waiting.push(v),
            }
        }
    }
}

const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [TEXTURE_WIDTH, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, TEXTURE_WIDTH];
const BOTTOM_RIGHT: [f32; 2] = [TEXTURE_WIDTH, TEXTURE_WIDTH];

#[inline]
fn create_grass_face(
    tex_offset: [f32; 2],
    world_position: [f32; 3],
    diagonal: bool,
) -> std::array::IntoIter<Vertex, 4> {
    let [x, y, z] = world_position;
    let (add0, add1) = if diagonal {
        (FAR_CORNER, CLOSE_CORNER)
    } else {
        (CLOSE_CORNER, FAR_CORNER)
    };
    [
        Vertex(
            [x + CLOSE_CORNER, y + 1.0, z + add0],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + CLOSE_CORNER, y, z + add0],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y, z + add1],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y + 1.0, z + add1],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            1.0,
        ),
    ]
    .into_iter()
}

const FLOWER_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const STEM_CORNERS: [[f32; 2]; 4] = [
    [TEXTURE_WIDTH * (3.0 + (6.0 / 16.0)), TEXTURE_WIDTH],
    [
        TEXTURE_WIDTH * (3.0 + (6.0 / 16.0)),
        TEXTURE_WIDTH * (23.0 / 16.0),
    ],
    [TEXTURE_WIDTH * 3.0, TEXTURE_WIDTH * (23.0 / 16.0)],
    [TEXTURE_WIDTH * 3.0, TEXTURE_WIDTH],
];
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;
#[inline]
fn generate_flower(position: [f32; 3]) -> std::array::IntoIter<Vertex, 4> {
    let [x, y, z] = position;
    [
        Vertex(
            [x + CLOSE_FLOWER_CORNER, y + 1.0, z + FAR_CORNER],
            STEM_CORNERS[0].map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + CLOSE_FLOWER_CORNER, y, z + FAR_CORNER],
            STEM_CORNERS[1].map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y, z + CLOSE_FLOWER_CORNER],
            STEM_CORNERS[2].map(f16::from_f32),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y + 1.0, z + CLOSE_FLOWER_CORNER],
            STEM_CORNERS[3].map(f16::from_f32),
            1.0,
        ),
    ]
    .into_iter()
}

const GRASS_INDICES: [Index; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [Index; 6] = [0, 1, 2, 0, 2, 3];
const TOP_LEFT_WATER: [f32; 2] = [TOP_LEFT[0], TOP_LEFT[1] + HALF_TEXTURE_WIDTH];
const TOP_RIGHT_WATER: [f32; 2] = [TOP_RIGHT[0], TOP_RIGHT[1] + HALF_TEXTURE_WIDTH];
const AO_BRIGHTNESS: f32 = 0.5;

struct MeshGenerationContext<'a> {
    chunk: &'a Chunk,
    position: Vec3<usize>,
    chunk_location: [i32; 2],
    indices: &'a mut Vec<Index>,
    vertices: &'a mut Vec<Vertex>,
    surrounding_chunks: [Option<&'a Chunk>; 4],
}

impl MeshGenerationContext<'_> {
    /// Get the position vector as an array
    fn position_array(&self) -> [usize; 3] {
        self.position.into_array()
    }
    #[allow(clippy::cast_precision_loss)]
    fn worldpos_f32(&self) -> [f32; 3] {
        [
            (i32::try_from(self.position.x).unwrap() + (self.chunk_location[0] * CHUNK_WIDTH_I32))
                as f32,
            self.position.y as f32,
            (i32::try_from(self.position.z).unwrap() + (self.chunk_location[1] * CHUNK_WIDTH_I32))
                as f32,
        ]
    }
    const fn type_at_position(&self) -> BlockType {
        self.chunk[self.position.x][self.position.y][self.position.z]
    }
    fn extend_indicies(&mut self, base_indices: &[Index]) {
        let len_index = Index::try_from(self.vertices.len()).unwrap();
        self.indices
            .extend(base_indices.iter().map(|i| *i + len_index));
    }
}

pub fn generate_chunk_mesh(
    location: [i32; 2],
    chunk: &Chunk,
    surrounding_chunks: [Option<&Chunk>; 4], // north, south, east, west for... reasons...
) -> (Vec<Vertex>, Vec<Index>) {
    let (mut vertices, mut indices) = (vec![], vec![]);
    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                let mut context = MeshGenerationContext {
                    chunk,
                    position: Vec3 { x, y, z },
                    chunk_location: location,
                    indices: &mut indices,
                    vertices: &mut vertices,
                    surrounding_chunks,
                };
                let block_type = context.type_at_position();
                let tex_offsets = block_type.get_offset();
                match block_type {
                    BlockType::Air => {}
                    BlockType::Flower0 => {
                        context.extend_indicies(&FLOWER_INDICES);
                        context
                            .vertices
                            .extend(generate_flower(context.worldpos_f32()));
                    }
                    _ if block_type.is_liquid() => {
                        generate_liquid(&mut context, tex_offsets);
                    }
                    _ if block_type.is_grasslike() => {
                        let tex_offset = tex_offsets[0];
                        let world_position = context.worldpos_f32();
                        context.extend_indicies(&GRASS_INDICES);
                        context.vertices.extend(create_grass_face(
                            tex_offset,
                            world_position,
                            false,
                        ));
                        context.vertices.extend(create_grass_face(
                            tex_offset,
                            world_position,
                            true,
                        ));
                    }
                    _ => {
                        generate_solid(&mut context, tex_offsets);
                    }
                }
            }
        }
    }
    (vertices, indices)
}

const LAST_CHUNK_DEPTH: usize = CHUNK_DEPTH - 1;

#[inline]
fn generate_solid(context: &mut MeshGenerationContext, tex_offsets: [[f32; 2]; 6]) {
    let [x, y, z] = context.position_array();
    let [rel_x, y_f32, rel_z] = context.worldpos_f32();
    let z_on_end = z == CHUNK_DEPTH - 1;
    let x_on_end = x == CHUNK_WIDTH - 1;
    let y_top = y == CHUNK_HEIGHT - 1;
    let chunk = context.chunk;
    let surrounding_chunks = context.surrounding_chunks;
    let position = context.position;
    // first face
    if (z_on_end && surrounding_chunks[2].is_none_or(|other| other[x][y][0].is_transparent()))
        || (!z_on_end && chunk[x][y][z + 1].is_transparent())
    {
        let tex_offset = tex_offsets[1];

        context.extend_indicies(&QUAD_INDICES);
        context.vertices.append(&mut gen_face_1(
            rel_x,
            y_f32,
            rel_z + 1.0,
            tex_offset,
            surrounding_chunks,
            position,
            chunk,
        ));
    }
    // second face
    if (x_on_end && surrounding_chunks[0].is_none_or(|other| other[0][y][z].is_transparent()))
        || (!x_on_end && chunk[x + 1][y][z].is_transparent())
    {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&QUAD_INDICES);
        context.vertices.append(&mut gen_face_2(
            1.0 + rel_x,
            y_f32,
            rel_z,
            tex_offset,
            surrounding_chunks,
            position,
            chunk,
        ));
    }
    // third face
    if (z == 0
        && surrounding_chunks[3].is_none_or(|other| other[x][y][LAST_CHUNK_DEPTH].is_transparent()))
        || (z != 0 && chunk[x][y][z - 1].is_transparent())
    {
        let tex_offset = tex_offsets[3];
        context.extend_indicies(&QUAD_INDICES);
        context.vertices.append(&mut gen_face_3(
            rel_x, y_f32, rel_z, tex_offset, chunk, position,
        ));
    }
    // fourth face
    if (x == 0
        && surrounding_chunks[1].is_none_or(|other| other[CHUNK_WIDTH - 1][y][z].is_transparent()))
        || (x != 0 && chunk[x - 1][y][z].is_transparent())
    {
        let tex_offset = tex_offsets[4];
        context.extend_indicies(&QUAD_INDICES);
        context.vertices.append(&mut gen_face_4(
            rel_x,
            y_f32,
            rel_z,
            tex_offset,
            surrounding_chunks,
            position,
            chunk,
        ));
    }
    // top face
    if y_top || chunk[x][y + 1][z].is_transparent() {
        let tex_offset = tex_offsets[0];
        context.extend_indicies(&QUAD_INDICES);
        let yplusone = y_f32 + 1.0;
        if y_top {
            context
                .vertices
                .append(&mut top_face_no_ao(rel_x, yplusone, rel_z, tex_offset));
        } else {
            context.vertices.append(&mut top_face_ao(
                rel_x,
                yplusone,
                rel_z,
                tex_offset,
                surrounding_chunks,
                position,
                chunk,
            ));
        }
    }
    // bottom face
    if y == 0 || chunk[x][y - 1][z].is_transparent() {
        let tex_offset = tex_offsets[5];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_bottom_face(rel_x, y_f32, rel_z, tex_offset));
    }
}

fn gen_bottom_face(rel_x: f32, y_f32: f32, rel_z: f32, tex_offset: [f32; 2]) -> Vec<Vertex> {
    vec![
        Vertex(
            [1.0 + rel_x, y_f32, rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            BOTTOM_BRIGHTNESS,
        ),
        Vertex(
            [1.0 + rel_x, y_f32, 1.0 + rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            BOTTOM_BRIGHTNESS,
        ),
        Vertex(
            [rel_x, y_f32, 1.0 + rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            BOTTOM_BRIGHTNESS,
        ),
        Vertex(
            [rel_x, y_f32, rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            BOTTOM_BRIGHTNESS,
        ),
    ]
}
// This is the face that faces toward negative x
fn gen_face_4(
    rel_x: f32,
    y_f32: f32,
    rel_z: f32,
    tex_offset: [f32; 2],
    _surrounding_chunks: [Option<&Chunk>; 4],
    position: Vec3<usize>,
    chunk: &Chunk,
) -> Vec<Vertex> {
    let [x, y, z] = position.into_array();
    let up_valid = y != CHUNK_HEIGHT - 1;
    vec![
        Vertex(
            [rel_x, 1.0 + y_f32, rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            if x != 0
                && ((up_valid && !chunk[x - 1][y + 1][z].is_transparent())
                    || (z != 0
                        && (up_valid && !chunk[x - 1][y + 1][z - 1].is_transparent()
                            || !chunk[x - 1][y][z - 1].is_transparent())))
            {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, y_f32, rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            if x != 0 && y != 0 && (!chunk[x - 1][y - 1][z].is_transparent()) {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, y_f32, 1.0 + rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            if x != 0 {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, 1.0 + y_f32, 1.0 + rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            if x != 0 {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
    ]
}

fn top_face_ao(
    rel_x: f32,
    yplusone: f32,
    rel_z: f32,
    tex_offset: [f32; 2],
    surrounding_chunks: [Option<&Chunk>; 4],
    position: Vec3<usize>,
    chunk: &Chunk,
) -> Vec<Vertex> {
    let [x, y, z] = position.into_array();
    vec![
        Vertex(
            [rel_x, yplusone, rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            if (x == 0
                && surrounding_chunks[1].is_some_and(|other| {
                    !other[CHUNK_WIDTH - 1][y + 1][z].is_transparent()
                        || (z != 0 && !other[CHUNK_WIDTH - 1][y + 1][z - 1].is_transparent())
                }))
                || (x != 0
                    && (!chunk[x - 1][y + 1][z].is_transparent()
                        || (z == 0
                            && surrounding_chunks[3].is_some_and(|other| {
                                !other[x - 1][y + 1][LAST_CHUNK_DEPTH].is_transparent()
                            }))
                        || (z != 0 && !chunk[x - 1][y + 1][z - 1].is_transparent())))
                || (z == 0
                    && surrounding_chunks[3]
                        .is_some_and(|other| !other[x][y + 1][LAST_CHUNK_DEPTH].is_transparent()))
                || (z != 0 && !chunk[x][y + 1][z - 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                TOP_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, yplusone, 1.0 + rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            if (x == 0
                && surrounding_chunks[1].is_some_and(|other| {
                    !other[CHUNK_WIDTH - 1][y + 1][z].is_transparent()
                        || (z != LAST_CHUNK_DEPTH
                            && !other[CHUNK_WIDTH - 1][y + 1][z + 1].is_transparent())
                }))
                || (x != 0
                    && y != CHUNK_HEIGHT - 1
                    && (!chunk[x - 1][y + 1][z].is_transparent()
                        || ((z == LAST_CHUNK_DEPTH
                            && surrounding_chunks[2]
                                .is_some_and(|other| !other[x - 1][y + 1][0].is_transparent()))
                            || (z != LAST_CHUNK_DEPTH
                                && !chunk[x - 1][y + 1][z + 1].is_transparent()))))
                || (z == LAST_CHUNK_DEPTH
                    && surrounding_chunks[2]
                        .is_some_and(|other| !other[x][y + 1][0].is_transparent()))
                || (z != LAST_CHUNK_DEPTH
                    && y != CHUNK_HEIGHT - 1
                    && !chunk[x][y + 1][z + 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                TOP_BRIGHTNESS
            },
        ),
        Vertex(
            [1.0 + rel_x, yplusone, 1.0 + rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            if (x == CHUNK_WIDTH - 1
                && surrounding_chunks[0].is_some_and(|other| {
                    !other[0][y + 1][z].is_transparent()
                        || (z != LAST_CHUNK_DEPTH && !other[0][y + 1][z + 1].is_transparent())
                }))
                || (x != CHUNK_WIDTH - 1
                    && y != CHUNK_HEIGHT - 1
                    && (!chunk[x + 1][y + 1][z].is_transparent()
                        || (z == LAST_CHUNK_DEPTH
                            && surrounding_chunks[2]
                                .is_some_and(|other| !other[x + 1][y + 1][0].is_transparent()))
                        || (z != LAST_CHUNK_DEPTH && !chunk[x + 1][y + 1][z + 1].is_transparent())))
                || (z == LAST_CHUNK_DEPTH
                    && surrounding_chunks[2]
                        .is_some_and(|other| !other[x][y + 1][0].is_transparent()))
                || (z != LAST_CHUNK_DEPTH
                    && y != CHUNK_HEIGHT - 1
                    && !chunk[x][y + 1][z + 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                TOP_BRIGHTNESS
            },
        ),
        Vertex(
            [1.0 + rel_x, yplusone, rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            if (x == CHUNK_WIDTH - 1
                && surrounding_chunks[0].is_some_and(|other| {
                    !other[0][y + 1][z].is_transparent()
                        || (z != 0 && !other[0][y + 1][z - 1].is_transparent())
                }))
                || (x != CHUNK_WIDTH - 1
                    && y != CHUNK_HEIGHT - 1
                    && ((z == 0
                        && surrounding_chunks[3].is_some_and(|other| {
                            !other[x + 1][y + 1][LAST_CHUNK_DEPTH].is_transparent()
                        }))
                        || (z != 0 && !chunk[x + 1][y + 1][z - 1].is_transparent())
                        || !chunk[x + 1][y + 1][z].is_transparent()))
                || (z == 0
                    && surrounding_chunks[3]
                        .is_some_and(|chunk| !chunk[x][y + 1][LAST_CHUNK_DEPTH].is_transparent()))
                || (z != 0 && y != CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z - 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                TOP_BRIGHTNESS
            },
        ),
    ]
}

fn top_face_no_ao(rel_x: f32, yplusone: f32, rel_z: f32, tex_offset: [f32; 2]) -> Vec<Vertex> {
    vec![
        Vertex(
            [rel_x, yplusone, rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            TOP_BRIGHTNESS,
        ),
        Vertex(
            [rel_x, yplusone, 1.0 + rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            TOP_BRIGHTNESS,
        ),
        Vertex(
            [1.0 + rel_x, yplusone, 1.0 + rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            TOP_BRIGHTNESS,
        ),
        Vertex(
            [1.0 + rel_x, yplusone, rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            TOP_BRIGHTNESS,
        ),
    ]
}
// This is the face that faces toward negative z
fn gen_face_3(
    rel_x: f32,
    y_f32: f32,
    rel_z: f32,
    tex_offset: [f32; 2],
    chunk: &Chunk,
    position: Vec3<usize>,
) -> Vec<Vertex> {
    let [x, y, z] = position.into_array();
    vec![
        Vertex(
            [1.0 + rel_x, 1.0 + y_f32, rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            BACK_BRIGHTNESS,
        ),
        Vertex(
            [1.0 + rel_x, y_f32, rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            if y != 0 && z != 0 && !chunk[x][y - 1][z - 1].is_transparent() {
                AO_BRIGHTNESS
            } else {
                BACK_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, y_f32, rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            if y != 0 && z != 0 && !chunk[x][y - 1][z - 1].is_transparent() {
                AO_BRIGHTNESS
            } else {
                BACK_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, 1.0 + y_f32, rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            BACK_BRIGHTNESS,
        ),
    ]
}

// This face faces towards positive x
fn gen_face_2(
    xplusone: f32,
    y_f32: f32,
    rel_z: f32,
    tex_offset: [f32; 2],
    surrounding_chunks: [Option<&Chunk>; 4],
    position: Vec3<usize>,
    chunk: &Chunk,
) -> Vec<Vertex> {
    let [x, y, z] = position.into_array();
    vec![
        Vertex(
            [xplusone, 1.0 + y_f32, 1.0 + rel_z],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            if (x == CHUNK_WIDTH - 1
                && surrounding_chunks[0].is_some_and(|other| {
                    z != LAST_CHUNK_DEPTH && !other[0][y][z + 1].is_transparent()
                }))
                || (x != CHUNK_WIDTH - 1
                    && z != LAST_CHUNK_DEPTH
                    && !chunk[x + 1][y][z + 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [xplusone, y_f32, 1.0 + rel_z],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            if y != 0 && x != CHUNK_WIDTH - 1 && !chunk[x + 1][y - 1][z].is_transparent() {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [xplusone, y_f32, rel_z],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            if y != 0 && x != CHUNK_WIDTH - 1 && !chunk[x + 1][y - 1][z].is_transparent() {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
        Vertex(
            [xplusone, 1.0 + y_f32, rel_z],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            if (x == CHUNK_WIDTH - 1
                && surrounding_chunks[1]
                    .is_some_and(|other| z != 0 && !other[0][y][z - 1].is_transparent()))
                || (x != CHUNK_WIDTH - 1 && z != 0 && !chunk[x + 1][y][z - 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                SIDE_BRIGHTNESS
            },
        ),
    ]
}

// This is the face that faces positive z
fn gen_face_1(
    rel_x: f32,
    y_f32: f32,
    zplusone: f32,
    tex_offset: [f32; 2],
    surrounding_chunks: [Option<&Chunk>; 4],
    position: Vec3<usize>,
    chunk: &Chunk,
) -> Vec<Vertex> {
    let [x, y, z] = position.into_array();
    vec![
        Vertex(
            [rel_x, 1.0 + y_f32, zplusone],
            add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            if (x == 0
                && surrounding_chunks[1].is_some_and(|other| {
                    z != LAST_CHUNK_DEPTH && !other[CHUNK_WIDTH - 1][y][z + 1].is_transparent()
                }))
                || (x != 0 && z != LAST_CHUNK_DEPTH && !chunk[x - 1][y][z + 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                FRONT_BRIGHTNESS
            },
        ),
        Vertex(
            [rel_x, y_f32, zplusone],
            add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            if y != 0
                && ((z != LAST_CHUNK_DEPTH && !chunk[x][y - 1][z + 1].is_transparent())
                    || (z == LAST_CHUNK_DEPTH
                        && surrounding_chunks[2]
                            .is_some_and(|other| !other[x][y - 1][0].is_transparent())))
            {
                AO_BRIGHTNESS
            } else {
                FRONT_BRIGHTNESS
            },
        ),
        Vertex(
            [1.0 + rel_x, y_f32, zplusone],
            add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            if y != 0
                && ((z != LAST_CHUNK_DEPTH && !chunk[x][y - 1][z + 1].is_transparent())
                    || (z == LAST_CHUNK_DEPTH
                        && surrounding_chunks[2]
                            .is_some_and(|other| !other[x][y - 1][0].is_transparent())))
            {
                AO_BRIGHTNESS
            } else {
                FRONT_BRIGHTNESS
            },
        ),
        Vertex(
            [1.0 + rel_x, 1.0 + y_f32, zplusone],
            add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            if (x == CHUNK_WIDTH - 1
                && surrounding_chunks[0].is_some_and(|other| {
                    z != LAST_CHUNK_DEPTH && !other[0][y][z + 1].is_transparent()
                }))
                || (x != CHUNK_WIDTH - 1
                    && z != LAST_CHUNK_DEPTH
                    && !chunk[x + 1][y][z + 1].is_transparent())
            {
                AO_BRIGHTNESS
            } else {
                FRONT_BRIGHTNESS
            },
        ),
    ]
}

#[inline]
fn generate_liquid(context: &mut MeshGenerationContext, tex_offsets: [[f32; 2]; 6]) {
    let [x, y, z] = context.position_array();
    let [rel_x, y_f32, rel_z] = context.worldpos_f32();
    let yplusoff = y_f32 + 0.5;
    let chunk = context.chunk;
    if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
        let tex_offset = tex_offsets[0];
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex(
                [rel_x, yplusoff, rel_z],
                add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, yplusoff, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, yplusoff, rel_z],
                add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
        ]);
    }
    if y != 0 && !(chunk[x][y - 1][z].is_liquid() || chunk[x][y - 1][z].is_solid()) {
        let tex_offset = tex_offsets[5];
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex(
                [rel_x, y_f32, rel_z],
                add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, rel_z],
                add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                TOP_BRIGHTNESS,
            ),
        ]);
    }
    if (z == LAST_CHUNK_DEPTH
        && context.surrounding_chunks[2]
            .is_none_or(|other| other[x][y][0].is_transparent() && !other[x][y][0].is_liquid()))
        || (z != LAST_CHUNK_DEPTH
            && (chunk[x][y][z + 1].is_transparent() && !chunk[x][y][z + 1].is_liquid()))
    {
        let tex_offset = tex_offsets[1];
        context.extend_indicies(&BIDIR_INDICES);
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    if (x == CHUNK_WIDTH - 1
        && context.surrounding_chunks[0]
            .is_none_or(|other| other[0][y][z].is_transparent() && !other[0][y][z].is_liquid()))
        || (x != CHUNK_WIDTH - 1
            && (chunk[x + 1][y][z].is_transparent() && !chunk[x + 1][y][z].is_liquid()))
    {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&BIDIR_INDICES);
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            context.vertices.append(&mut vec![
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            let yplusoff = y_f32 + 1.0;
            context.vertices.append(&mut vec![
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    let z_zero = z == 0;
    if (z_zero
        && context.surrounding_chunks[3].is_none_or(|other| {
            other[x][y][LAST_CHUNK_DEPTH].is_transparent()
                && !other[x][y][LAST_CHUNK_DEPTH].is_liquid()
        }))
        || (!z_zero && chunk[x][y][z - 1].is_transparent() && !chunk[x][y][z - 1].is_liquid())
    {
        let tex_offset = tex_offsets[1];
        context.extend_indicies(&BIDIR_INDICES);
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, y_f32 + 1.0, rel_z],
                    add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    if (x == 0
        && context.surrounding_chunks[1].is_none_or(|other| {
            other[CHUNK_WIDTH - 1][y][z].is_transparent()
                && !other[CHUNK_WIDTH - 1][y][z].is_liquid()
        }))
        || (x != 0 && chunk[x - 1][y][z].is_transparent() && !chunk[x - 1][y][z].is_liquid())
    {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&BIDIR_INDICES);
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            let yplusoff = y_f32 + 1.0;
            context.vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
}
