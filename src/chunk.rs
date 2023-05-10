use std::{collections::HashMap, f32::consts::FRAC_1_SQRT_2};

use noise::{NoiseFn, OpenSimplex};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{Vertex, MAX_DEPTH};
#[cfg(target_os = "windows")]
pub const CHUNK_WIDTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_WIDTH: usize = 32;
const CHUNK_WIDTH_I32: i32 = CHUNK_WIDTH as i32;
pub const CHUNK_HEIGHT: usize = 256;
#[cfg(target_os = "windows")]
pub const CHUNK_DEPTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_DEPTH: usize = 32;
const CHUNK_DEPTH_I32: i32 = CHUNK_DEPTH as i32;

const TEXTURE_WIDTH: f32 = 1.0 / 16.0;
const HALF_TEXTURE_WIDTH: f32 = TEXTURE_WIDTH / 2.0;

type Chunk = [[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];

#[derive(Clone, Copy)]
enum Biome {
    BirchFalls,
    GreenGrove,
    DarklogForest,
    // PineHills,
    // SnowDesert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
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
}

impl BlockType {
    fn get_offset(self) -> [[f32; 2]; 6] {
        match self {
            Self::Stone => [[0., TEXTURE_WIDTH * 8.]; 6],
            Self::GrassBlock0 => [
                [0., TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [TEXTURE_WIDTH, TEXTURE_WIDTH],
                [0., 0.],
            ],
            Self::GrassBlock1 => [
                [0., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 2.],
                [0., 0.],
            ],
            Self::GrassBlock2 => [
                [0., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH, TEXTURE_WIDTH * 3.],
                [0., 0.],
            ],
            Self::BirchWood => [
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH],
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH],
            ],
            Self::Wood => [
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 2.],
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH * 2.],
            ],
            Self::DarkWood => [
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH * 6., TEXTURE_WIDTH * 3.],
                [TEXTURE_WIDTH * 7., TEXTURE_WIDTH * 3.],
            ],
            Self::BirchLeaf => [[TEXTURE_WIDTH * 5., TEXTURE_WIDTH]; 6],
            Self::Leaf => [[TEXTURE_WIDTH * 5., TEXTURE_WIDTH * 2.]; 6],
            Self::DarkLeaf => [[TEXTURE_WIDTH * 5., TEXTURE_WIDTH * 3.]; 6],
            Self::Grass0 => [[TEXTURE_WIDTH * 2., TEXTURE_WIDTH]; 6],
            Self::Grass1 => [[TEXTURE_WIDTH * 2., TEXTURE_WIDTH * 2.]; 6],
            Self::Grass2 => [[TEXTURE_WIDTH * 2., TEXTURE_WIDTH * 3.]; 6],
            Self::Flower0 => [[TEXTURE_WIDTH * 3., TEXTURE_WIDTH]; 6],
            Self::Flower1 => [[TEXTURE_WIDTH * 3., TEXTURE_WIDTH * 2.]; 6],
            Self::Flower2 => [[TEXTURE_WIDTH * 3., TEXTURE_WIDTH * 3.]; 6],
            Self::Water => [
                [TEXTURE_WIDTH * 4., 0.],
                [TEXTURE_WIDTH * 5., 0.],
                [TEXTURE_WIDTH * 5., 0.],
                [TEXTURE_WIDTH * 5., 0.],
                [TEXTURE_WIDTH * 5., 0.],
                [TEXTURE_WIDTH * 4., 0.],
            ],
            Self::Sand => [[TEXTURE_WIDTH * 9., 0.]; 6],
            Self::Air => panic!("This is not supposed to be called!"),
        }
    }

    pub const fn is_solid(self) -> bool {
        !(matches!(self, Self::Air | Self::Water) || self.is_grasslike())
    }

    pub const fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Air | Self::Leaf | Self::BirchLeaf | Self::DarkLeaf
        ) || self.is_liquid()
            || self.is_grasslike()
    }

    pub const fn is_liquid(self) -> bool {
        matches!(self, Self::Water)
    }

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

#[serde_as]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkData {
    // pub location: [i32; 2],
    #[serde_as(as = "[[[_; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]")]
    pub contents: Chunk,
}

const MAX_DISTANCE_X: i32 = MAX_DEPTH as i32 / CHUNK_WIDTH_I32 + 1;
const MAX_DISTANCE_Y: i32 = MAX_DEPTH as i32 / CHUNK_DEPTH_I32 + 1;
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
) -> Option<ChunkData> {
    let chunk_x = x.div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = z.div_euclid(CHUNK_DEPTH_I32);
    generated_chunks.get(&[chunk_x, chunk_z]).copied()
}

pub fn get_block(
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    x: i32,
    y: i32,
    z: i32,
) -> Option<BlockType> {
    let chunk = chunk_at_block(generated_chunks, x, z)?;
    let x = (x - (x.div_euclid(CHUNK_WIDTH_I32) * CHUNK_WIDTH_I32)) as usize;
    let z = (z - (z.div_euclid(CHUNK_DEPTH_I32) * CHUNK_DEPTH_I32)) as usize;
    if y >= 0 && (y as usize) < CHUNK_HEIGHT {
        Some(chunk.contents[x][y as usize][z])
    } else {
        None
    }
}

pub fn get_nearest_chunk_location(
    x: f32,
    z: f32,
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
) -> Option<[i32; 2]> {
    let chunk_x = (x as i32).div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = (z as i32).div_euclid(CHUNK_WIDTH_I32);
    let length = |a, b| (a * a + b * b);
    (-MAX_DISTANCE_X..=MAX_DISTANCE_X)
        .flat_map(|i| {
            (-MAX_DISTANCE_Y..=MAX_DISTANCE_Y).filter_map(move |j| {
                let distance = length(i, j);
                let location = [i + chunk_x, j + chunk_z];
                if distance <= (MAX_DEPTH * MAX_DEPTH) as i32
                    && !generated_chunks.contains_key(&location)
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
    assert_eq!(add_arrs([1.0, 2.0], [3.0, 4.0]), [4.0, 6.0])
}

#[inline]
fn add_arrs(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

const WATER_HEIGHT: usize = 20;
const LARGE_SCALE: f64 = 100.0;
const SMALL_SCALE: f64 = 25.0;
const LARGE_HEIGHT: f64 = 40.0;
const TERRAIN_HEIGHT: f64 = 0.8;
const BIOME_SCALE: f64 = 250.0;

pub fn generate(noise: &OpenSimplex, chunk_location: [i32; 2]) -> Chunk {
    let heightmap = generate_heightmap(noise, chunk_location);
    let biomemap = generate_biomemap(noise, chunk_location);
    let mut contents = [[[BlockType::Air; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH];
    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                let biome = biomemap[x][z];
                contents[x][y][z] = determine_type(heightmap, x, y, z, biome, noise);
            }
        }
    }
    contents
}

fn generate_biomemap(
    noise: &OpenSimplex,
    chunk_location: [i32; 2],
) -> [[Biome; CHUNK_DEPTH]; CHUNK_WIDTH] {
    let mut biomemap = [[Biome::BirchFalls; CHUNK_DEPTH]; CHUNK_WIDTH];

    for (x, row) in biomemap.iter_mut().enumerate() {
        for (z, item) in row.iter_mut().enumerate() {
            let v = noise_at(noise, x as i32, z as i32, chunk_location, BIOME_SCALE, 18.9);
            let biome = if v > 0.2 {
                Biome::DarklogForest
            } else if v > 0.0 {
                Biome::GreenGrove
            } else {
                Biome::BirchFalls
            };
            *item = biome;
        }
    }

    biomemap
}

fn generate_heightmap(
    noise: &OpenSimplex,
    chunk_location: [i32; 2],
) -> [[i32; CHUNK_DEPTH]; CHUNK_WIDTH] {
    let mut heightmap = [[0; CHUNK_DEPTH]; CHUNK_WIDTH];
    for (x, row) in heightmap.iter_mut().enumerate() {
        for (z, item) in row.iter_mut().enumerate() {
            let large_noise = noise_at(noise, x as i32, z as i32, chunk_location, LARGE_SCALE, 0.0);
            let small_noise =
                noise_at(noise, x as i32, z as i32, chunk_location, SMALL_SCALE, 10.0);
            let height =
                (large_noise + TERRAIN_HEIGHT).mul_add(LARGE_HEIGHT, small_noise * 10.0) as i32;
            *item = height;
        }
    }
    heightmap
}

fn determine_type(
    heightmap: [[i32; CHUNK_DEPTH]; CHUNK_WIDTH],
    x: usize,
    y: usize,
    z: usize,
    biome: Biome,
    noise: &OpenSimplex,
) -> BlockType {
    let y_i32 = y as i32;
    if y_i32 < heightmap[x][z] {
        BlockType::Stone
    } else if y_i32 == heightmap[x][z] {
        if heightmap[x][z] + 1 > WATER_HEIGHT as i32 {
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
    } else if y_i32 > heightmap[x][z]
        && y_i32 <= heightmap[x][z] + 5
        && heightmap[x][z] >= WATER_HEIGHT as i32
    {
        if noise.get([x as f64, f64::from(heightmap[x][z]), z as f64]) > 0.4 {
            if y_i32 == heightmap[x][z] + 5 {
                match biome {
                    Biome::BirchFalls => BlockType::BirchLeaf,
                    Biome::GreenGrove => BlockType::Leaf,
                    Biome::DarklogForest => BlockType::DarkLeaf,
                }
            } else {
                match biome {
                    Biome::BirchFalls => BlockType::BirchWood,
                    Biome::GreenGrove => BlockType::Wood,
                    Biome::DarklogForest => BlockType::DarkWood,
                }
            }
        } else if y_i32 == heightmap[x][z] + 1
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
    } else {
        BlockType::Air
    }
}

const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [TEXTURE_WIDTH, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, TEXTURE_WIDTH];
const BOTTOM_RIGHT: [f32; 2] = [TEXTURE_WIDTH, TEXTURE_WIDTH];

fn create_grass_face(
    tex_offset: [f32; 2],
    position: (f32, f32, f32),
    diagonal: bool,
) -> Vec<Vertex> {
    let (x, y, z) = position;
    let (add0, add1) = if diagonal {
        (FAR_CORNER, CLOSE_CORNER)
    } else {
        (CLOSE_CORNER, FAR_CORNER)
    };
    vec![
        Vertex(
            [x + CLOSE_CORNER, y + 1.0, z + add0],
            add_arrs(TOP_LEFT, tex_offset),
            1.0,
        ),
        Vertex(
            [x + CLOSE_CORNER, y, z + add0],
            add_arrs(BOTTOM_LEFT, tex_offset),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y, z + add1],
            add_arrs(BOTTOM_RIGHT, tex_offset),
            1.0,
        ),
        Vertex(
            [x + FAR_CORNER, y + 1.0, z + add1],
            add_arrs(TOP_RIGHT, tex_offset),
            1.0,
        ),
    ]
}

const GRASS_INDICES: [u32; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const TOP_LEFT_WATER: [f32; 2] = [TOP_LEFT[0], TOP_LEFT[1] + HALF_TEXTURE_WIDTH];
const TOP_RIGHT_WATER: [f32; 2] = [TOP_RIGHT[0], TOP_RIGHT[1] + HALF_TEXTURE_WIDTH];
const AO_BRIGHTNESS: f32 = 0.5;

pub fn generate_chunk_mesh(
    location: [i32; 2],
    chunk: &Chunk,
    surrounding_chunks: [Option<&ChunkData>; 4], // north, south, east, west for... reasons...
) -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = (vec![], vec![]);
    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                generate_block_mesh(
                    chunk,
                    (x, y, z),
                    location,
                    &mut indices,
                    &mut vertices,
                    surrounding_chunks,
                );
            }
        }
    }
    (vertices, indices)
}

fn generate_block_mesh(
    chunk: &[[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
    position: (usize, usize, usize),
    chunk_location: [i32; 2],
    indices: &mut Vec<u32>,
    vertices: &mut Vec<Vertex>,
    surrounding_chunks: [Option<&ChunkData>; 4],
) {
    let (x, y, z) = position;
    if chunk[x][y][z] == BlockType::Air {
    } else if chunk[x][y][z].is_grasslike() {
        let tex_offset = chunk[x][y][z].get_offset()[0];
        let x = (x as i32 + (chunk_location[0] * CHUNK_WIDTH_I32)) as f32;
        let z = (z as i32 + (chunk_location[1] * CHUNK_DEPTH_I32)) as f32;
        let y = y as f32;
        indices.extend(GRASS_INDICES.map(|i| i + vertices.len() as u32));
        vertices.append(&mut create_grass_face(tex_offset, (x, y, z), false));
        vertices.append(&mut create_grass_face(tex_offset, (x, y, z), true));
    } else if chunk[x][y][z].is_liquid() {
        generate_water(
            chunk,
            position,
            chunk_location,
            indices,
            vertices,
            surrounding_chunks,
        );
    } else {
        generate_solid(
            chunk,
            position,
            chunk_location,
            surrounding_chunks,
            indices,
            vertices,
        );
    }
}

fn generate_solid(
    chunk: &Chunk,
    position: (usize, usize, usize),
    location: [i32; 2],
    surrounding_chunks: [Option<&ChunkData>; 4],
    indices: &mut Vec<u32>,
    vertices: &mut Vec<Vertex>,
) {
    let (x, y, z) = position;
    let tex_offsets = chunk[x][y][z].get_offset();
    let rel_x = (x as i32 + (location[0] * CHUNK_WIDTH_I32)) as f32;
    let rel_z = (z as i32 + (location[1] * CHUNK_DEPTH_I32)) as f32;
    let y_f32 = y as f32;
    // first face
    if (z == CHUNK_DEPTH - 1
        && surrounding_chunks[2].map_or(true, |chunk| chunk.contents[x][y][0].is_transparent()))
        || (z != CHUNK_DEPTH - 1 && chunk[x][y][z + 1].is_transparent())
    {
        let tex_offset = tex_offsets[1];
        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        let zplusone = 1.0 + rel_z;
        vertices.append(&mut vec![
            Vertex(
                [rel_x, 1.0 + y_f32, zplusone],
                add_arrs(TOP_LEFT, tex_offset),
                if (x == 0
                    && surrounding_chunks[1].map_or(false, |chunk| {
                        z != CHUNK_DEPTH - 1
                            && !chunk.contents[CHUNK_WIDTH - 1][y][z + 1].is_transparent()
                    }))
                    || (x != 0 && z != CHUNK_DEPTH - 1 && !chunk[x - 1][y][z + 1].is_transparent())
                {
                    AO_BRIGHTNESS
                } else {
                    FRONT_BRIGHTNESS
                },
            ),
            Vertex(
                [rel_x, y_f32, zplusone],
                add_arrs(BOTTOM_LEFT, tex_offset),
                if y != 0
                    && ((z != CHUNK_DEPTH - 1 && !chunk[x][y - 1][z + 1].is_transparent())
                        || (z == CHUNK_DEPTH - 1
                            && surrounding_chunks[2].map_or(false, |chunk| {
                                !chunk.contents[x][y - 1][0].is_transparent()
                            })))
                {
                    AO_BRIGHTNESS
                } else {
                    FRONT_BRIGHTNESS
                },
            ),
            Vertex(
                [1.0 + rel_x, y_f32, zplusone],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                if y != 0
                    && ((z != CHUNK_DEPTH - 1 && !chunk[x][y - 1][z + 1].is_transparent())
                        || (z == CHUNK_DEPTH - 1
                            && surrounding_chunks[2].map_or(false, |chunk| {
                                !chunk.contents[x][y - 1][0].is_transparent()
                            })))
                {
                    AO_BRIGHTNESS
                } else {
                    FRONT_BRIGHTNESS
                },
            ),
            Vertex(
                [1.0 + rel_x, 1.0 + y_f32, zplusone],
                add_arrs(TOP_RIGHT, tex_offset),
                if (x == CHUNK_WIDTH - 1
                    && surrounding_chunks[0].map_or(false, |chunk| {
                        z != CHUNK_DEPTH - 1 && !chunk.contents[0][y][z + 1].is_transparent()
                    }))
                    || (x != CHUNK_WIDTH - 1
                        && z != CHUNK_DEPTH - 1
                        && !chunk[x + 1][y][z + 1].is_transparent())
                {
                    AO_BRIGHTNESS
                } else {
                    FRONT_BRIGHTNESS
                },
            ),
        ]);
    }
    // second face
    if (x == CHUNK_WIDTH - 1
        && surrounding_chunks[0].map_or(true, |chunk| chunk.contents[0][y][z].is_transparent()))
        || (x != CHUNK_WIDTH - 1 && chunk[x + 1][y][z].is_transparent())
    {
        let tex_offset = tex_offsets[2];
        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        let xplusone = 1.0 + rel_x;
        vertices.append(&mut vec![
            Vertex(
                [xplusone, 1.0 + y_f32, 1.0 + rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                if (x == CHUNK_WIDTH - 1
                    && surrounding_chunks[0].map_or(false, |chunk| {
                        z != CHUNK_DEPTH - 1 && !chunk.contents[0][y][z + 1].is_transparent()
                    }))
                    || (x != CHUNK_WIDTH - 1
                        && z != CHUNK_DEPTH - 1
                        && !chunk[x + 1][y][z + 1].is_transparent())
                {
                    AO_BRIGHTNESS
                } else {
                    SIDE_BRIGHTNESS
                },
            ),
            Vertex(
                [xplusone, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                if y != 0 && x != CHUNK_WIDTH - 1 && !chunk[x + 1][y - 1][z].is_transparent() {
                    AO_BRIGHTNESS
                } else {
                    SIDE_BRIGHTNESS
                },
            ),
            Vertex(
                [xplusone, y_f32, rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                if y != 0 && x != CHUNK_WIDTH - 1 && !chunk[x + 1][y - 1][z].is_transparent() {
                    AO_BRIGHTNESS
                } else {
                    SIDE_BRIGHTNESS
                },
            ),
            Vertex(
                [xplusone, 1.0 + y_f32, rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                if (x == CHUNK_WIDTH - 1
                    && surrounding_chunks[1].map_or(false, |chunk| {
                        z != 0 && !chunk.contents[0][y][z - 1].is_transparent()
                    }))
                    || (x != CHUNK_WIDTH - 1 && z != 0 && !chunk[x + 1][y][z - 1].is_transparent())
                {
                    AO_BRIGHTNESS
                } else {
                    SIDE_BRIGHTNESS
                },
            ),
        ]);
    }
    // third face
    if (z == 0
        && surrounding_chunks[3].map_or(true, |chunk| {
            chunk.contents[x][y][CHUNK_DEPTH - 1].is_transparent()
        }))
        || (z != 0 && chunk[x][y][z - 1].is_transparent())
    {
        let tex_offset = tex_offsets[3];
        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        vertices.append(&mut vec![
            Vertex(
                [1.0 + rel_x, 1.0 + y_f32, rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                BACK_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                if y != 0 && z != 0 && !chunk[x][y - 1][z - 1].is_transparent() {
                    AO_BRIGHTNESS
                } else {
                    BACK_BRIGHTNESS
                },
            ),
            Vertex(
                [rel_x, y_f32, rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                if y != 0 && z != 0 && !chunk[x][y - 1][z - 1].is_transparent() {
                    AO_BRIGHTNESS
                } else {
                    BACK_BRIGHTNESS
                },
            ),
            Vertex(
                [rel_x, 1.0 + y_f32, rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                BACK_BRIGHTNESS,
            ),
        ]);
    }
    // fourth face
    if (x == 0
        && surrounding_chunks[1].map_or(true, |chunk| {
            chunk.contents.last().unwrap()[y][z].is_transparent()
        }))
        || (x != 0 && chunk[x - 1][y][z].is_transparent())
    {
        let tex_offset = tex_offsets[4];
        indices.extend(&mut QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        vertices.append(&mut vec![
            Vertex(
                [rel_x, 1.0 + y_f32, rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                SIDE_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                SIDE_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                SIDE_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, 1.0 + y_f32, 1.0 + rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                SIDE_BRIGHTNESS,
            ),
        ]);
    }
    // top face
    if y == CHUNK_HEIGHT - 1 || chunk[x][y + 1][z].is_transparent() {
        let tex_offset = tex_offsets[0];
        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        let yplusone = y_f32 + 1.0;
        if y >= CHUNK_HEIGHT - 2 {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusone, rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, yplusone, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusone, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusone, rel_z],
                    add_arrs(TOP_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusone, rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    if (x == 0
                        && surrounding_chunks[1].map_or(false, |chunk| {
                            !chunk.contents[CHUNK_WIDTH - 1][y + 1][z].is_transparent()
                                || (z != 0
                                    && !chunk.contents[CHUNK_WIDTH - 1][y + 1][z - 1]
                                        .is_transparent())
                        }))
                        || (x != 0
                            && (!chunk[x - 1][y + 1][z].is_transparent()
                                || (z == 0
                                    && surrounding_chunks[3].map_or(false, |chunk| {
                                        !chunk.contents[x - 1][y + 1][CHUNK_DEPTH - 1]
                                            .is_transparent()
                                    }))
                                || (z != 0 && !chunk[x - 1][y + 1][z - 1].is_transparent())))
                        || (z == 0
                            && surrounding_chunks[3].map_or(false, |chunk| {
                                !chunk.contents[x][y + 1][CHUNK_DEPTH - 1].is_transparent()
                            }))
                        || (z != 0 && !chunk[x][y + 1][z - 1].is_transparent())
                    {
                        AO_BRIGHTNESS
                    } else {
                        TOP_BRIGHTNESS
                    },
                ),
                Vertex(
                    [rel_x, yplusone, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    if (x == 0
                        && surrounding_chunks[1].map_or(false, |chunk| {
                            !chunk.contents[CHUNK_WIDTH - 1][y + 1][z].is_transparent()
                                || (z != CHUNK_DEPTH - 1
                                    && !chunk.contents[CHUNK_WIDTH - 1][y + 1][z + 1]
                                        .is_transparent())
                        }))
                        || (x != 0
                            && y != CHUNK_HEIGHT - 1
                            && (!chunk[x - 1][y + 1][z].is_transparent()
                                || ((z == CHUNK_DEPTH - 1
                                    && surrounding_chunks[2].map_or(false, |chunk| {
                                        !chunk.contents[x - 1][y + 1][0].is_transparent()
                                    }))
                                    || (z != CHUNK_DEPTH - 1
                                        && !chunk[x - 1][y + 1][z + 1].is_transparent()))))
                        || (z == CHUNK_DEPTH - 1
                            && surrounding_chunks[2].map_or(false, |chunk| {
                                !chunk.contents[x][y + 1][0].is_transparent()
                            }))
                        || (z != CHUNK_DEPTH - 1
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
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    if (x == CHUNK_WIDTH - 1
                        && surrounding_chunks[0].map_or(false, |chunk| {
                            !chunk.contents[0][y + 1][z].is_transparent()
                                || (z != CHUNK_DEPTH - 1
                                    && !chunk.contents[0][y + 1][z + 1].is_transparent())
                        }))
                        || (x != CHUNK_WIDTH - 1
                            && y != CHUNK_HEIGHT - 1
                            && (!chunk[x + 1][y + 1][z].is_transparent()
                                || (z == CHUNK_DEPTH - 1
                                    && surrounding_chunks[2].map_or(false, |chunk| {
                                        !chunk.contents[x + 1][y + 1][0].is_transparent()
                                    }))
                                || (z != CHUNK_DEPTH - 1
                                    && !chunk[x + 1][y + 1][z + 1].is_transparent())))
                        || (z == CHUNK_DEPTH - 1
                            && surrounding_chunks[2].map_or(false, |chunk| {
                                !chunk.contents[x][y + 1][0].is_transparent()
                            }))
                        || (z != CHUNK_DEPTH - 1
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
                    add_arrs(TOP_RIGHT, tex_offset),
                    if (x == CHUNK_WIDTH - 1
                        && surrounding_chunks[0].map_or(false, |chunk| {
                            !chunk.contents[0][y + 1][z].is_transparent()
                                || (z != 0 && !chunk.contents[0][y + 1][z - 1].is_transparent())
                        }))
                        || (x != CHUNK_WIDTH - 1
                            && y != CHUNK_HEIGHT - 1
                            && ((z == 0
                                && surrounding_chunks[3].map_or(false, |chunk| {
                                    !chunk.contents[x + 1][y + 1][CHUNK_DEPTH - 1].is_transparent()
                                }))
                                || (z != 0 && !chunk[x + 1][y + 1][z - 1].is_transparent())
                                || !chunk[x + 1][y + 1][z].is_transparent()))
                        || (z == 0
                            && surrounding_chunks[3].map_or(false, |chunk| {
                                !chunk.contents[x][y + 1][CHUNK_DEPTH - 1].is_transparent()
                            }))
                        || (z != 0
                            && y != CHUNK_HEIGHT - 1
                            && !chunk[x][y + 1][z - 1].is_transparent())
                    {
                        AO_BRIGHTNESS
                    } else {
                        TOP_BRIGHTNESS
                    },
                ),
            ]);
        }
    }
    // bottom face
    if y == 0 || chunk[x][y - 1][z].is_transparent() {
        let tex_offset = tex_offsets[5];
        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
        vertices.append(&mut vec![
            // start of bottom
            Vertex(
                [1.0 + rel_x, y_f32, rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                BOTTOM_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                BOTTOM_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                BOTTOM_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                BOTTOM_BRIGHTNESS,
            ),
        ]);
    }
}

fn generate_water(
    chunk: &Chunk,
    position: (usize, usize, usize),
    location: [i32; 2],
    indices: &mut Vec<u32>,
    vertices: &mut Vec<Vertex>,
    surrounding_chunks: [Option<&ChunkData>; 4],
) {
    let (x, y, z) = position;
    let tex_offsets = chunk[x][y][z].get_offset();
    let rel_x = (x as i32 + (location[0] * CHUNK_WIDTH_I32)) as f32;
    let rel_z = (z as i32 + (location[1] * CHUNK_DEPTH_I32)) as f32;
    let y_f32 = y as f32;
    let yplusoff = y_f32 + 0.5;
    if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
        let tex_offset = tex_offsets[0];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        vertices.append(&mut vec![
            Vertex(
                [rel_x, yplusoff, rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, yplusoff, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, yplusoff, rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                TOP_BRIGHTNESS,
            ),
        ]);
    }
    if y != 0 && !(chunk[x][y - 1][z].is_liquid() || chunk[x][y - 1][z].is_solid()) {
        let tex_offset = tex_offsets[5];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        vertices.append(&mut vec![
            Vertex(
                [rel_x, y_f32, rel_z],
                add_arrs(TOP_LEFT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_LEFT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, 1.0 + rel_z],
                add_arrs(BOTTOM_RIGHT, tex_offset),
                TOP_BRIGHTNESS,
            ),
            Vertex(
                [1.0 + rel_x, y_f32, rel_z],
                add_arrs(TOP_RIGHT, tex_offset),
                TOP_BRIGHTNESS,
            ),
        ]);
    }
    if (z == CHUNK_DEPTH - 1
        && surrounding_chunks[2].map_or(true, |chunk| {
            chunk.contents[x][y][0].is_transparent() && !chunk.contents[x][y][0].is_liquid()
        }))
        || (z != CHUNK_DEPTH - 1
            && (chunk[x][y][z + 1].is_transparent() && !chunk[x][y][z + 1].is_liquid()))
    {
        let tex_offset = tex_offsets[1];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    if (x == CHUNK_WIDTH - 1
        && surrounding_chunks[0].map_or(true, |chunk| {
            chunk.contents[0][y][z].is_transparent() && !chunk.contents[0][y][z].is_liquid()
        }))
        || (x != CHUNK_WIDTH - 1
            && (chunk[x + 1][y][z].is_transparent() && !chunk[x + 1][y][z].is_liquid()))
    {
        let tex_offset = tex_offsets[2];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            vertices.append(&mut vec![
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            let yplusoff = y_f32 + 1.0;
            vertices.append(&mut vec![
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    if (z == 0
        && surrounding_chunks[3].map_or(true, |chunk| {
            chunk.contents[x][y][CHUNK_DEPTH - 1].is_transparent()
                && !chunk.contents[x][y][CHUNK_DEPTH - 1].is_liquid()
        }))
        || (z != 0 && chunk[x][y][z - 1].is_transparent() && !chunk[x][y][z - 1].is_liquid())
    {
        let tex_offset = tex_offsets[1];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, yplusoff, rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, y_f32 + 1.0, rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    add_arrs(TOP_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
    if (x == 0
        && surrounding_chunks[1].map_or(true, |chunk| {
            chunk.contents[CHUNK_WIDTH - 1][y][z].is_transparent()
                && !chunk.contents[CHUNK_WIDTH - 1][y][z].is_liquid()
        }))
        || (x != 0 && chunk[x - 1][y][z].is_transparent() && !chunk[x - 1][y][z].is_liquid())
    {
        let tex_offset = tex_offsets[2];
        indices.extend(BIDIR_INDICES.iter().map(|i| *i + vertices.len() as u32));
        if y < CHUNK_HEIGHT - 1 && !chunk[x][y + 1][z].is_liquid() {
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT_WATER, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        } else {
            let yplusoff = y_f32 + 1.0;
            vertices.append(&mut vec![
                Vertex(
                    [rel_x, yplusoff, rel_z],
                    add_arrs(TOP_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, rel_z],
                    add_arrs(BOTTOM_LEFT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, y_f32, 1.0 + rel_z],
                    add_arrs(BOTTOM_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
                Vertex(
                    [rel_x, yplusoff, 1.0 + rel_z],
                    add_arrs(TOP_RIGHT, tex_offset),
                    TOP_BRIGHTNESS,
                ),
            ]);
        }
    }
}
