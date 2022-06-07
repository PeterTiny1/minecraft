use std::{collections::HashMap, f32::consts::FRAC_1_SQRT_2};

use noise::{NoiseFn, OpenSimplex};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{Vertex, MAX_DEPTH};
#[cfg(target_os = "windows")]
pub const CHUNK_WIDTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_HEIGHT: usize = 256;
#[cfg(target_os = "windows")]
pub const CHUNK_DEPTH: usize = 16;
#[cfg(not(target_os = "windows"))]
pub const CHUNK_DEPTH: usize = 32;

const TEXTURE_WIDTH: f32 = 0.25;

const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [TEXTURE_WIDTH, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, TEXTURE_WIDTH];
const BOTTOM_RIGHT: [f32; 2] = [TEXTURE_WIDTH, TEXTURE_WIDTH];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rotation {
    Up,
    Down,
    North,
    East,
    South,
    West,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    Air,
    Stone,
    GrassBlock,
    Grass,
    Wood(Rotation),
    Leaf,
}

impl BlockType {
    fn get_offset(&self) -> [[f32; 2]; 6] {
        match self {
            BlockType::Stone => [[0.0, 0.0]; 6],
            BlockType::GrassBlock => [
                [TEXTURE_WIDTH, 0.0],
                [TEXTURE_WIDTH * 2.0, 0.0],
                [TEXTURE_WIDTH * 2.0, 0.0],
                [TEXTURE_WIDTH * 2.0, 0.0],
                [TEXTURE_WIDTH * 2.0, 0.0],
                [0.0, 0.0],
            ],
            BlockType::Wood(rotation) => match rotation {
                Rotation::Up => [
                    [0.0, TEXTURE_WIDTH],
                    [TEXTURE_WIDTH, TEXTURE_WIDTH],
                    [TEXTURE_WIDTH, TEXTURE_WIDTH],
                    [TEXTURE_WIDTH, TEXTURE_WIDTH],
                    [TEXTURE_WIDTH, TEXTURE_WIDTH],
                    [0.0, TEXTURE_WIDTH],
                ],
                Rotation::Down => todo!(),
                Rotation::North => todo!(),
                Rotation::East => todo!(),
                Rotation::South => todo!(),
                Rotation::West => todo!(),
            },
            BlockType::Leaf => [[TEXTURE_WIDTH * 2.0, TEXTURE_WIDTH]; 6],
            BlockType::Grass => [[TEXTURE_WIDTH * 3.0, 0.0]; 6],
            _ => panic!("This is not supposed to be called!"),
        }
    }
    pub fn is_solid(&self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Grass | BlockType::Leaf)
    }
}

#[serde_as]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ChunkData {
    // pub location: [i32; 2],
    #[serde_as(as = "[[[_; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH]")]
    pub contents: [[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
}

const MAX_DISTANCE_X: i32 = MAX_DEPTH as i32 / CHUNK_WIDTH as i32 + 1;
const MAX_DISTANCE_Y: i32 = MAX_DEPTH as i32 / CHUNK_DEPTH as i32 + 1;
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
        (x + (chunk_location[0] * CHUNK_WIDTH as i32)) as f64 / scale + offset,
        (z + (chunk_location[1] * CHUNK_DEPTH as i32)) as f64 / scale + offset,
    ])
}

fn chunk_at_block(
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    x: i32,
    z: i32,
) -> Option<ChunkData> {
    let chunk_x = x.div_euclid(CHUNK_WIDTH as i32);
    let chunk_z = z.div_euclid(CHUNK_DEPTH as i32);
    generated_chunks.get(&[chunk_x, chunk_z]).cloned()
}

pub fn get_block(
    generated_chunks: &HashMap<[i32; 2], ChunkData>,
    x: i32,
    y: i32,
    z: i32,
) -> Option<BlockType> {
    let chunk = chunk_at_block(generated_chunks, x, z)?;
    let x = (x - (x.div_euclid(CHUNK_WIDTH as i32) * CHUNK_WIDTH as i32)) as usize;
    let z = (z - (z.div_euclid(CHUNK_DEPTH as i32) * CHUNK_DEPTH as i32)) as usize;
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
    let chunk_x = (x as i32).div_euclid(CHUNK_WIDTH as i32);
    let chunk_z = (z as i32).div_euclid(CHUNK_WIDTH as i32);
    let length = |a, b| (a * a + b * b);
    let mut collector: Option<[i32; 2]> = None;
    for i in -MAX_DISTANCE_X..=MAX_DISTANCE_X {
        for j in -MAX_DISTANCE_Y..=MAX_DISTANCE_Y {
            let distance = length(i, j);
            if distance <= MAX_DEPTH as i32
                && ((collector.is_some()
                    && distance < length(collector.unwrap()[0], collector.unwrap()[1]))
                    || collector.is_none())
                && !generated_chunks.contains_key(&[i + chunk_x, j + chunk_z])
            {
                collector = Some([i, j]);
            }
        }
    }
    collector.map(|value: [i32; 2]| [value[0] + chunk_x, value[1] + chunk_z])
}

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;

fn add_arrs(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

pub fn generate_chunk_mesh(
    location: [i32; 2],
    chunk: [[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
    surrounding_chunks: [Option<&ChunkData>; 4], // north, south, east, west for... reasons...
) -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = (vec![], vec![]);
    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                if chunk[x][y][z] == BlockType::Air {
                    continue;
                } else if chunk[x][y][z] == BlockType::Grass {
                    let tex_offset = BlockType::Grass.get_offset()[0];
                    let x = (x as i32 + (location[0] * CHUNK_WIDTH as i32)) as f32;
                    let z = (z as i32 + (location[1] * CHUNK_WIDTH as i32)) as f32;
                    let y = y as f32;
                    indices.extend(
                        [
                            0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
                        ]
                        .iter()
                        .map(|i| *i + vertices.len() as u32),
                    );
                    vertices.append(&mut vec![
                        Vertex(
                            [x + CLOSE_CORNER, y + 1.0, z + CLOSE_CORNER],
                            add_arrs(TOP_LEFT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + CLOSE_CORNER, y, z + CLOSE_CORNER],
                            add_arrs(BOTTOM_LEFT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + FAR_CORNER, y, z + FAR_CORNER],
                            add_arrs(BOTTOM_RIGHT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + FAR_CORNER, y + 1.0, z + FAR_CORNER],
                            add_arrs(TOP_RIGHT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + CLOSE_CORNER, y + 1.0, z + FAR_CORNER],
                            add_arrs(TOP_LEFT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + CLOSE_CORNER, y, z + FAR_CORNER],
                            add_arrs(BOTTOM_LEFT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + FAR_CORNER, y, z + CLOSE_CORNER],
                            add_arrs(BOTTOM_RIGHT, tex_offset),
                            1.0,
                        ),
                        Vertex(
                            [x + FAR_CORNER, y + 1.0, z + CLOSE_CORNER],
                            add_arrs(TOP_RIGHT, tex_offset),
                            1.0,
                        ),
                    ]);
                    continue;
                }
                let tex_offsets = chunk[x][y][z].get_offset();
                let rel_x = (x as i32 + (location[0] * CHUNK_WIDTH as i32)) as f32;
                let rel_z = (z as i32 + (location[1] * CHUNK_DEPTH as i32)) as f32;
                let y_f32 = y as f32;
                // first face
                if (z == CHUNK_DEPTH - 1
                    && surrounding_chunks[2]
                        .map_or(true, |chunk| !chunk.contents[x][y][0].is_solid()))
                    || (z != CHUNK_DEPTH - 1 && !chunk[x][y][z + 1].is_solid())
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
                                        && chunk.contents[CHUNK_WIDTH - 1][y][z + 1].is_solid()
                                }))
                                || (x != 0
                                    && z != CHUNK_DEPTH - 1
                                    && chunk[x - 1][y][z + 1].is_solid())
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [rel_x, y_f32, zplusone],
                            add_arrs(BOTTOM_LEFT, tex_offset),
                            if y != 0
                                && ((z != CHUNK_DEPTH - 1 && chunk[x][y - 1][z + 1].is_solid())
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y - 1][0].is_solid()
                                        })))
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [1.0 + rel_x, y_f32, zplusone],
                            add_arrs(BOTTOM_RIGHT, tex_offset),
                            if y != 0
                                && ((z != CHUNK_DEPTH - 1 && chunk[x][y - 1][z + 1].is_solid())
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y - 1][0].is_solid()
                                        })))
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [1.0 + rel_x, 1.0 + y_f32, zplusone],
                            add_arrs(TOP_RIGHT, tex_offset),
                            if (x == CHUNK_WIDTH - 1
                                && surrounding_chunks[0].map_or(false, |chunk| {
                                    z != CHUNK_DEPTH - 1 && chunk.contents[0][y][z + 1].is_solid()
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != CHUNK_DEPTH - 1
                                    && chunk[x + 1][y][z + 1].is_solid())
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        ),
                    ]);
                }
                // second face
                if (x == CHUNK_WIDTH - 1
                    && surrounding_chunks[0]
                        .map_or(true, |chunk| !chunk.contents[0][y][z].is_solid()))
                    || (x != CHUNK_WIDTH - 1 && !chunk[x + 1][y][z].is_solid())
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
                                    z != CHUNK_WIDTH - 1 && chunk.contents[0][y][z + 1].is_solid()
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != CHUNK_WIDTH - 1
                                    && chunk[x + 1][y][z + 1].is_solid())
                            {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [xplusone, y_f32, 1.0 + rel_z],
                            add_arrs(BOTTOM_LEFT, tex_offset),
                            if y != 0 && x != CHUNK_WIDTH - 1 && chunk[x + 1][y - 1][z].is_solid() {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [xplusone, y_f32, rel_z],
                            add_arrs(BOTTOM_RIGHT, tex_offset),
                            if y != 0 && x != CHUNK_WIDTH - 1 && chunk[x + 1][y - 1][z].is_solid() {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [xplusone, 1.0 + y_f32, rel_z],
                            add_arrs(TOP_RIGHT, tex_offset),
                            if (x == CHUNK_WIDTH - 1
                                && surrounding_chunks[1].map_or(false, |chunk| {
                                    z != 0 && chunk.contents[0][y][z - 1].is_solid()
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != 0
                                    && chunk[x + 1][y][z - 1].is_solid())
                            {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        ),
                    ]);
                }
                // third face
                if (z == 0
                    && surrounding_chunks[3].map_or(true, |chunk| {
                        !chunk.contents[x][y][CHUNK_DEPTH - 1].is_solid()
                    }))
                    || (z != 0 && !chunk[x][y][z - 1].is_solid())
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
                            if y != 0 && z != 0 && chunk[x][y - 1][z - 1].is_solid() {
                                0.5
                            } else {
                                BACK_BRIGHTNESS
                            },
                        ),
                        Vertex(
                            [rel_x, y_f32, rel_z],
                            add_arrs(BOTTOM_RIGHT, tex_offset),
                            if y != 0 && z != 0 && chunk[x][y - 1][z - 1].is_solid() {
                                0.5
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
                        !chunk.contents.last().unwrap()[y][z].is_solid()
                    }))
                    || (x != 0 && !chunk[x - 1][y][z].is_solid())
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
                if y == CHUNK_HEIGHT - 1 || !chunk[x][y + 1][z].is_solid() {
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
                                        chunk.contents[CHUNK_WIDTH - 1][y + 1][z].is_solid()
                                            || (z != 0
                                                && chunk.contents[CHUNK_WIDTH - 1][y + 1][z - 1]
                                                    .is_solid())
                                    }))
                                    || (x != 0
                                        && (chunk[x - 1][y + 1][z].is_solid()
                                            || (z == 0
                                                && surrounding_chunks[3].map_or(false, |chunk| {
                                                    chunk.contents[x - 1][y + 1][CHUNK_DEPTH - 1]
                                                        .is_solid()
                                                }))
                                            || (z != 0 && chunk[x - 1][y + 1][z - 1].is_solid())))
                                    || (z == 0
                                        && surrounding_chunks[3].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][CHUNK_DEPTH - 1].is_solid()
                                        }))
                                    || (z != 0 && chunk[x][y + 1][z - 1].is_solid())
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            ),
                            Vertex(
                                [rel_x, yplusone, 1.0 + rel_z],
                                add_arrs(BOTTOM_LEFT, tex_offset),
                                if (x == 0
                                    && surrounding_chunks[1].map_or(false, |chunk| {
                                        chunk.contents[CHUNK_WIDTH - 1][y + 1][z].is_solid()
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk.contents[CHUNK_WIDTH - 1][y + 1][z + 1]
                                                    .is_solid())
                                    }))
                                    || (x != 0
                                        && y != CHUNK_HEIGHT - 1
                                        && (chunk[x - 1][y + 1][z].is_solid()
                                            || ((z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(
                                                    false,
                                                    |chunk| {
                                                        chunk.contents[x - 1][y + 1][0].is_solid()
                                                    },
                                                ))
                                                || (z != CHUNK_DEPTH - 1
                                                    && chunk[x - 1][y + 1][z + 1].is_solid()))))
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][0].is_solid()
                                        }))
                                    || (z != CHUNK_DEPTH - 1
                                        && y != CHUNK_HEIGHT - 1
                                        && chunk[x][y + 1][z + 1].is_solid())
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            ),
                            Vertex(
                                [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                add_arrs(BOTTOM_RIGHT, tex_offset),
                                if (x == CHUNK_WIDTH - 1
                                    && surrounding_chunks[0].map_or(false, |chunk| {
                                        chunk.contents[0][y + 1][z].is_solid()
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk.contents[0][y + 1][z + 1].is_solid())
                                    }))
                                    || (x != CHUNK_WIDTH - 1
                                        && y != CHUNK_HEIGHT - 1
                                        && (chunk[x + 1][y + 1][z].is_solid()
                                            || (z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(false, |chunk| {
                                                    chunk.contents[x + 1][y + 1][0].is_solid()
                                                }))
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk[x + 1][y + 1][z + 1].is_solid())))
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][0].is_solid()
                                        }))
                                    || (z != CHUNK_DEPTH - 1
                                        && y != CHUNK_HEIGHT - 1
                                        && chunk[x][y + 1][z + 1].is_solid())
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            ),
                            Vertex(
                                [1.0 + rel_x, yplusone, rel_z],
                                add_arrs(TOP_RIGHT, tex_offset),
                                if (x == CHUNK_WIDTH - 1
                                    && surrounding_chunks[0].map_or(false, |chunk| {
                                        chunk.contents[0][y + 1][z].is_solid()
                                            || (z != 0
                                                && chunk.contents[0][y + 1][z - 1].is_solid())
                                    }))
                                    || (x != CHUNK_WIDTH - 1
                                        && y != CHUNK_HEIGHT - 1
                                        && ((z == 0
                                            && surrounding_chunks[3].map_or(false, |chunk| {
                                                chunk.contents[x + 1][y + 1][CHUNK_DEPTH - 1]
                                                    .is_solid()
                                            }))
                                            || (z != 0 && chunk[x + 1][y + 1][z - 1].is_solid())
                                            || chunk[x + 1][y + 1][z].is_solid()))
                                    || (z == 0
                                        && surrounding_chunks[3].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][CHUNK_DEPTH - 1].is_solid()
                                        }))
                                    || (z != 0
                                        && y != CHUNK_HEIGHT - 1
                                        && chunk[x][y + 1][z - 1].is_solid())
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            ),
                        ])
                    }
                }
                // bottom face
                if y == 0 || !chunk[x][y - 1][z].is_solid() {
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
        }
    }
    (vertices, indices)
}
