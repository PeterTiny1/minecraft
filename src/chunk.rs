use noise::{NoiseFn, OpenSimplex};

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

const TEXTURE_WIDTH: f32 = 0.5;

const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [TEXTURE_WIDTH, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, TEXTURE_WIDTH];
const BOTTOM_RIGHT: [f32; 2] = [TEXTURE_WIDTH, TEXTURE_WIDTH];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockType {
    Air,
    Stone,
    GrassBlock,
}

impl BlockType {
    fn get_offset(&self) -> [[f32; 2]; 6] {
        match self {
            BlockType::Stone => [[0.0, 0.0]; 6],
            BlockType::GrassBlock => [
                [TEXTURE_WIDTH, 0.0],
                [0.0, TEXTURE_WIDTH],
                [0.0, TEXTURE_WIDTH],
                [0.0, TEXTURE_WIDTH],
                [0.0, TEXTURE_WIDTH],
                [0.0, 0.0],
            ],
            _ => panic!("This is not supposed to be called!"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkData {
    pub location: [i32; 2],
    pub contents: [[[BlockType; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
}

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
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

pub fn get_nearest_chunk_location(
    x: f32,
    z: f32,
    generated_chunks: &Vec<ChunkData>,
) -> Option<[i32; 2]> {
    let (chunk_x, chunk_y) = (
        (x / CHUNK_WIDTH as f32).floor() as i32,
        (z / CHUNK_DEPTH as f32).floor() as i32,
    );
    let length = |a, b| (a * a + b * b);
    let mut collector: Option<[i32; 2]> = None;
    for i in -MAX_DISTANCE_X..=MAX_DISTANCE_X {
        for j in -MAX_DISTANCE_Y..=MAX_DISTANCE_Y {
            let distance = length(i, j);
            if distance <= MAX_DEPTH as i32
                && ((collector.is_some()
                    && distance < length(collector.unwrap()[0], collector.unwrap()[1]))
                    || collector.is_none())
                && generated_chunks
                    .iter()
                    .all(|chunk| chunk.location != [i + chunk_x, j + chunk_y])
            {
                collector = Some([i, j]);
            }
        }
    }
    collector.and_then(|value: [i32; 2]| Some([value[0] + chunk_x, value[1] + chunk_y]))
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
                }
                let tex_offsets = chunk[x][y][z].get_offset();
                let rel_x = (x as i32 + (location[0] * CHUNK_WIDTH as i32)) as f32;
                let rel_z = (z as i32 + (location[1] * CHUNK_DEPTH as i32)) as f32;
                let y_f32 = y as f32;
                // first face
                if (z == CHUNK_DEPTH - 1
                    && surrounding_chunks[2]
                        .map_or(true, |chunk| chunk.contents[x][y][0] == BlockType::Air))
                    || (z != CHUNK_DEPTH - 1 && chunk[x][y][z + 1] == BlockType::Air)
                {
                    let tex_offset = tex_offsets[1];
                    indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    let zplusone = 1.0 + rel_z;
                    vertices.append(&mut vec![
                        Vertex {
                            position: [rel_x, 1.0 + y_f32, zplusone],
                            tex_coords: [TOP_LEFT[0] + tex_offset[0], TOP_LEFT[1] + tex_offset[1]],
                            brightness: if (x == 0
                                && surrounding_chunks[1].map_or(false, |chunk| {
                                    z != CHUNK_DEPTH - 1
                                        && chunk.contents[CHUNK_WIDTH - 1][y][z + 1]
                                            != BlockType::Air
                                }))
                                || (x != 0
                                    && z != CHUNK_DEPTH - 1
                                    && chunk[x - 1][y][z + 1] != BlockType::Air)
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [rel_x, y_f32, zplusone],
                            tex_coords: [
                                BOTTOM_LEFT[0] + tex_offset[0],
                                BOTTOM_LEFT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0
                                && ((z != CHUNK_DEPTH - 1
                                    && chunk[x][y - 1][z + 1] != BlockType::Air)
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y - 1][0] != BlockType::Air
                                        }))) {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [1.0 + rel_x, y_f32, zplusone],
                            tex_coords: [
                                BOTTOM_RIGHT[0] + tex_offset[0],
                                BOTTOM_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0
                                && ((z != CHUNK_DEPTH - 1
                                    && chunk[x][y - 1][z + 1] != BlockType::Air)
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y - 1][0] != BlockType::Air
                                        }))) {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [1.0 + rel_x, 1.0 + y_f32, zplusone],
                            tex_coords: [
                                TOP_RIGHT[0] + tex_offset[0],
                                TOP_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: if (x == CHUNK_WIDTH - 1
                                && surrounding_chunks[0].map_or(false, |chunk| {
                                    z != CHUNK_DEPTH - 1
                                        && chunk.contents[0][y][z + 1] != BlockType::Air
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != CHUNK_DEPTH - 1
                                    && chunk[x + 1][y][z + 1] != BlockType::Air)
                            {
                                0.5
                            } else {
                                FRONT_BRIGHTNESS
                            },
                        },
                    ]);
                }
                // second face
                if (x == CHUNK_WIDTH - 1
                    && surrounding_chunks[0]
                        .map_or(true, |chunk| chunk.contents[0][y][z] == BlockType::Air))
                    || (x != CHUNK_WIDTH - 1 && chunk[x + 1][y][z] == BlockType::Air)
                {
                    let tex_offset = tex_offsets[2];
                    indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    let xplusone = 1.0 + rel_x;
                    vertices.append(&mut vec![
                        Vertex {
                            position: [xplusone, 1.0 + y_f32, 1.0 + rel_z],
                            tex_coords: [TOP_LEFT[0] + tex_offset[0], TOP_LEFT[1] + tex_offset[1]],
                            brightness: if (x == CHUNK_WIDTH - 1
                                && surrounding_chunks[0].map_or(false, |chunk| {
                                    z != CHUNK_WIDTH - 1
                                        && chunk.contents[0][y][z + 1] != BlockType::Air
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != CHUNK_WIDTH - 1
                                    && chunk[x + 1][y][z + 1] != BlockType::Air)
                            {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [xplusone, y_f32, 1.0 + rel_z],
                            tex_coords: [
                                BOTTOM_LEFT[0] + tex_offset[0],
                                BOTTOM_LEFT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0 && chunk[x][y - 1][z] != BlockType::Air {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [xplusone, y_f32, rel_z],
                            tex_coords: [
                                BOTTOM_RIGHT[0] + tex_offset[0],
                                BOTTOM_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0 && chunk[x][y - 1][z] != BlockType::Air {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [xplusone, 1.0 + y_f32, rel_z],
                            tex_coords: [
                                TOP_RIGHT[0] + tex_offset[0],
                                TOP_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: if (x == CHUNK_WIDTH - 1
                                && surrounding_chunks[1].map_or(false, |chunk| {
                                    z != 0 && chunk.contents[0][y][z - 1] != BlockType::Air
                                }))
                                || (x != CHUNK_WIDTH - 1
                                    && z != 0
                                    && chunk[x + 1][y][z - 1] != BlockType::Air)
                            {
                                0.5
                            } else {
                                SIDE_BRIGHTNESS
                            },
                        },
                    ]);
                }
                // third face
                if (z == 0
                    && surrounding_chunks[3].map_or(true, |chunk| {
                        chunk.contents[x][y].last().unwrap() == &BlockType::Air
                    }))
                    || (z != 0 && chunk[x][y][z - 1] == BlockType::Air)
                {
                    let tex_offset = tex_offsets[3];
                    indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    vertices.append(&mut vec![
                        Vertex {
                            position: [1.0 + rel_x, 1.0 + y_f32, rel_z],
                            tex_coords: [TOP_LEFT[0] + tex_offset[0], TOP_LEFT[1] + tex_offset[1]],
                            brightness: BACK_BRIGHTNESS,
                        },
                        Vertex {
                            position: [1.0 + rel_x, y_f32, rel_z],
                            tex_coords: [
                                BOTTOM_LEFT[0] + tex_offset[0],
                                BOTTOM_LEFT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0 && chunk[x][y - 1][z] != BlockType::Air {
                                0.5
                            } else {
                                BACK_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [rel_x, y_f32, rel_z],
                            tex_coords: [
                                BOTTOM_RIGHT[0] + tex_offset[0],
                                BOTTOM_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: if y != 0 && chunk[x][y - 1][z] != BlockType::Air {
                                0.5
                            } else {
                                BACK_BRIGHTNESS
                            },
                        },
                        Vertex {
                            position: [rel_x, 1.0 + y_f32, rel_z],
                            tex_coords: [
                                TOP_RIGHT[0] + tex_offset[0],
                                TOP_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: BACK_BRIGHTNESS,
                        },
                    ]);
                }
                // fourth face
                if (x == 0
                    && surrounding_chunks[1].map_or(true, |chunk| {
                        chunk.contents.last().unwrap()[y][z] == BlockType::Air
                    }))
                    || (x != 0 && chunk[x - 1][y][z] == BlockType::Air)
                {
                    let tex_offset = tex_offsets[4];
                    indices.extend(&mut QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    vertices.append(&mut vec![
                        Vertex {
                            position: [rel_x, 1.0 + y_f32, rel_z],
                            tex_coords: [TOP_LEFT[0] + tex_offset[0], TOP_LEFT[1] + tex_offset[1]],
                            brightness: SIDE_BRIGHTNESS,
                        },
                        Vertex {
                            position: [rel_x, y_f32, rel_z],
                            tex_coords: [
                                BOTTOM_LEFT[0] + tex_offset[0],
                                BOTTOM_LEFT[1] + tex_offset[1],
                            ],
                            brightness: SIDE_BRIGHTNESS,
                        },
                        Vertex {
                            position: [rel_x, y_f32, 1.0 + rel_z],
                            tex_coords: [
                                BOTTOM_RIGHT[0] + tex_offset[0],
                                BOTTOM_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: SIDE_BRIGHTNESS,
                        },
                        Vertex {
                            position: [rel_x, 1.0 + y_f32, 1.0 + rel_z],
                            tex_coords: [
                                TOP_RIGHT[0] + tex_offset[0],
                                TOP_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: SIDE_BRIGHTNESS,
                        },
                    ]);
                }
                // top face
                if y == CHUNK_HEIGHT - 1 || chunk[x][y + 1][z] == BlockType::Air {
                    let tex_offset = tex_offsets[0];
                    indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    let yplusone = y_f32 + 1.0;
                    if y == CHUNK_HEIGHT - 2 {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [rel_x, yplusone, rel_z],
                                tex_coords: [
                                    TOP_LEFT[0] + tex_offset[0],
                                    TOP_LEFT[1] + tex_offset[1],
                                ],
                                brightness: TOP_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: [
                                    BOTTOM_LEFT[0] + tex_offset[0],
                                    BOTTOM_LEFT[1] + tex_offset[1],
                                ],
                                brightness: TOP_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: [
                                    BOTTOM_RIGHT[0] + tex_offset[0],
                                    BOTTOM_RIGHT[1] + tex_offset[1],
                                ],
                                brightness: TOP_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, rel_z],
                                tex_coords: [
                                    TOP_RIGHT[0] + tex_offset[0],
                                    TOP_RIGHT[1] + tex_offset[1],
                                ],
                                brightness: TOP_BRIGHTNESS,
                            },
                        ]);
                    } else {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [rel_x, yplusone, rel_z],
                                tex_coords: [
                                    TOP_LEFT[0] + tex_offset[0],
                                    TOP_LEFT[1] + tex_offset[1],
                                ],
                                brightness: if (x == 0
                                    && surrounding_chunks[1].map_or(false, |chunk| {
                                        chunk.contents[CHUNK_WIDTH - 1][y + 1][z] != BlockType::Air
                                            || (z != 0
                                                && chunk.contents[CHUNK_WIDTH - 1][y + 1][z - 1]
                                                    != BlockType::Air)
                                    }))
                                    || (x != 0
                                        && (chunk[x - 1][y + 1][z] != BlockType::Air
                                            || (z == 0
                                                && surrounding_chunks[3].map_or(false, |chunk| {
                                                    chunk.contents[x - 1][y + 1][CHUNK_DEPTH - 1]
                                                        != BlockType::Air
                                                }))
                                            || (z != 0
                                                && chunk[x - 1][y + 1][z - 1] != BlockType::Air)))
                                    || (z == 0
                                        && surrounding_chunks[3].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][CHUNK_DEPTH - 1]
                                                != BlockType::Air
                                        }))
                                    || (z != 0 && chunk[x][y + 1][z - 1] != BlockType::Air)
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            },
                            Vertex {
                                position: [rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: [
                                    BOTTOM_LEFT[0] + tex_offset[0],
                                    BOTTOM_LEFT[1] + tex_offset[1],
                                ],
                                brightness: if (x == 0
                                    && surrounding_chunks[1].map_or(false, |chunk| {
                                        chunk.contents[CHUNK_WIDTH - 1][y + 1][z] != BlockType::Air
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk.contents[CHUNK_WIDTH - 1][y + 1][z + 1]
                                                    != BlockType::Air)
                                    }))
                                    || (x != 0
                                        && (chunk[x - 1][y + 1][z] != BlockType::Air
                                            || ((z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(
                                                    false,
                                                    |chunk| {
                                                        chunk.contents[x - 1][y + 1][0]
                                                            != BlockType::Air
                                                    },
                                                ))
                                                || (z != CHUNK_DEPTH - 1
                                                    && chunk[x - 1][y + 1][z + 1]
                                                        != BlockType::Air))))
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][0] != BlockType::Air
                                        }))
                                    || (z != CHUNK_DEPTH - 1
                                        && chunk[x][y + 1][z + 1] != BlockType::Air)
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: [
                                    BOTTOM_RIGHT[0] + tex_offset[0],
                                    BOTTOM_RIGHT[1] + tex_offset[1],
                                ],
                                brightness: if (x == CHUNK_WIDTH - 1
                                    && surrounding_chunks[0].map_or(false, |chunk| {
                                        chunk.contents[0][y + 1][z] != BlockType::Air
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk.contents[0][y + 1][z + 1]
                                                    != BlockType::Air)
                                    }))
                                    || (x != CHUNK_WIDTH - 1
                                        && (chunk[x + 1][y + 1][z] != BlockType::Air
                                            || (z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(false, |chunk| {
                                                    chunk.contents[x + 1][y + 1][0]
                                                        != BlockType::Air
                                                }))
                                            || (z != CHUNK_DEPTH - 1
                                                && chunk[x + 1][y + 1][z + 1] != BlockType::Air)))
                                    || (z == CHUNK_DEPTH - 1
                                        && surrounding_chunks[2].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][0] != BlockType::Air
                                        }))
                                    || (z != CHUNK_DEPTH - 1
                                        && chunk[x][y + 1][z + 1] != BlockType::Air)
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, rel_z],
                                tex_coords: [
                                    TOP_RIGHT[0] + tex_offset[0],
                                    TOP_RIGHT[1] + tex_offset[1],
                                ],
                                brightness: if (x == CHUNK_WIDTH - 1
                                    && surrounding_chunks[0].map_or(false, |chunk| {
                                        chunk.contents[0][y + 1][z] != BlockType::Air
                                            || (z != 0
                                                && chunk.contents[0][y + 1][z - 1]
                                                    != BlockType::Air)
                                    }))
                                    || (x != CHUNK_WIDTH - 1
                                        && ((z == 0
                                            && surrounding_chunks[3].map_or(false, |chunk| {
                                                chunk.contents[x + 1][y + 1][CHUNK_DEPTH - 1]
                                                    != BlockType::Air
                                            }))
                                            || (z != 0
                                                && chunk[x + 1][y + 1][z - 1] != BlockType::Air)
                                            || chunk[x + 1][y + 1][z] != BlockType::Air))
                                    || (z == 0
                                        && surrounding_chunks[3].map_or(false, |chunk| {
                                            chunk.contents[x][y + 1][CHUNK_DEPTH - 1]
                                                != BlockType::Air
                                        }))
                                    || (z != 0 && chunk[x][y + 1][z - 1] != BlockType::Air)
                                {
                                    0.5
                                } else {
                                    TOP_BRIGHTNESS
                                },
                            },
                        ])
                    }
                }
                // bottom face
                if y == 0 || chunk[x][y - 1][z] == BlockType::Air {
                    let tex_offset = tex_offsets[5];
                    indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                    vertices.append(&mut vec![
                        // start of bottom
                        Vertex {
                            position: [1.0 + rel_x, y_f32, rel_z],
                            tex_coords: [TOP_LEFT[0] + tex_offset[0], TOP_LEFT[1] + tex_offset[1]],
                            brightness: BOTTOM_BRIGHTNESS,
                        },
                        Vertex {
                            position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                            tex_coords: [
                                BOTTOM_LEFT[0] + tex_offset[0],
                                BOTTOM_LEFT[1] + tex_offset[1],
                            ],
                            brightness: BOTTOM_BRIGHTNESS,
                        },
                        Vertex {
                            position: [rel_x, y_f32, 1.0 + rel_z],
                            tex_coords: [
                                BOTTOM_RIGHT[0] + tex_offset[0],
                                BOTTOM_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: BOTTOM_BRIGHTNESS,
                        },
                        Vertex {
                            position: [rel_x, y_f32, rel_z],
                            tex_coords: [
                                TOP_RIGHT[0] + tex_offset[0],
                                TOP_RIGHT[1] + tex_offset[1],
                            ],
                            brightness: BOTTOM_BRIGHTNESS,
                        },
                    ]);
                }
            }
        }
    }
    (vertices, indices)
}
