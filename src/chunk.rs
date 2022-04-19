use noise::{NoiseFn, OpenSimplex};

use crate::{Vertex, MAX_DEPTH};

pub const CHUNK_WIDTH: usize = 32;
pub const CHUNK_HEIGHT: usize = 256;
pub const CHUNK_DEPTH: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct ChunkData {
    pub location: [i32; 2],
    pub contents: [[[u16; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
}

const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [0.0, 0.5];
const BOTTOM_LEFT: [f32; 2] = [0.5, 0.0];
const BOTTOM_RIGHT: [f32; 2] = [0.5, 0.5];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const MAX_DISTANCE_X: i32 = MAX_DEPTH as i32 / CHUNK_WIDTH as i32 + 1;
const MAX_DISTANCE_Y: i32 = MAX_DEPTH as i32 / CHUNK_DEPTH as i32 + 1;
const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.3;
const SIDE_BRIGHTNESS: f32 = 0.6;
const FRONT_BRIGHTNESS: f32 = 0.7;
const BACK_BRIGHTNESS: f32 = 0.5;

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
    chunk: [[[u16; CHUNK_DEPTH]; CHUNK_HEIGHT]; CHUNK_WIDTH],
    surrounding_chunks: [Option<&ChunkData>; 4], // north, south, east, west for... reasons...
) -> (Vec<Vertex>, Vec<u32>) {
    let (mut vertices, mut indices) = (vec![], vec![]);
    for x in 0..CHUNK_WIDTH {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH {
                if chunk[x][y][z] != 0 {
                    let rel_x = (x as i32 + (location[0] * CHUNK_WIDTH as i32)) as f32;
                    let rel_z = (z as i32 + (location[1] * CHUNK_DEPTH as i32)) as f32;
                    let y_f32 = y as f32;
                    // first face
                    if (z == CHUNK_DEPTH - 1
                        && surrounding_chunks[2].map_or(true, |chunk| chunk.contents[x][y][0] == 0))
                        || (z != CHUNK_DEPTH - 1 && chunk[x][y][z + 1] == 0)
                    {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        let zplusone = 1.0 + rel_z;
                        vertices.append(&mut vec![
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, zplusone],
                                tex_coords: TOP_LEFT,
                                brightness: FRONT_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, zplusone],
                                tex_coords: TOP_RIGHT,
                                brightness: FRONT_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, zplusone],
                                tex_coords: BOTTOM_RIGHT,
                                brightness: FRONT_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, 1.0 + y_f32, zplusone],
                                tex_coords: BOTTOM_LEFT,
                                brightness: FRONT_BRIGHTNESS,
                            },
                        ]);
                    }
                    // second face
                    if (x == CHUNK_WIDTH - 1
                        && surrounding_chunks[0].map_or(true, |chunk| chunk.contents[0][y][z] == 0))
                        || (x != CHUNK_WIDTH - 1 && chunk[x + 1][y][z] == 0)
                    {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        let xplusone = 1.0 + rel_x;
                        vertices.append(&mut vec![
                            Vertex {
                                position: [xplusone, 1.0 + y_f32, 1.0 + rel_z],
                                tex_coords: TOP_LEFT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [xplusone, y_f32, 1.0 + rel_z],
                                tex_coords: TOP_RIGHT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [xplusone, y_f32, rel_z],
                                tex_coords: BOTTOM_RIGHT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [xplusone, 1.0 + y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                        ]);
                    }
                    // third face
                    if (z == 0
                        && surrounding_chunks[3]
                            .map_or(true, |chunk| chunk.contents[x][y].last().unwrap() == &0_u16))
                        || (z != 0 && chunk[x][y][z - 1] == 0)
                    {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        vertices.append(&mut vec![
                            Vertex {
                                position: [1.0 + rel_x, 1.0 + y_f32, rel_z],
                                tex_coords: TOP_LEFT,
                                brightness: BACK_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, rel_z],
                                tex_coords: TOP_RIGHT,
                                brightness: BACK_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: BOTTOM_RIGHT,
                                brightness: BACK_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
                                brightness: BACK_BRIGHTNESS,
                            },
                        ]);
                    }
                    // fourth face
                    if (x == 0
                        && surrounding_chunks[1]
                            .map_or(true, |chunk| chunk.contents.last().unwrap()[y][z] == 0))
                        || (x != 0 && chunk[x - 1][y][z] == 0)
                    {
                        indices
                            .extend(&mut QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        vertices.append(&mut vec![
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, rel_z],
                                tex_coords: TOP_LEFT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: TOP_RIGHT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_RIGHT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_LEFT,
                                brightness: SIDE_BRIGHTNESS,
                            },
                        ]);
                    }
                    // top face
                    if y == CHUNK_HEIGHT - 1 || chunk[x][y + 1][z] == 0 {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        let yplusone = y_f32 + 1.0;
                        if y == CHUNK_HEIGHT - 2 {
                            vertices.append(&mut vec![
                                Vertex {
                                    position: [rel_x, yplusone, rel_z],
                                    tex_coords: TOP_LEFT,
                                    brightness: TOP_BRIGHTNESS,
                                },
                                Vertex {
                                    position: [rel_x, yplusone, 1.0 + rel_z],
                                    tex_coords: TOP_RIGHT,
                                    brightness: TOP_BRIGHTNESS,
                                },
                                Vertex {
                                    position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                    tex_coords: BOTTOM_RIGHT,
                                    brightness: TOP_BRIGHTNESS,
                                },
                                Vertex {
                                    position: [1.0 + rel_x, yplusone, rel_z],
                                    tex_coords: BOTTOM_LEFT,
                                    brightness: TOP_BRIGHTNESS,
                                },
                            ]);
                        } else {
                            vertices.append(&mut vec![
                                Vertex {
                                    position: [rel_x, yplusone, rel_z],
                                    tex_coords: TOP_LEFT,
                                    brightness: TOP_BRIGHTNESS
                                        - if (x == 0
                                            && surrounding_chunks[1].map_or(false, |chunk| {
                                                chunk.contents[CHUNK_WIDTH - 1][y + 1][z] != 0
                                                    || (z != 0
                                                        && chunk.contents[CHUNK_WIDTH - 1][y + 1]
                                                            [z - 1]
                                                            != 0)
                                            }))
                                            || (x != 0
                                                && (chunk[x - 1][y + 1][z] != 0
                                                    || (z == 0
                                                        && surrounding_chunks[3].map_or(
                                                            false,
                                                            |chunk| {
                                                                chunk.contents[x - 1][y + 1]
                                                                    [CHUNK_DEPTH - 1]
                                                                    != 0
                                                            },
                                                        ))
                                                    || (z != 0 && chunk[x - 1][y + 1][z - 1] != 0)))
                                            || (z == 0
                                                && surrounding_chunks[3].map_or(false, |chunk| {
                                                    chunk.contents[x][y + 1][CHUNK_DEPTH - 1] != 0
                                                }))
                                            || (z != 0 && chunk[x][y + 1][z - 1] != 0)
                                        {
                                            0.3
                                        } else {
                                            0.0
                                        },
                                },
                                Vertex {
                                    position: [rel_x, yplusone, 1.0 + rel_z],
                                    tex_coords: TOP_RIGHT,
                                    brightness: TOP_BRIGHTNESS
                                        - if (x == 0
                                            && surrounding_chunks[1].map_or(false, |chunk| {
                                                chunk.contents[CHUNK_WIDTH - 1][y + 1][z] != 0
                                                    || (z != CHUNK_DEPTH - 1
                                                        && chunk.contents[CHUNK_WIDTH - 1][y + 1]
                                                            [z + 1]
                                                            != 0)
                                            }))
                                            || (x != 0
                                                && (chunk[x - 1][y + 1][z] != 0
                                                    || ((z == CHUNK_DEPTH - 1
                                                        && surrounding_chunks[2].map_or(
                                                            false,
                                                            |chunk| {
                                                                chunk.contents[x - 1][y + 1][0] != 0
                                                            },
                                                        ))
                                                        || (z != CHUNK_DEPTH - 1
                                                            && chunk[x - 1][y + 1][z + 1] != 0))))
                                            || (z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(false, |chunk| {
                                                    chunk.contents[x][y + 1][0] != 0
                                                }))
                                            || (z != CHUNK_DEPTH - 1 && chunk[x][y + 1][z + 1] != 0)
                                        {
                                            0.3
                                        } else {
                                            0.0
                                        },
                                },
                                Vertex {
                                    position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                    tex_coords: BOTTOM_RIGHT,
                                    brightness: TOP_BRIGHTNESS
                                        - if (x == CHUNK_WIDTH - 1
                                            && surrounding_chunks[0].map_or(false, |chunk| {
                                                chunk.contents[0][y + 1][z] != 0
                                                    || (z != CHUNK_DEPTH - 1
                                                        && chunk.contents[0][y + 1][z + 1] != 0)
                                            }))
                                            || (x != CHUNK_WIDTH - 1
                                                && (chunk[x + 1][y + 1][z] != 0
                                                    || (z == CHUNK_DEPTH - 1
                                                        && surrounding_chunks[2].map_or(
                                                            false,
                                                            |chunk| {
                                                                chunk.contents[x + 1][y + 1][0] != 0
                                                            },
                                                        ))
                                                    || (z != CHUNK_DEPTH - 1
                                                        && chunk[x + 1][y + 1][z + 1] != 0)))
                                            || (z == CHUNK_DEPTH - 1
                                                && surrounding_chunks[2].map_or(false, |chunk| {
                                                    chunk.contents[x][y + 1][0] != 0
                                                }))
                                            || (z != CHUNK_DEPTH - 1 && chunk[x][y + 1][z + 1] != 0)
                                        {
                                            0.3
                                        } else {
                                            0.0
                                        },
                                },
                                Vertex {
                                    position: [1.0 + rel_x, yplusone, rel_z],
                                    tex_coords: BOTTOM_LEFT,
                                    brightness: TOP_BRIGHTNESS
                                        - if (x == CHUNK_WIDTH - 1
                                            && surrounding_chunks[0].map_or(false, |chunk| {
                                                chunk.contents[0][y + 1][z] != 0
                                                    || (z != 0
                                                        && chunk.contents[0][y + 1][z - 1] != 0)
                                            }))
                                            || (x != CHUNK_WIDTH - 1
                                                && ((z == 0
                                                    && surrounding_chunks[3].map_or(
                                                        false,
                                                        |chunk| {
                                                            chunk.contents[x + 1][y + 1]
                                                                [CHUNK_DEPTH - 1]
                                                                != 0
                                                        },
                                                    ))
                                                    || (z != 0 && chunk[x + 1][y + 1][z - 1] != 0)
                                                    || chunk[x + 1][y + 1][z] != 0))
                                            || (z == 0
                                                && surrounding_chunks[3].map_or(false, |chunk| {
                                                    chunk.contents[x][y + 1][CHUNK_DEPTH - 1] != 0
                                                }))
                                            || (z != 0 && chunk[x][y + 1][z - 1] != 0)
                                        {
                                            0.3
                                        } else {
                                            0.0
                                        },
                                },
                            ])
                        }
                    }
                    // bottom face
                    if y == 0 || chunk[x][y - 1][z] == 0 {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        vertices.append(&mut vec![
                            // start of bottom
                            Vertex {
                                position: [1.0 + rel_x, y_f32, rel_z],
                                tex_coords: TOP_LEFT,
                                brightness: BOTTOM_BRIGHTNESS,
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: TOP_RIGHT,
                                brightness: BOTTOM_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_RIGHT,
                                brightness: BOTTOM_BRIGHTNESS,
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
                                brightness: BOTTOM_BRIGHTNESS,
                            },
                        ]);
                    }
                }
            }
        }
    }
    (vertices, indices)
}
