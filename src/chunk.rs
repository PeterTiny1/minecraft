use noise::{NoiseFn, OpenSimplex};

use crate::Vertex;

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
    for i in -16..=16 {
        for j in -16..=16 {
            let distance = length(i, j);
            if distance <= 256
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
                            },
                            Vertex {
                                position: [rel_x, y_f32, zplusone],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, zplusone],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [1.0 + rel_x, 1.0 + y_f32, zplusone],
                                tex_coords: BOTTOM_LEFT,
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
                            },
                            Vertex {
                                position: [xplusone, y_f32, 1.0 + rel_z],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [xplusone, y_f32, rel_z],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [xplusone, 1.0 + y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
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
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, rel_z],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
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
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, 1.0 + y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_LEFT,
                            },
                        ]);
                    }
                    // top face
                    if y == chunk[x].len() - 1 || chunk[x][y + 1][z] == 0 {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        let yplusone = y_f32 + 1.0;
                        vertices.append(&mut vec![
                            Vertex {
                                position: [rel_x, yplusone, rel_z],
                                tex_coords: TOP_LEFT,
                            },
                            Vertex {
                                position: [rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [1.0 + rel_x, yplusone, rel_z],
                                tex_coords: BOTTOM_LEFT,
                            },
                        ]);
                    }
                    // bottom face
                    if y == 0 || chunk[x][y - 1][z] == 0 {
                        indices.extend(QUAD_INDICES.iter().map(|i| *i + vertices.len() as u32));
                        vertices.append(&mut vec![
                            // start of bottom
                            Vertex {
                                position: [1.0 + rel_x, y_f32, rel_z],
                                tex_coords: TOP_LEFT,
                            },
                            Vertex {
                                position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: TOP_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, y_f32, 1.0 + rel_z],
                                tex_coords: BOTTOM_RIGHT,
                            },
                            Vertex {
                                position: [rel_x, y_f32, rel_z],
                                tex_coords: BOTTOM_LEFT,
                            },
                        ]);
                    }
                }
            }
        }
    }
    (vertices, indices)
}
