use crate::Vertex;

#[derive(Clone, Debug)]
pub struct Chunk {
    pub location: [i32; 2],
    pub contents: [[[u16; 16]; 256]; 16],
    pub mesh: Vec<Vertex>,
}

pub fn generate_chunk_mesh(
    location: [i32; 2],
    chunk: [[[u16; 16]; 256]; 16],
    north_chunk: Option<&Chunk>,
    south_chunk: Option<&Chunk>,
    east_chunk: Option<&Chunk>,
    west_chunk: Option<&Chunk>,
) -> Vec<Vertex> {
    let mut vertices = vec![];
    for x in 0..chunk.len() {
        for y in 0..chunk[x].len() {
            for z in 0..chunk[x][y].len() {
                if chunk[x][y][z] == 1 {
                    // first face
                    if (z == chunk[x][y].len() - 1
                        && (east_chunk.is_none() || east_chunk.unwrap().contents[x][y][0] == 0))
                        || (z != chunk[x][y].len() - 1 && chunk[x][y][z + 1] == 0)
                    {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                    // second face
                    if (x == chunk.len() - 1
                        && (north_chunk.is_none() || north_chunk.unwrap().contents[0][y][z] == 0))
                        || (x != chunk.len() - 1 && chunk[x + 1][y][z] == 0)
                    {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                    // third face
                    if (z == 0
                        && (west_chunk.is_none()
                            || west_chunk.unwrap().contents[x][y]
                                [west_chunk.unwrap().contents[x][y].len() - 1]
                                == 0))
                        || (z != 0 && chunk[x][y][z - 1] == 0)
                    {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                    // fourth face
                    if (x == 0
                        && (south_chunk.is_none()
                            || south_chunk.unwrap().contents
                                [south_chunk.unwrap().contents.len() - 1][y][z]
                                == 0))
                        || (x != 0 && chunk[x - 1][y][z] == 0)
                    {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                    // top face
                    if y == chunk[x].len() - 1 || chunk[x][y + 1][z] == 0 {
                        vertices.append(&mut vec![
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    1.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                    // bottom face
                    if y == 0 || chunk[x][y - 1][z] == 0 {
                        vertices.append(&mut vec![
                            // start of bottom
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    1.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [0.0, 0.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    1.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 1.0],
                            },
                            Vertex {
                                position: [
                                    0.0 + (x as i32 + (location[0] * 16)) as f32,
                                    0.0 + y as f32,
                                    0.0 + (z as i32 + (location[1] * 16)) as f32,
                                ],
                                tex_coords: [1.0, 0.0],
                            },
                        ]);
                    }
                }
            }
        }
    }
    vertices
}
