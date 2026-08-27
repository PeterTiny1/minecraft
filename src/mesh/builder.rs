use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    block::BlockType,
    mesh::{LocatedChunk, context::MeshGenerationContext, textures::get_texture_indices},
    renderer::Vertex,
    world::block_index,
    world::chunk::{CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH_I32},
};

// Shading constants
const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;
const AO_BRIGHTNESS: f32 = 0.5;

// Diagonal mesh geometry offsets
const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;

// Index buffers
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const BIDIR_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const FLOWER_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const GRASS_INDICES: [u32; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];

// Structural descriptors for solid block faces
struct SolidFaceConfig {
    dir: (i32, i32, i32),
    tex_idx: usize,
    base_brightness: f32,
    // (y1, z1), (y2, z2) for side 1/2 relative offsets used in AO calculation
    ao_offsets: [((i32, i32, i32), (i32, i32, i32), (i32, i32, i32)); 4],
    // Position offsets [x, y, z] for 4 vertices
    v_pos: [[f32; 3]; 4],
}

const SOLID_FACES: [SolidFaceConfig; 6] = [
    // +X (Right)
    SolidFaceConfig {
        dir: (1, 0, 0),
        tex_idx: 2,
        base_brightness: SIDE_BRIGHTNESS,
        ao_offsets: [
            ((1, 1, 0), (1, 0, -1), (1, 1, -1)),
            ((1, -1, 0), (1, 0, -1), (1, -1, -1)),
            ((1, -1, 0), (1, 0, 1), (1, -1, 1)),
            ((1, 1, 0), (1, 0, 1), (1, 1, 1)),
        ],
        v_pos: [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
    },
    // -X (Left)
    SolidFaceConfig {
        dir: (-1, 0, 0),
        tex_idx: 4,
        base_brightness: SIDE_BRIGHTNESS,
        ao_offsets: [
            ((-1, 1, 0), (-1, 0, -1), (-1, 1, -1)),
            ((-1, -1, 0), (-1, 0, -1), (-1, -1, -1)),
            ((-1, -1, 0), (-1, 0, 1), (-1, -1, 1)),
            ((-1, 1, 0), (-1, 0, 1), (-1, 1, 1)),
        ],
        v_pos: [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
    },
    // +Y (Top)
    SolidFaceConfig {
        dir: (0, 1, 0),
        tex_idx: 0,
        base_brightness: TOP_BRIGHTNESS,
        ao_offsets: [
            ((-1, 1, 0), (0, 1, -1), (-1, 1, -1)),
            ((-1, 1, 0), (0, 1, 1), (-1, 1, 1)),
            ((1, 1, 0), (0, 1, 1), (1, 1, 1)),
            ((1, 1, 0), (0, 1, -1), (1, 1, -1)),
        ],
        v_pos: [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
    },
    // -Y (Bottom)
    SolidFaceConfig {
        dir: (0, -1, 0),
        tex_idx: 5,
        base_brightness: BOTTOM_BRIGHTNESS,
        ao_offsets: [
            ((-1, -1, 0), (0, -1, -1), (-1, -1, -1)),
            ((-1, -1, 0), (0, -1, 1), (-1, -1, 1)),
            ((1, -1, 0), (0, -1, 1), (1, -1, 1)),
            ((1, -1, 0), (0, -1, -1), (1, -1, -1)),
        ],
        v_pos: [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
    },
    // +Z (Front)
    SolidFaceConfig {
        dir: (0, 0, 1),
        tex_idx: 1,
        base_brightness: FRONT_BRIGHTNESS,
        ao_offsets: [
            ((0, 1, 1), (-1, 0, 1), (-1, 1, 1)),
            ((0, -1, 1), (-1, 0, 1), (-1, -1, 1)),
            ((0, -1, 1), (1, 0, 1), (1, -1, 1)),
            ((0, 1, 1), (1, 0, 1), (1, 1, 1)),
        ],
        v_pos: [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
    },
    // -Z (Back)
    SolidFaceConfig {
        dir: (0, 0, -1),
        tex_idx: 3,
        base_brightness: BACK_BRIGHTNESS,
        ao_offsets: [
            ((0, 1, -1), (1, 0, -1), (1, 1, -1)),
            ((0, -1, -1), (1, 0, -1), (1, -1, -1)),
            ((0, -1, -1), (-1, 0, -1), (-1, -1, -1)),
            ((0, 1, -1), (-1, 0, -1), (-1, 1, -1)),
        ],
        v_pos: [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    },
];

pub fn generate(chunk: &LocatedChunk, neighbours: &[LocatedChunk]) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(8000);
    let mut indices = Vec::with_capacity(12000);

    let base_x = chunk.loc[0] * CHUNK_WIDTH_I32;
    let base_z = chunk.loc[1] * CHUNK_DEPTH_I32;
    let contents = &chunk.data.contents;

    for x in 0..CHUNK_WIDTH_I32 {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH_I32 {
                let block_type = contents[block_index(x as usize, y, z as usize)];
                if block_type == BlockType::Air {
                    continue;
                }

                let global_x = base_x + x;
                let global_y = y as i32;
                let global_z = base_z + z;

                let mut context = MeshGenerationContext {
                    center: chunk,
                    neighbors: neighbours,
                    local_x: x,
                    local_y: global_y,
                    local_z: z,
                    global_x,
                    global_y,
                    global_z,
                    indices: &mut indices,
                    vertices: &mut vertices,
                };

                let tex_indices = get_texture_indices(block_type);

                match block_type {
                    BlockType::Flower0 => generate_flower(&mut context),
                    bt if bt.is_liquid() => generate_liquid(&mut context, tex_indices),
                    bt if bt.is_grasslike() => generate_grass(&mut context, tex_indices[0]),
                    _ => generate_solid(&mut context, tex_indices),
                }
            }
        }
    }

    (vertices, indices)
}

#[inline(always)]
fn calculate_ao_light(side1: bool, side2: bool, corner: bool, default_brightness: f32) -> u8 {
    let factor = if side1 || side2 || corner {
        AO_BRIGHTNESS
    } else {
        default_brightness
    };
    (factor * 255.0) as u8
}

fn generate_solid(context: &mut MeshGenerationContext, tex_indices: [u8; 6]) {
    let [x, y, z] = context.worldpos_f32();
    let uv_coords = [[0, 0], [0, 16], [16, 16], [16, 0]];

    for face in &SOLID_FACES {
        let (dx, dy, dz) = face.dir;
        if !context.should_draw_face(dx, dy, dz) {
            continue;
        }

        let tex_index = tex_indices[face.tex_idx];
        context.extend_indices(&QUAD_INDICES);

        for (i, &offset) in face.v_pos.iter().enumerate() {
            let (s1, s2, c) = face.ao_offsets[i];
            let light = calculate_ao_light(
                context.is_neighbor_solid(s1.0, s1.1, s1.2),
                context.is_neighbor_solid(s2.0, s2.1, s2.2),
                context.is_neighbor_solid(c.0, c.1, c.2),
                face.base_brightness,
            );

            let uv = uv_coords[i];
            context.vertices.push(Vertex {
                position: [x + offset[0], y + offset[1], z + offset[2]],
                data: [uv[0], uv[1], tex_index, light],
            });
        }
    }
}

#[inline]
fn generate_grass(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();
    context.extend_indices(&GRASS_INDICES);

    context.vertices.extend_from_slice(&[
        // Diagonal 1
        Vertex {
            position: [x + CLOSE_CORNER, y + 1.0, z + FAR_CORNER],
            data: [0, 0, tex_index, 255],
        },
        Vertex {
            position: [x + CLOSE_CORNER, y, z + FAR_CORNER],
            data: [0, 16, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + CLOSE_CORNER],
            data: [16, 16, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + CLOSE_CORNER],
            data: [16, 0, tex_index, 255],
        },
        // Diagonal 2
        Vertex {
            position: [x + CLOSE_CORNER, y + 1.0, z + CLOSE_CORNER],
            data: [0, 0, tex_index, 255],
        },
        Vertex {
            position: [x + CLOSE_CORNER, y, z + CLOSE_CORNER],
            data: [0, 16, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + FAR_CORNER],
            data: [16, 16, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + FAR_CORNER],
            data: [16, 0, tex_index, 255],
        },
    ]);
}

#[inline]
fn generate_flower(context: &mut MeshGenerationContext) {
    let [x, y, z] = context.worldpos_f32();
    const TEX_INDEX: u8 = 20;

    context.extend_indices(&FLOWER_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y + 1.0, z + FAR_CORNER],
            data: [0, 0, TEX_INDEX, 255],
        },
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y, z + FAR_CORNER],
            data: [0, 7, TEX_INDEX, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + CLOSE_FLOWER_CORNER],
            data: [6, 7, TEX_INDEX, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + CLOSE_FLOWER_CORNER],
            data: [6, 0, TEX_INDEX, 255],
        },
    ]);
}

const BLOCK_WATER_HEIGHT: f32 = 0.5;
const WATER_V_TOP_OFF: u8 = (16.0 - BLOCK_WATER_HEIGHT * 16.0) as u8;

struct SideFace {
    dir: (i32, i32, i32),
    tex_idx: usize,
    p0: (f32, f32),
    p1: (f32, f32),
}

const LIQUID_SIDE_FACES: [SideFace; 4] = [
    SideFace {
        dir: (0, 0, 1),
        tex_idx: 1,
        p0: (0.0, 1.0),
        p1: (1.0, 1.0),
    }, // +Z
    SideFace {
        dir: (1, 0, 0),
        tex_idx: 2,
        p0: (1.0, 0.0),
        p1: (1.0, 1.0),
    }, // +X
    SideFace {
        dir: (0, 0, -1),
        tex_idx: 1,
        p0: (0.0, 0.0),
        p1: (1.0, 0.0),
    }, // -Z
    SideFace {
        dir: (-1, 0, 0),
        tex_idx: 2,
        p0: (0.0, 0.0),
        p1: (0.0, 1.0),
    }, // -X
];

#[inline]
fn generate_liquid(context: &mut MeshGenerationContext, tex_indices: [u8; 6]) {
    let [x, y, z] = context.worldpos_f32();
    let is_submerged = context.is_neighbor_liquid(0, 1, 0);

    let (top_y, side_v_top) = if is_submerged {
        (y + 1.0, 0)
    } else {
        (y + BLOCK_WATER_HEIGHT, WATER_V_TOP_OFF)
    };

    // Top face
    if !is_submerged {
        let tex = tex_indices[0];
        context.extend_indices(&BIDIR_INDICES);
        context.vertices.extend_from_slice(&[
            Vertex {
                position: [x, top_y, z],
                data: [0, 0, tex, 255],
            },
            Vertex {
                position: [x, top_y, 1.0 + z],
                data: [0, 16, tex, 255],
            },
            Vertex {
                position: [1.0 + x, top_y, 1.0 + z],
                data: [16, 16, tex, 255],
            },
            Vertex {
                position: [1.0 + x, top_y, z],
                data: [16, 0, tex, 255],
            },
        ]);
    }

    // Bottom face
    if !context.is_neighbor_liquid(0, -1, 0) && context.should_draw_face(0, -1, 0) {
        let tex = tex_indices[5];
        context.extend_indices(&BIDIR_INDICES);
        context.vertices.extend_from_slice(&[
            Vertex {
                position: [x, y, z],
                data: [0, 0, tex, 255],
            },
            Vertex {
                position: [x, y, 1.0 + z],
                data: [0, 16, tex, 255],
            },
            Vertex {
                position: [1.0 + x, y, 1.0 + z],
                data: [16, 16, tex, 255],
            },
            Vertex {
                position: [1.0 + x, y, z],
                data: [16, 0, tex, 255],
            },
        ]);
    }

    // Side faces
    for face in &LIQUID_SIDE_FACES {
        let (dx, dy, dz) = face.dir;
        if !context.is_neighbor_liquid(dx, dy, dz) && context.should_draw_face(dx, dy, dz) {
            let tex = tex_indices[face.tex_idx];
            let (x0, z0) = (x + face.p0.0, z + face.p0.1);
            let (x1, z1) = (x + face.p1.0, z + face.p1.1);

            context.extend_indices(&BIDIR_INDICES);
            context.vertices.extend_from_slice(&[
                Vertex {
                    position: [x0, top_y, z0],
                    data: [0, side_v_top, tex, 255],
                },
                Vertex {
                    position: [x0, y, z0],
                    data: [0, 16, tex, 255],
                },
                Vertex {
                    position: [x1, y, z1],
                    data: [16, 16, tex, 255],
                },
                Vertex {
                    position: [x1, top_y, z1],
                    data: [16, side_v_top, tex, 255],
                },
            ]);
        }
    }
}
