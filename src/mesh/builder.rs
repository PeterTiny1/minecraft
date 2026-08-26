use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    block::BlockType,
    chunk::{CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH_I32, block_index},
    mesh::{LocatedChunk, context::MeshGenerationContext, textures::get_texture_indices},
    renderer::Vertex,
};

const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;
const AO_BRIGHTNESS: f32 = 0.5;

const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;

const FLOWER_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const GRASS_INDICES: [u32; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

pub fn generate(chunk: &LocatedChunk, neighbours: &[LocatedChunk]) -> (Vec<Vertex>, Vec<u32>) {
    // Reserve typical chunk geometry capacity upfront to prevent re-allocations
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
                    local_y: y as i32,
                    local_z: z,
                    global_x,
                    global_y,
                    global_z,
                    indices: &mut indices,
                    vertices: &mut vertices,
                };

                let tex_indices = get_texture_indices(block_type);

                match block_type {
                    BlockType::Flower0 => {
                        generate_flower(&mut context);
                    }
                    _ if block_type.is_liquid() => {
                        generate_liquid(&mut context, tex_indices);
                    }
                    _ if block_type.is_grasslike() => {
                        generate_grass(&mut context, tex_indices[0]);
                    }
                    _ => {
                        generate_solid(&mut context, tex_indices);
                    }
                }
            }
        }
    }
    (vertices, indices)
}

#[inline]
const fn calculate_ao_light(
    side1_solid: bool,
    side2_solid: bool,
    corner_solid: bool,
    default_brightness: f32,
) -> f32 {
    if side1_solid || side2_solid || corner_solid {
        AO_BRIGHTNESS
    } else {
        default_brightness
    }
}

#[inline]
fn generate_grass(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();
    context.extend_indices(&GRASS_INDICES);

    context.vertices.extend_from_slice(&[
        // Face 1 (First diagonal)
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
        // Face 2 (Second diagonal)
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
    let tex_index = 20;
    context.extend_indices(&FLOWER_INDICES);

    context.vertices.extend_from_slice(&[
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y + 1.0, z + FAR_CORNER],
            data: [0, 0, tex_index, 255],
        },
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y, z + FAR_CORNER],
            data: [0, 7, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + CLOSE_FLOWER_CORNER],
            data: [6, 7, tex_index, 255],
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + CLOSE_FLOWER_CORNER],
            data: [6, 0, tex_index, 255],
        },
    ]);
}

fn generate_solid(context: &mut MeshGenerationContext, tex_indices: [u8; 6]) {
    if context.should_draw_face(1, 0, 0) {
        gen_face_pos_x(context, tex_indices[2]);
    }
    if context.should_draw_face(-1, 0, 0) {
        gen_face_neg_x(context, tex_indices[4]);
    }
    if context.should_draw_face(0, 1, 0) {
        gen_face_pos_y(context, tex_indices[0]);
    }
    if context.should_draw_face(0, -1, 0) {
        gen_face_neg_y(context, tex_indices[5]);
    }
    if context.should_draw_face(0, 0, 1) {
        gen_face_pos_z(context, tex_indices[1]);
    }
    if context.should_draw_face(0, 0, -1) {
        gen_face_neg_z(context, tex_indices[3]);
    }
}

fn gen_face_pos_x(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();
    let xplusone = x + 1.0;

    let s_y_plus = context.is_neighbor_solid(1, 1, 0);
    let s_y_minus = context.is_neighbor_solid(1, -1, 0);
    let s_z_plus = context.is_neighbor_solid(1, 0, 1);
    let s_z_minus = context.is_neighbor_solid(1, 0, -1);

    let c_y_plus_z_plus = context.is_neighbor_solid(1, 1, 1);
    let c_y_plus_z_minus = context.is_neighbor_solid(1, 1, -1);
    let c_y_minus_z_plus = context.is_neighbor_solid(1, -1, 1);
    let c_y_minus_z_minus = context.is_neighbor_solid(1, -1, -1);

    let l0 =
        (calculate_ao_light(s_y_plus, s_z_minus, c_y_plus_z_minus, SIDE_BRIGHTNESS) * 255.0) as u8;
    let l1 = (calculate_ao_light(s_y_minus, s_z_minus, c_y_minus_z_minus, SIDE_BRIGHTNESS) * 255.0)
        as u8;
    let l2 =
        (calculate_ao_light(s_y_minus, s_z_plus, c_y_minus_z_plus, SIDE_BRIGHTNESS) * 255.0) as u8;
    let l3 =
        (calculate_ao_light(s_y_plus, s_z_plus, c_y_plus_z_plus, SIDE_BRIGHTNESS) * 255.0) as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [xplusone, 1.0 + y, 1.0 + z],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [xplusone, y, 1.0 + z],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [xplusone, y, z],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [xplusone, 1.0 + y, z],
            data: [16, 0, tex_index, l3],
        },
    ]);
}

fn gen_face_neg_x(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();

    let s_y_plus = context.is_neighbor_solid(-1, 1, 0);
    let s_y_minus = context.is_neighbor_solid(-1, -1, 0);
    let s_z_plus = context.is_neighbor_solid(-1, 0, 1);
    let s_z_minus = context.is_neighbor_solid(-1, 0, -1);

    let c_y_plus_z_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_y_plus_z_plus = context.is_neighbor_solid(-1, 1, 1);
    let c_y_minus_z_minus = context.is_neighbor_solid(-1, -1, -1);
    let c_y_minus_z_plus = context.is_neighbor_solid(-1, -1, 1);

    let l0 =
        (calculate_ao_light(s_y_plus, s_z_minus, c_y_plus_z_minus, SIDE_BRIGHTNESS) * 255.0) as u8;
    let l1 = (calculate_ao_light(s_y_minus, s_z_minus, c_y_minus_z_minus, SIDE_BRIGHTNESS) * 255.0)
        as u8;
    let l2 =
        (calculate_ao_light(s_y_minus, s_z_plus, c_y_minus_z_plus, SIDE_BRIGHTNESS) * 255.0) as u8;
    let l3 =
        (calculate_ao_light(s_y_plus, s_z_plus, c_y_plus_z_plus, SIDE_BRIGHTNESS) * 255.0) as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [x, 1.0 + y, z],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [x, y, z],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [x, y, 1.0 + z],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [x, 1.0 + y, 1.0 + z],
            data: [16, 0, tex_index, l3],
        },
    ]);
}

fn gen_face_pos_y(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();
    let yplusone = y + 1.0;

    let s_x_minus = context.is_neighbor_solid(-1, 1, 0);
    let s_x_plus = context.is_neighbor_solid(1, 1, 0);
    let s_z_minus = context.is_neighbor_solid(0, 1, -1);
    let s_z_plus = context.is_neighbor_solid(0, 1, 1);

    let c_x_minus_z_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_x_plus_z_minus = context.is_neighbor_solid(1, 1, -1);
    let c_x_minus_z_plus = context.is_neighbor_solid(-1, 1, 1);
    let c_x_plus_z_plus = context.is_neighbor_solid(1, 1, 1);

    let l0 =
        (calculate_ao_light(s_x_minus, s_z_minus, c_x_minus_z_minus, TOP_BRIGHTNESS) * 255.0) as u8;
    let l1 =
        (calculate_ao_light(s_x_minus, s_z_plus, c_x_minus_z_plus, TOP_BRIGHTNESS) * 255.0) as u8;
    let l2 =
        (calculate_ao_light(s_x_plus, s_z_plus, c_x_plus_z_plus, TOP_BRIGHTNESS) * 255.0) as u8;
    let l3 =
        (calculate_ao_light(s_x_plus, s_z_minus, c_x_plus_z_minus, TOP_BRIGHTNESS) * 255.0) as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [x, yplusone, z],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [x, yplusone, 1.0 + z],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [1.0 + x, yplusone, 1.0 + z],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [1.0 + x, yplusone, z],
            data: [16, 0, tex_index, l3],
        },
    ]);
}

fn gen_face_neg_y(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();

    let s_x_minus = context.is_neighbor_solid(-1, -1, 0);
    let s_x_plus = context.is_neighbor_solid(1, -1, 0);
    let s_z_minus = context.is_neighbor_solid(0, -1, -1);
    let s_z_plus = context.is_neighbor_solid(0, -1, 1);

    let c_x_minus_z_minus = context.is_neighbor_solid(-1, -1, -1);
    let c_x_plus_z_minus = context.is_neighbor_solid(1, -1, -1);
    let c_x_minus_z_plus = context.is_neighbor_solid(-1, -1, 1);
    let c_x_plus_z_plus = context.is_neighbor_solid(1, -1, 1);

    let l0 = (calculate_ao_light(s_x_minus, s_z_minus, c_x_minus_z_minus, BOTTOM_BRIGHTNESS)
        * 255.0) as u8;
    let l1 = (calculate_ao_light(s_x_minus, s_z_plus, c_x_minus_z_plus, BOTTOM_BRIGHTNESS) * 255.0)
        as u8;
    let l2 =
        (calculate_ao_light(s_x_plus, s_z_plus, c_x_plus_z_plus, BOTTOM_BRIGHTNESS) * 255.0) as u8;
    let l3 = (calculate_ao_light(s_x_plus, s_z_minus, c_x_plus_z_minus, BOTTOM_BRIGHTNESS) * 255.0)
        as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [1.0 + x, y, z],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [1.0 + x, y, 1.0 + z],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [x, y, 1.0 + z],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [x, y, z],
            data: [16, 0, tex_index, l3],
        },
    ]);
}

fn gen_face_pos_z(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();
    let zplusone = z + 1.0;

    let s_y_plus = context.is_neighbor_solid(0, 1, 1);
    let s_y_minus = context.is_neighbor_solid(0, -1, 1);
    let s_x_plus = context.is_neighbor_solid(1, 0, 1);
    let s_x_minus = context.is_neighbor_solid(-1, 0, 1);

    let c_y_plus_x_plus = context.is_neighbor_solid(1, 1, 1);
    let c_y_plus_x_minus = context.is_neighbor_solid(-1, 1, 1);
    let c_y_minus_x_plus = context.is_neighbor_solid(1, -1, 1);
    let c_y_minus_x_minus = context.is_neighbor_solid(-1, -1, 1);

    let l0 =
        (calculate_ao_light(s_y_plus, s_x_minus, c_y_plus_x_minus, FRONT_BRIGHTNESS) * 255.0) as u8;
    let l1 = (calculate_ao_light(s_y_minus, s_x_minus, c_y_minus_x_minus, FRONT_BRIGHTNESS) * 255.0)
        as u8;
    let l2 =
        (calculate_ao_light(s_y_minus, s_x_plus, c_y_minus_x_plus, FRONT_BRIGHTNESS) * 255.0) as u8;
    let l3 =
        (calculate_ao_light(s_y_plus, s_x_plus, c_y_plus_x_plus, FRONT_BRIGHTNESS) * 255.0) as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [x, 1.0 + y, zplusone],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [x, y, zplusone],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [1.0 + x, y, zplusone],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [1.0 + x, 1.0 + y, zplusone],
            data: [16, 0, tex_index, l3],
        },
    ]);
}

fn gen_face_neg_z(context: &mut MeshGenerationContext, tex_index: u8) {
    let [x, y, z] = context.worldpos_f32();

    let s_y_plus = context.is_neighbor_solid(0, 1, -1);
    let s_y_minus = context.is_neighbor_solid(0, -1, -1);
    let s_x_plus = context.is_neighbor_solid(1, 0, -1);
    let s_x_minus = context.is_neighbor_solid(-1, 0, -1);

    let c_y_plus_x_plus = context.is_neighbor_solid(1, 1, -1);
    let c_y_plus_x_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_y_minus_x_plus = context.is_neighbor_solid(1, -1, -1);
    let c_y_minus_x_minus = context.is_neighbor_solid(-1, -1, -1);

    let l0 =
        (calculate_ao_light(s_y_plus, s_x_plus, c_y_plus_x_plus, BACK_BRIGHTNESS) * 255.0) as u8;
    let l1 =
        (calculate_ao_light(s_y_minus, s_x_plus, c_y_minus_x_plus, BACK_BRIGHTNESS) * 255.0) as u8;
    let l2 = (calculate_ao_light(s_y_minus, s_x_minus, c_y_minus_x_minus, BACK_BRIGHTNESS) * 255.0)
        as u8;
    let l3 =
        (calculate_ao_light(s_y_plus, s_x_minus, c_y_plus_x_minus, BACK_BRIGHTNESS) * 255.0) as u8;

    context.extend_indices(&QUAD_INDICES);
    context.vertices.extend_from_slice(&[
        Vertex {
            position: [1.0 + x, 1.0 + y, z],
            data: [0, 0, tex_index, l0],
        },
        Vertex {
            position: [1.0 + x, y, z],
            data: [0, 16, tex_index, l1],
        },
        Vertex {
            position: [x, y, z],
            data: [16, 16, tex_index, l2],
        },
        Vertex {
            position: [x, 1.0 + y, z],
            data: [16, 0, tex_index, l3],
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

const SIDE_FACES: [SideFace; 4] = [
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

    for face in &SIDE_FACES {
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
