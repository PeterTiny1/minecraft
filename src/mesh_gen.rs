use std::{
    f32::consts::FRAC_1_SQRT_2,
    sync::{Arc, mpsc},
    thread,
};

use crate::{
    ChunkData,
    block::BlockType,
    chunk::{CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_HEIGHT_I32, CHUNK_WIDTH_I32, block_index},
    renderer::Vertex,
};

pub type Index = u32;

#[inline]
const fn get_texture_indices(block_type: BlockType) -> [u8; 6] {
    match block_type {
        BlockType::Stone => [0; 6],
        BlockType::Dirt => [1; 6],
        BlockType::GrassBlock0 => [2, 3, 3, 3, 3, 1],
        BlockType::GrassBlock1 => [4, 5, 5, 5, 5, 1],
        BlockType::GrassBlock2 => [6, 7, 7, 7, 7, 1],
        BlockType::BirchWood => [8, 9, 9, 9, 9, 8],
        BlockType::Wood => [10, 11, 11, 11, 11, 10],
        BlockType::DarkWood => [12, 13, 13, 13, 13, 12],
        BlockType::BirchLeaf => [14; 6],
        BlockType::Leaf => [15; 6],
        BlockType::DarkLeaf => [16; 6],
        BlockType::Grass0 => [17; 6],
        BlockType::Grass1 => [18; 6],
        BlockType::Grass2 => [19; 6],
        BlockType::Flower0 => [20; 6],
        BlockType::Flower1 => [21; 6],
        BlockType::Flower2 => [22; 6],
        BlockType::Sand => [23; 6],
        BlockType::Water => [24, 25, 25, 25, 25, 24],
        BlockType::Air => [0; 6],
    }
}

const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;
const AO_BRIGHTNESS: f32 = 0.5;

const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;

const FLOWER_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const GRASS_INDICES: [Index; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [Index; 6] = [0, 1, 2, 0, 2, 3];

pub struct MeshGenerationContext<'a> {
    pub center: &'a LocatedChunk,
    pub neighbors: &'a [LocatedChunk],
    pub local_x: i32,
    pub local_y: i32,
    pub local_z: i32,
    pub global_x: i32,
    pub global_y: i32,
    pub global_z: i32,
    pub indices: &'a mut Vec<Index>,
    pub vertices: &'a mut Vec<Vertex>,
}

impl MeshGenerationContext<'_> {
    #[allow(clippy::cast_precision_loss)]
    #[inline]
    pub const fn worldpos_f32(&self) -> [f32; 3] {
        [
            self.global_x as f32,
            self.global_y as f32,
            self.global_z as f32,
        ]
    }

    #[inline]
    pub fn extend_indices(&mut self, base_indices: &[Index]) {
        let len_index =
            Index::try_from(self.vertices.len()).expect("mesh count exceeded u32 limit");
        self.indices
            .extend(base_indices.iter().map(|i| *i + len_index));
    }

    fn get_block_at_offset(&self, dx: i32, dy: i32, dz: i32) -> Option<BlockType> {
        let target_y = self.local_y + dy;

        if target_y < 0 || target_y >= CHUNK_HEIGHT_I32 {
            return Some(BlockType::Air);
        }

        let target_x = self.local_x + dx;
        let target_z = self.local_z + dz;

        if (0..CHUNK_WIDTH_I32).contains(&target_x) && (0..CHUNK_DEPTH_I32).contains(&target_z) {
            return Some(
                self.center.data.contents
                    [block_index(target_x as usize, target_y as usize, target_z as usize)],
            );
        }

        let mut target_chunk_x = self.center.loc[0];
        let mut target_chunk_z = self.center.loc[1];
        let mut rem_x = target_x;
        let mut rem_z = target_z;

        if target_x < 0 {
            target_chunk_x -= 1;
            rem_x += CHUNK_WIDTH_I32;
        } else if target_x >= CHUNK_WIDTH_I32 {
            target_chunk_x += 1;
            rem_x -= CHUNK_WIDTH_I32;
        }

        if target_z < 0 {
            target_chunk_z -= 1;
            rem_z += CHUNK_DEPTH_I32;
        } else if target_z >= CHUNK_DEPTH_I32 {
            target_chunk_z += 1;
            rem_z -= CHUNK_DEPTH_I32;
        }

        self.neighbors
            .iter()
            .find(|n| n.loc == [target_chunk_x, target_chunk_z])
            .map(|neighbor| {
                neighbor.data.contents
                    [block_index(rem_x as usize, target_y as usize, rem_z as usize)]
            })
    }

    #[inline]
    pub fn should_draw_face(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_none_or(super::block::BlockType::is_transparent)
    }

    #[inline]
    pub fn is_neighbor_liquid(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_some_and(super::block::BlockType::is_liquid)
    }

    #[inline]
    pub fn is_neighbor_solid(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.get_block_at_offset(dx, dy, dz)
            .is_some_and(|block| !block.is_transparent())
    }
}

pub fn generate_chunk_mesh(
    chunk: &LocatedChunk,
    neighbours: &[LocatedChunk],
) -> (Vec<Vertex>, Vec<Index>) {
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

pub struct LocatedChunk {
    pub loc: [i32; 2],
    pub data: Arc<ChunkData>,
}

pub struct MeshJob {
    pub chunk: LocatedChunk,
    // NORTH CLOCKWISE
    pub neighbours: Vec<LocatedChunk>,
}

pub fn start_meshgen(
    recv_generate: mpsc::Receiver<MeshJob>,
    send_chunk: mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || manage_meshgen(&recv_generate, &send_chunk))
}

fn manage_meshgen(
    recv_generate: &mpsc::Receiver<MeshJob>,
    send_chunk: &mpsc::SyncSender<(Vec<Vertex>, Vec<Index>, [i32; 2])>,
) {
    let mut waiting: Vec<(Vec<Vertex>, Vec<Index>, [i32; 2])> = Vec::new();

    loop {
        // 1. Drain pending waiting items without cloning heavy vertex arrays
        let mut still_waiting = Vec::new();
        for item in waiting.drain(..) {
            if let Err(mpsc::TrySendError::Full(rejected)) = send_chunk.try_send(item) {
                still_waiting.push(rejected);
            }
        }
        waiting = still_waiting;

        // 2. Fetch next job
        let job = if waiting.is_empty() {
            match recv_generate.recv() {
                Ok(j) => j,
                Err(_) => break, // Channel closed
            }
        } else {
            match recv_generate.try_recv() {
                Ok(j) => j,
                Err(mpsc::TryRecvError::Empty) => {
                    thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        };

        // 3. Process mesh job
        let (mesh, indices) = generate_chunk_mesh(&job.chunk, &job.neighbours);
        let result = (mesh, indices, job.chunk.loc);

        if let Err(mpsc::TrySendError::Full(item)) = send_chunk.try_send(result) {
            waiting.push(item);
        }
    }
}
