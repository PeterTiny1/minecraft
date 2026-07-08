use half::f16;
use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    block::BlockType,
    chunk::{LocatedChunk, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH_I32},
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
const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [1.0, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, 1.0];
const BOTTOM_RIGHT: [f32; 2] = [1.0, 1.0];
#[inline]
fn create_grass_face(
    tex_index: u8,
    world_position: [f32; 3],
    diagonal: bool,
) -> std::array::IntoIter<Vertex, 4> {
    let [x, y, z] = world_position;
    let (add0, add1) = if diagonal {
        (FAR_CORNER, CLOSE_CORNER)
    } else {
        (CLOSE_CORNER, FAR_CORNER)
    };
    let tex_index = tex_index as u32;
    [
        Vertex {
            position: [x + CLOSE_CORNER, y + 1.0, z + add0],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: 1.0,
            tex_index,
        },
        Vertex {
            position: [x + CLOSE_CORNER, y, z + add0],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: 1.0,
            tex_index,
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + add1],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: 1.0,
            tex_index,
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + add1],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: 1.0,
            tex_index,
        },
    ]
    .into_iter()
}

const FLOWER_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const STEM_CORNERS: [[f32; 2]; 4] = [
    [6.0 / 16.0, 0.0],        // Top-right local coordinate
    [6.0 / 16.0, 7.0 / 16.0], // Bottom-right local coordinate (23/16 - 16/16)
    [0.0, 7.0 / 16.0],        // Bottom-left local coordinate
    [0.0, 0.0],               // Top-left local coordinate
];
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;
#[inline]
fn generate_flower(position: [f32; 3]) -> std::array::IntoIter<Vertex, 4> {
    let [x, y, z] = position;
    [
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y + 1.0, z + FAR_CORNER],
            uv: STEM_CORNERS[0].map(f16::from_f32),
            light_level: 1.0,
            tex_index: 20,
        },
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y, z + FAR_CORNER],
            uv: STEM_CORNERS[1].map(f16::from_f32),
            light_level: 1.0,
            tex_index: 20,
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + CLOSE_FLOWER_CORNER],
            uv: STEM_CORNERS[2].map(f16::from_f32),
            light_level: 1.0,
            tex_index: 20,
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + CLOSE_FLOWER_CORNER],
            uv: STEM_CORNERS[3].map(f16::from_f32),
            light_level: 1.0,
            tex_index: 20,
        },
    ]
    .into_iter()
}

const GRASS_INDICES: [Index; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [Index; 6] = [0, 1, 2, 0, 2, 3];
const TOP_LEFT_WATER: [f32; 2] = [TOP_LEFT[0], TOP_LEFT[1] + BLOCK_WATER_HEIGHT];
const TOP_RIGHT_WATER: [f32; 2] = [TOP_RIGHT[0], TOP_RIGHT[1] + BLOCK_WATER_HEIGHT];

pub struct MeshGenerationContext<'a> {
    pub center: &'a LocatedChunk,
    pub neighbors: &'a [LocatedChunk],
    // Storing both local and global positions prevents recalculations later
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
    pub const fn worldpos_f32(&self) -> [f32; 3] {
        [
            self.global_x as f32,
            self.global_y as f32,
            self.global_z as f32,
        ]
    }

    pub fn extend_indices(&mut self, base_indices: &[Index]) {
        let len_index = Index::try_from(self.vertices.len()).unwrap();
        self.indices
            .extend(base_indices.iter().map(|i| *i + len_index));
    }

    /// Internal Helper: Resolves a relative block offset locally or via neighbors
    fn get_block_at_offset(&self, dx: i32, dy: i32, dz: i32) -> Option<BlockType> {
        let target_y = self.local_y + dy;

        // 1. Height safety check (bedrock / sky limits)
        if target_y < 0 || target_y >= CHUNK_HEIGHT as i32 {
            return Some(BlockType::Air);
        }

        let target_x = self.local_x + dx;
        let target_z = self.local_z + dz;

        // 2. Fast Path: The block resides entirely inside the current center chunk
        if (0..CHUNK_WIDTH_I32).contains(&target_x) && (0..CHUNK_DEPTH_I32).contains(&target_z) {
            return Some(
                self.center.data.contents[target_x as usize][target_y as usize][target_z as usize],
            );
        }

        // 3. Slow Path: The offset crossed a chunk boundary. Determine target chunk coordinates.
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

        // 4. Query the neighborhood snapshot vec passed to this job
        self.neighbors
            .iter()
            .find(|n| n.loc == [target_chunk_x, target_chunk_z])
            .map(|neighbor| {
                neighbor.data.contents[rem_x as usize][target_y as usize][rem_z as usize]
            })
    }

    pub fn should_draw_face(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        // If the neighbor chunk isn't loaded (None), we default to drawing the face
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_none_or(super::block::BlockType::is_transparent)
    }

    pub fn is_neighbor_liquid(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_some_and(super::block::BlockType::is_liquid)
    }

    pub fn is_neighbor_solid(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.get_block_at_offset(dx, dy, dz)
            .is_some_and(|block| !block.is_transparent())
    }
    fn extend_indicies(&mut self, base_indices: &[Index]) {
        let len_index = Index::try_from(self.vertices.len()).unwrap();

        self.indices
            .extend(base_indices.iter().map(|i| *i + len_index));
    }
}

pub fn generate_chunk_mesh(
    chunk: &LocatedChunk,
    neighbours: &[LocatedChunk],
) -> (Vec<Vertex>, Vec<Index>) {
    let (mut vertices, mut indices) = (vec![], vec![]);

    let base_x = chunk.loc[0] * CHUNK_WIDTH_I32;
    let base_z = chunk.loc[1] * CHUNK_DEPTH_I32;

    // 1. Grab direct, lightning-fast reference to our local array data
    let contents = &chunk.data.contents;

    for x in 0..CHUNK_WIDTH_I32 {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH_I32 {
                // 2. Direct indexing into the dense heap array (Incredibly cache-friendly)
                let block_type = contents[x as usize][y][z as usize];
                if block_type == BlockType::Air {
                    continue;
                }

                let global_x = base_x + x;
                let global_y = y as i32;
                let global_z = base_z + z;

                // 3. Create a lightweight structure to pass down your local/neighbor context
                // (You can modify your MeshGenerationContext to own local pointers instead of a world trait)
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
                        context.extend_indices(&FLOWER_INDICES);
                        let world_pos = context.worldpos_f32();
                        context.vertices.extend(generate_flower(world_pos));
                    }
                    _ if block_type.is_liquid() => {
                        generate_liquid(&mut context, tex_indices);
                    }
                    _ if block_type.is_grasslike() => {
                        let tex_index = tex_indices[0];
                        let world_position = context.worldpos_f32();
                        context.extend_indices(&GRASS_INDICES);
                        context.vertices.extend(create_grass_face(
                            tex_index,
                            world_position,
                            false,
                        ));
                        context
                            .vertices
                            .extend(create_grass_face(tex_index, world_position, true));
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

fn generate_solid(context: &mut MeshGenerationContext, tex_indices: [u8; 6]) {
    // --- North Face (Positive X) ---
    // The "should_draw_face" handles the edge check AND the neighbor check automatically.
    if context.should_draw_face(1, 0, 0) {
        let tex_index = tex_indices[2];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_pos_x(context, tex_index));
    }

    // --- South Face (Negative X) ---
    if context.should_draw_face(-1, 0, 0) {
        let tex_offset = tex_indices[4];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_x(context, tex_offset));
    }

    // --- Top Face (Positive Y) ---
    if context.should_draw_face(0, 1, 0) {
        let tex_offset = tex_indices[0];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_pos_y(context, tex_offset));
    }

    // --- Bottom Face (Negative Y) ---
    if context.should_draw_face(0, -1, 0) {
        let tex_offset = tex_indices[5];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_y(context, tex_offset));
    }

    // --- East Face (Positive Z) ---
    if context.should_draw_face(0, 0, 1) {
        let tex_offset = tex_indices[1];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_pos_z(context, tex_offset));
    }

    // --- West Face (Negative Z) ---
    if context.should_draw_face(0, 0, -1) {
        let tex_offset = tex_indices[3];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_z(context, tex_offset));
    }
}

/// Calculates the AO brightness for a vertex based on its three neighbors.
#[inline]
const fn calculate_ao_light(
    side1_solid: bool,
    side2_solid: bool,
    corner_solid: bool,
    default_brightness: f32,
) -> f32 {
    // This is the Minecraft-style AO logic you're using:
    // If any of the 3 neighbors are solid, apply AO.
    if side1_solid || side2_solid || corner_solid {
        AO_BRIGHTNESS
    } else {
        default_brightness
    }

    // ---
    // Note for later: A more advanced (but slower) AO model
    // would be:
    // let mut occlusion = 0;
    // if side1_solid { occlusion += 1; }
    // if side2_solid { occlusion += 1; }
    // if corner_solid && side1_solid && side2_solid { occlusion += 1; }
    // return default_brightness / (1.0 + occlusion * AO_FACTOR);
    // ---
    // But for now, we'll stick to previous logic.
}
fn gen_face_pos_x(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    let [xplusone, y_f32, rel_z] = [
        context.worldpos_f32()[0] + 1.0,
        context.worldpos_f32()[1],
        context.worldpos_f32()[2],
    ];
    // Sides
    let s_y_plus = context.is_neighbor_solid(1, 1, 0);
    let s_y_minus = context.is_neighbor_solid(1, -1, 0);
    let s_z_plus = context.is_neighbor_solid(1, 0, 1);
    let s_z_minus = context.is_neighbor_solid(1, 0, -1);
    // Corners
    let c_y_plus_z_plus = context.is_neighbor_solid(1, 1, 1);
    let c_y_plus_z_minus = context.is_neighbor_solid(1, 1, -1);
    let c_y_minus_z_plus = context.is_neighbor_solid(1, -1, 1);
    let c_y_minus_z_minus = context.is_neighbor_solid(1, -1, -1);
    // Calculate AO for each vertex
    let light_1 = calculate_ao_light(s_y_plus, s_z_minus, c_y_plus_z_minus, SIDE_BRIGHTNESS);
    let light_2 = calculate_ao_light(s_y_minus, s_z_minus, c_y_minus_z_minus, SIDE_BRIGHTNESS);
    let light_3 = calculate_ao_light(s_y_minus, s_z_plus, c_y_minus_z_plus, SIDE_BRIGHTNESS);
    let light_4 = calculate_ao_light(s_y_plus, s_z_plus, c_y_plus_z_plus, SIDE_BRIGHTNESS);
    vec![
        Vertex {
            position: [xplusone, 1.0 + y_f32, 1.0 + rel_z],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [xplusone, y_f32, 1.0 + rel_z],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [xplusone, y_f32, rel_z],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
        Vertex {
            position: [xplusone, 1.0 + y_f32, rel_z],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_4,
            tex_index,
        },
    ]
}
fn gen_face_neg_x(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    // --- 1. Get Coordinates ---
    let [rel_x, y_f32, rel_z] = [
        context.worldpos_f32()[0],
        context.worldpos_f32()[1],
        context.worldpos_f32()[2],
    ];

    // --- 2. Pre-calculate Neighbors (Naming Scheme for -X Face) ---
    // All neighbors have an x_offset of -1
    // "Sides"
    let s_y_plus = context.is_neighbor_solid(-1, 1, 0);
    let s_y_minus = context.is_neighbor_solid(-1, -1, 0);
    let s_z_plus = context.is_neighbor_solid(-1, 0, 1);
    let s_z_minus = context.is_neighbor_solid(-1, 0, -1);

    // "Corners"
    let c_y_plus_z_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_y_plus_z_plus = context.is_neighbor_solid(-1, 1, 1);
    let c_y_minus_z_minus = context.is_neighbor_solid(-1, -1, -1);
    let c_y_minus_z_plus = context.is_neighbor_solid(-1, -1, 1);

    // --- 3. Calculate AO for each vertex ---
    let light_0 = calculate_ao_light(s_y_plus, s_z_minus, c_y_plus_z_minus, SIDE_BRIGHTNESS); // Top-Left
    let light_1 = calculate_ao_light(s_y_minus, s_z_minus, c_y_minus_z_minus, SIDE_BRIGHTNESS); // Bottom-Left
    let light_2 = calculate_ao_light(s_y_minus, s_z_plus, c_y_minus_z_plus, SIDE_BRIGHTNESS); // Bottom-Right
    let light_3 = calculate_ao_light(s_y_plus, s_z_plus, c_y_plus_z_plus, SIDE_BRIGHTNESS); // Top-Right

    // --- 4. Build the Vec ---
    vec![
        Vertex {
            position: [rel_x, 1.0 + y_f32, rel_z],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_0,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, 1.0 + rel_z],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [rel_x, 1.0 + y_f32, 1.0 + rel_z],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
    ]
}

fn gen_face_pos_y(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    let [rel_x, yplusone, rel_z] = [
        context.worldpos_f32()[0],
        context.worldpos_f32()[1] + 1.0,
        context.worldpos_f32()[2],
    ];

    // --- 1. Pre-calculate all 8 neighbor solid states (0 repeats!) ---
    // "Sides"
    let s_x_minus = context.is_neighbor_solid(-1, 1, 0);
    let s_x_plus = context.is_neighbor_solid(1, 1, 0);
    let s_z_minus = context.is_neighbor_solid(0, 1, -1);
    let s_z_plus = context.is_neighbor_solid(0, 1, 1);

    // "Corners"
    let c_x_minus_z_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_x_plus_z_minus = context.is_neighbor_solid(1, 1, -1);
    let c_x_minus_z_plus = context.is_neighbor_solid(-1, 1, 1);
    let c_x_plus_z_plus = context.is_neighbor_solid(1, 1, 1);

    // --- 2. Calculate AO light for each vertex ---
    let light_0 = calculate_ao_light(s_x_minus, s_z_minus, c_x_minus_z_minus, TOP_BRIGHTNESS); // Top-Left
    let light_1 = calculate_ao_light(s_x_minus, s_z_plus, c_x_minus_z_plus, TOP_BRIGHTNESS); // Bottom-Left
    let light_2 = calculate_ao_light(s_x_plus, s_z_plus, c_x_plus_z_plus, TOP_BRIGHTNESS); // Bottom-Right
    let light_3 = calculate_ao_light(s_x_plus, s_z_minus, c_x_plus_z_minus, TOP_BRIGHTNESS); // Top-Right

    // --- 3. Build the vec! ---
    vec![
        Vertex {
            position: [rel_x, yplusone, rel_z],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_0,
            tex_index,
        },
        Vertex {
            position: [rel_x, yplusone, 1.0 + rel_z],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, yplusone, rel_z],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
    ]
}

fn gen_face_neg_y(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    let [rel_x, y_f32, rel_z] = [
        context.worldpos_f32()[0],
        context.worldpos_f32()[1],
        context.worldpos_f32()[2],
    ];
    // --- 1. Pre-calculate all 8 neighbor solid states (0 repeats!) ---
    // "Sides"
    let s_x_minus = context.is_neighbor_solid(-1, -1, 0);
    let s_x_plus = context.is_neighbor_solid(1, -1, 0);
    let s_z_minus = context.is_neighbor_solid(0, -1, -1);
    let s_z_plus = context.is_neighbor_solid(0, -1, 1);

    // "Corners"
    let c_x_minus_z_minus = context.is_neighbor_solid(-1, -1, -1);
    let c_x_plus_z_minus = context.is_neighbor_solid(1, -1, -1);
    let c_x_minus_z_plus = context.is_neighbor_solid(-1, -1, 1);
    let c_x_plus_z_plus = context.is_neighbor_solid(1, -1, 1);

    // --- 2. Calculate AO light for each vertex ---
    let light_0 = calculate_ao_light(s_x_minus, s_z_minus, c_x_minus_z_minus, BOTTOM_BRIGHTNESS); // Top-Left
    let light_1 = calculate_ao_light(s_x_minus, s_z_plus, c_x_minus_z_plus, BOTTOM_BRIGHTNESS); // Bottom-Left
    let light_2 = calculate_ao_light(s_x_plus, s_z_plus, c_x_plus_z_plus, BOTTOM_BRIGHTNESS); // Bottom-Right
    let light_3 = calculate_ao_light(s_x_plus, s_z_minus, c_x_plus_z_minus, BOTTOM_BRIGHTNESS); // Top-Right
    vec![
        Vertex {
            position: [1.0 + rel_x, y_f32, rel_z],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_0,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, 1.0 + rel_z],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
    ]
}

fn gen_face_pos_z(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    // --- 1. Get Coordinates ---
    let [rel_x, y_f32, rel_z] = [
        context.worldpos_f32()[0],
        context.worldpos_f32()[1],
        context.worldpos_f32()[2],
    ];
    let zplusone = rel_z + 1.0;

    // --- 2. Pre-calculate Neighbors (Naming Scheme for +Z Face) ---
    // All neighbors have a z_offset of +1
    // "Sides"
    let s_y_plus = context.is_neighbor_solid(0, 1, 1);
    let s_y_minus = context.is_neighbor_solid(0, -1, 1);
    let s_x_plus = context.is_neighbor_solid(1, 0, 1);
    let s_x_minus = context.is_neighbor_solid(-1, 0, 1);

    // "Corners"
    let c_y_plus_x_plus = context.is_neighbor_solid(1, 1, 1);
    let c_y_plus_x_minus = context.is_neighbor_solid(-1, 1, 1);
    let c_y_minus_x_plus = context.is_neighbor_solid(1, -1, 1);
    let c_y_minus_x_minus = context.is_neighbor_solid(-1, -1, 1);

    // --- 3. Calculate AO for each vertex ---
    let light_0 = calculate_ao_light(s_y_plus, s_x_minus, c_y_plus_x_minus, FRONT_BRIGHTNESS); // Top-Left
    let light_1 = calculate_ao_light(s_y_minus, s_x_minus, c_y_minus_x_minus, FRONT_BRIGHTNESS); // Bottom-Left
    let light_2 = calculate_ao_light(s_y_minus, s_x_plus, c_y_minus_x_plus, FRONT_BRIGHTNESS); // Bottom-Right
    let light_3 = calculate_ao_light(s_y_plus, s_x_plus, c_y_plus_x_plus, FRONT_BRIGHTNESS); // Top-Right

    // --- 4. Build the Vec ---
    vec![
        Vertex {
            position: [rel_x, 1.0 + y_f32, zplusone],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_0,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, zplusone],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, zplusone],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, 1.0 + y_f32, zplusone],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
    ]
}

fn gen_face_neg_z(context: &MeshGenerationContext, tex_index: u8) -> Vec<Vertex> {
    let tex_index = tex_index as u32;
    // --- 1. Get Coordinates ---
    let [rel_x, y_f32, rel_z] = [
        context.worldpos_f32()[0],
        context.worldpos_f32()[1],
        context.worldpos_f32()[2],
    ];

    // --- 2. Pre-calculate Neighbors (Naming Scheme for -Z Face) ---
    // All neighbors have a z_offset of -1
    // "Sides"
    let s_y_plus = context.is_neighbor_solid(0, 1, -1);
    let s_y_minus = context.is_neighbor_solid(0, -1, -1);
    let s_x_plus = context.is_neighbor_solid(1, 0, -1);
    let s_x_minus = context.is_neighbor_solid(-1, 0, -1);

    // "Corners"
    let c_y_plus_x_plus = context.is_neighbor_solid(1, 1, -1);
    let c_y_plus_x_minus = context.is_neighbor_solid(-1, 1, -1);
    let c_y_minus_x_plus = context.is_neighbor_solid(1, -1, -1);
    let c_y_minus_x_minus = context.is_neighbor_solid(-1, -1, -1);

    // --- 3. Calculate AO for each vertex ---
    let light_0 = calculate_ao_light(s_y_plus, s_x_plus, c_y_plus_x_plus, BACK_BRIGHTNESS); // Top-Left
    let light_1 = calculate_ao_light(s_y_minus, s_x_plus, c_y_minus_x_plus, BACK_BRIGHTNESS); // Bottom-Left
    let light_2 = calculate_ao_light(s_y_minus, s_x_minus, c_y_minus_x_minus, BACK_BRIGHTNESS); // Bottom-Right
    let light_3 = calculate_ao_light(s_y_plus, s_x_minus, c_y_plus_x_minus, BACK_BRIGHTNESS); // Top-Right

    // --- 4. Build the Vec ---
    vec![
        Vertex {
            position: [1.0 + rel_x, 1.0 + y_f32, rel_z],
            uv: TOP_LEFT.map(f16::from_f32),
            light_level: light_0,
            tex_index,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, rel_z],
            uv: BOTTOM_LEFT.map(f16::from_f32),
            light_level: light_1,
            tex_index,
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: BOTTOM_RIGHT.map(f16::from_f32),
            light_level: light_2,
            tex_index,
        },
        Vertex {
            position: [rel_x, 1.0 + y_f32, rel_z],
            uv: TOP_RIGHT.map(f16::from_f32),
            light_level: light_3,
            tex_index,
        },
    ]
}

const BLOCK_WATER_HEIGHT: f32 = 0.5;

#[inline]
fn generate_liquid(context: &mut MeshGenerationContext, tex_indices: [u8; 6]) {
    let [rel_x, y_f32, rel_z] = context.worldpos_f32();
    let yplusoff = y_f32 + BLOCK_WATER_HEIGHT;
    if !context.is_neighbor_liquid(0, 1, 0) {
        let tex_index = tex_indices[0] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex {
                position: [rel_x, yplusoff, rel_z],
                uv: TOP_LEFT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [rel_x, yplusoff, 1.0 + rel_z],
                uv: BOTTOM_LEFT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                uv: BOTTOM_RIGHT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [1.0 + rel_x, yplusoff, rel_z],
                uv: TOP_RIGHT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
        ]);
    }
    if !context.is_neighbor_liquid(0, -1, 0) && context.should_draw_face(0, -1, 0) {
        let tex_index = tex_indices[5] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex {
                position: [rel_x, y_f32, rel_z],
                uv: TOP_LEFT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [rel_x, y_f32, 1.0 + rel_z],
                uv: BOTTOM_LEFT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                uv: BOTTOM_RIGHT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
            Vertex {
                position: [1.0 + rel_x, y_f32, rel_z],
                uv: TOP_RIGHT.map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
                tex_index,
            },
        ]);
    }
    if !context.is_neighbor_liquid(0, 0, 1) && context.should_draw_face(0, 0, 1) {
        let tex_index = tex_indices[1] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: TOP_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: TOP_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, 1.0 + rel_z],
                    uv: TOP_LEFT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    uv: TOP_RIGHT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(1, 0, 0) && context.should_draw_face(1, 0, 0) {
        let tex_index = tex_indices[2] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    uv: TOP_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: TOP_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [1.0 + rel_x, yplusoff, rel_z],
                    uv: TOP_LEFT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    uv: TOP_RIGHT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(0, 0, -1) && context.should_draw_face(0, 0, -1) {
        let tex_index = tex_indices[1] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, rel_z],
                    uv: TOP_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    uv: TOP_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, rel_z],
                    uv: TOP_LEFT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, rel_z],
                    uv: TOP_RIGHT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(-1, 0, 0) && context.should_draw_face(-1, 0, 0) {
        let tex_index = tex_indices[2] as u32;
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, rel_z],
                    uv: TOP_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: TOP_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, rel_z],
                    uv: TOP_LEFT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: BOTTOM_LEFT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: BOTTOM_RIGHT.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
                Vertex {
                    position: [rel_x, yplusoff, 1.0 + rel_z],
                    uv: TOP_RIGHT_WATER.map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                    tex_index,
                },
            ]);
        }
    }
}
