use half::f16;
use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    block::BlockType,
    chunk::{BlockProvider, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH_I32},
    renderer::Vertex,
};

const TEXTURE_WIDTH: f32 = 1.0 / 16.0;
pub type Index = u32;
#[inline]
const fn get_texture_offsets(block_type: BlockType) -> [[f32; 2]; 6] {
    const TEXTURE_WIDTH_2: f32 = TEXTURE_WIDTH * 2.;
    const TEXTURE_WIDTH_3: f32 = TEXTURE_WIDTH * 3.;
    const TEXTURE_WIDTH_4: f32 = TEXTURE_WIDTH * 4.;
    const TEXTURE_WIDTH_5: f32 = TEXTURE_WIDTH * 5.;
    const TEXTURE_WIDTH_6: f32 = TEXTURE_WIDTH * 6.;
    const TEXTURE_WIDTH_7: f32 = TEXTURE_WIDTH * 7.;
    const TEXTURE_WIDTH_8: f32 = TEXTURE_WIDTH * 8.;
    const TEXTURE_WIDTH_9: f32 = TEXTURE_WIDTH * 9.;
    match block_type {
        BlockType::Stone => [[0., TEXTURE_WIDTH_8]; 6],
        BlockType::GrassBlock0 => [
            [0., TEXTURE_WIDTH],
            [TEXTURE_WIDTH, TEXTURE_WIDTH],
            [TEXTURE_WIDTH, TEXTURE_WIDTH],
            [TEXTURE_WIDTH, TEXTURE_WIDTH],
            [TEXTURE_WIDTH, TEXTURE_WIDTH],
            [0., 0.],
        ],
        BlockType::GrassBlock1 => [
            [0., TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_2],
            [0., 0.],
        ],
        BlockType::GrassBlock2 => [
            [0., TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH, TEXTURE_WIDTH_3],
            [0., 0.],
        ],
        BlockType::BirchWood => [
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH],
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH],
        ],
        BlockType::Wood => [
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_2],
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH_2],
        ],
        BlockType::DarkWood => [
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH_6, TEXTURE_WIDTH_3],
            [TEXTURE_WIDTH_7, TEXTURE_WIDTH_3],
        ],
        BlockType::BirchLeaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH]; 6],
        BlockType::Leaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH_2]; 6],
        BlockType::DarkLeaf => [[TEXTURE_WIDTH_5, TEXTURE_WIDTH_3]; 6],
        BlockType::Grass0 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH]; 6],
        BlockType::Grass1 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH_2]; 6],
        BlockType::Grass2 => [[TEXTURE_WIDTH_2, TEXTURE_WIDTH_3]; 6],
        BlockType::Flower0 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH]; 6],
        BlockType::Flower1 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH_2]; 6],
        BlockType::Flower2 => [[TEXTURE_WIDTH_3, TEXTURE_WIDTH_3]; 6],
        BlockType::Water => [
            [TEXTURE_WIDTH_4, 0.],
            [TEXTURE_WIDTH_5, 0.],
            [TEXTURE_WIDTH_5, 0.],
            [TEXTURE_WIDTH_5, 0.],
            [TEXTURE_WIDTH_5, 0.],
            [TEXTURE_WIDTH_4, 0.],
        ],
        BlockType::Sand => [[TEXTURE_WIDTH_9, 0.]; 6],
        BlockType::Dirt => [[0., 0.]; 6],
        BlockType::Air => [[0.; 2]; 6],
    }
}
const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;
const AO_BRIGHTNESS: f32 = 0.5;
#[test]
fn test_add_arrs() {
    assert_eq!(add_arrs([1.0, 2.0], [3.0, 4.0]), [4.0, 6.0]);
}

#[inline]
fn add_arrs(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}
const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const TOP_LEFT: [f32; 2] = [0.0, 0.0];
const TOP_RIGHT: [f32; 2] = [TEXTURE_WIDTH, 0.0];
const BOTTOM_LEFT: [f32; 2] = [0.0, TEXTURE_WIDTH];
const BOTTOM_RIGHT: [f32; 2] = [TEXTURE_WIDTH, TEXTURE_WIDTH];
#[inline]
fn create_grass_face(
    tex_offset: [f32; 2],
    world_position: [f32; 3],
    diagonal: bool,
) -> std::array::IntoIter<Vertex, 4> {
    let [x, y, z] = world_position;
    let (add0, add1) = if diagonal {
        (FAR_CORNER, CLOSE_CORNER)
    } else {
        (CLOSE_CORNER, FAR_CORNER)
    };
    [
        Vertex {
            position: [x + CLOSE_CORNER, y + 1.0, z + add0],
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: 1.0,
        },
        Vertex {
            position: [x + CLOSE_CORNER, y, z + add0],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: 1.0,
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + add1],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: 1.0,
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + add1],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: 1.0,
        },
    ]
    .into_iter()
}

const FLOWER_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const STEM_CORNERS: [[f32; 2]; 4] = [
    [TEXTURE_WIDTH * (3.0 + (6.0 / 16.0)), TEXTURE_WIDTH],
    [
        TEXTURE_WIDTH * (3.0 + (6.0 / 16.0)),
        TEXTURE_WIDTH * (23.0 / 16.0),
    ],
    [TEXTURE_WIDTH * 3.0, TEXTURE_WIDTH * (23.0 / 16.0)],
    [TEXTURE_WIDTH * 3.0, TEXTURE_WIDTH],
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
        },
        Vertex {
            position: [x + CLOSE_FLOWER_CORNER, y, z + FAR_CORNER],
            uv: STEM_CORNERS[1].map(f16::from_f32),
            light_level: 1.0,
        },
        Vertex {
            position: [x + FAR_CORNER, y, z + CLOSE_FLOWER_CORNER],
            uv: STEM_CORNERS[2].map(f16::from_f32),
            light_level: 1.0,
        },
        Vertex {
            position: [x + FAR_CORNER, y + 1.0, z + CLOSE_FLOWER_CORNER],
            uv: STEM_CORNERS[3].map(f16::from_f32),
            light_level: 1.0,
        },
    ]
    .into_iter()
}

const GRASS_INDICES: [Index; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];
const BIDIR_INDICES: [Index; 12] = [0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0];
const QUAD_INDICES: [Index; 6] = [0, 1, 2, 0, 2, 3];
const TOP_LEFT_WATER: [f32; 2] = [
    TOP_LEFT[0],
    TOP_LEFT[1] + TEXTURE_WIDTH * BLOCK_WATER_HEIGHT,
];
const TOP_RIGHT_WATER: [f32; 2] = [
    TOP_RIGHT[0],
    TOP_RIGHT[1] + TEXTURE_WIDTH * BLOCK_WATER_HEIGHT,
];

struct MeshGenerationContext<'a, T: BlockProvider> {
    world: &'a T,
    global_x: i32,
    global_y: i32,
    global_z: i32,
    indices: &'a mut Vec<Index>,
    vertices: &'a mut Vec<Vertex>,
}

impl<T: BlockProvider> MeshGenerationContext<'_, T> {
    #[allow(clippy::cast_precision_loss)]
    fn worldpos_f32(&self) -> [f32; 3] {
        [self.global_x, self.global_y, self.global_z].map(|c| c as f32)
    }
    fn extend_indicies(&mut self, base_indices: &[Index]) {
        let len_index = Index::try_from(self.vertices.len()).unwrap();
        self.indices
            .extend(base_indices.iter().map(|i| *i + len_index));
    }

    fn should_draw_face(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        let neighbor_pos = [
            self.global_x + offset_x,
            self.global_y + offset_y,
            self.global_z + offset_z,
        ];

        // Ask the world for the block
        self.world
            .get_block(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2])
            .is_none_or(super::block::BlockType::is_transparent)
    }
    fn is_neighbor_liquid(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        let neighbor_pos = [
            self.global_x + offset_x,
            self.global_y + offset_y,
            self.global_z + offset_z,
        ];

        // Ask the world for the block
        self.world
            .get_block(neighbor_pos[0], neighbor_pos[1], neighbor_pos[2])
            .is_some_and(super::block::BlockType::is_liquid)
    }

    pub fn is_neighbor_solid(&self, dx: i32, dy: i32, dz: i32) -> bool {
        // 1. Calculate Global Position
        let gx = self.global_x + dx;
        let gy = self.global_y + dy;
        let gz = self.global_z + dz;

        // 2. Check the World
        self.world
            .get_block(gx, gy, gz)
            .is_some_and(|block| !block.is_transparent())
    }
}

/// # Panics
///
/// If the chunk at the passed in coordinates doesn't exist
pub fn generate_chunk_mesh<T: BlockProvider>(
    world: &T,
    chunk_x: i32,
    chunk_z: i32,
) -> (Vec<Vertex>, Vec<Index>) {
    let (mut vertices, mut indices) = (vec![], vec![]);
    let base_x = chunk_x * CHUNK_WIDTH_I32;
    let base_z = chunk_z * CHUNK_DEPTH_I32;
    for x in 0..CHUNK_WIDTH_I32 {
        for y in 0..CHUNK_HEIGHT {
            for z in 0..CHUNK_DEPTH_I32 {
                let global_x = base_x + x;
                let global_y = y as i32;
                let global_z = base_z + z;
                let mut context = MeshGenerationContext {
                    world,
                    global_x,
                    global_y,
                    global_z,
                    indices: &mut indices,
                    vertices: &mut vertices,
                };
                let block_type = world.get_block(global_x, global_y, global_z).unwrap();
                let tex_offsets = get_texture_offsets(block_type);
                match block_type {
                    BlockType::Air => {}
                    BlockType::Flower0 => {
                        context.extend_indicies(&FLOWER_INDICES);
                        context
                            .vertices
                            .extend(generate_flower(context.worldpos_f32()));
                    }
                    _ if block_type.is_liquid() => {
                        generate_liquid(&mut context, tex_offsets);
                    }
                    _ if block_type.is_grasslike() => {
                        let tex_offset = tex_offsets[0];
                        let world_position = context.worldpos_f32();
                        context.extend_indicies(&GRASS_INDICES);
                        context.vertices.extend(create_grass_face(
                            tex_offset,
                            world_position,
                            false,
                        ));
                        context.vertices.extend(create_grass_face(
                            tex_offset,
                            world_position,
                            true,
                        ));
                    }
                    _ => {
                        generate_solid(&mut context, tex_offsets);
                    }
                }
            }
        }
    }
    (vertices, indices)
}

fn generate_solid<T: BlockProvider>(
    context: &mut MeshGenerationContext<T>,
    tex_offsets: [[f32; 2]; 6],
) {
    // --- North Face (Positive X) ---
    // The "should_draw_face" handles the edge check AND the neighbor check automatically.
    if context.should_draw_face(1, 0, 0) {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&QUAD_INDICES);

        // Note: For now, we use the AO path or No-AO path based on preference.
        // If you want to keep the "Fast Path vs Slow Path" logic,
        // you can check if the neighbor is None explicitly, but usually
        // passing 'None' to your AO calculator is fine (it treats missing blocks as bright air).
        context
            .vertices
            .append(&mut gen_face_pos_x_ao(context, tex_offset));
    }

    // --- South Face (Negative X) ---
    if context.should_draw_face(-1, 0, 0) {
        let tex_offset = tex_offsets[4];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_x_ao(context, tex_offset));
    }

    // --- Top Face (Positive Y) ---
    if context.should_draw_face(0, 1, 0) {
        let tex_offset = tex_offsets[0];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_pos_y_ao(context, tex_offset));
    }

    // --- Bottom Face (Negative Y) ---
    if context.should_draw_face(0, -1, 0) {
        let tex_offset = tex_offsets[5];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_y_ao(context, tex_offset));
    }

    // --- East Face (Positive Z) ---
    if context.should_draw_face(0, 0, 1) {
        let tex_offset = tex_offsets[1];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_pos_z_ao(context, tex_offset));
    }

    // --- West Face (Negative Z) ---
    if context.should_draw_face(0, 0, -1) {
        let tex_offset = tex_offsets[3];
        context.extend_indicies(&QUAD_INDICES);
        context
            .vertices
            .append(&mut gen_face_neg_z_ao(context, tex_offset));
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
fn gen_face_pos_x_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1,
        },
        Vertex {
            position: [xplusone, y_f32, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_2,
        },
        Vertex {
            position: [xplusone, y_f32, rel_z],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3,
        },
        Vertex {
            position: [xplusone, 1.0 + y_f32, rel_z],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_4,
        },
    ]
}
fn gen_face_neg_x_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_0, // <- Clean
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1, // <- Clean
        },
        Vertex {
            position: [rel_x, y_f32, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_2, // <- Clean
        },
        Vertex {
            position: [rel_x, 1.0 + y_f32, 1.0 + rel_z],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3, // <- Clean
        },
    ]
}

fn gen_face_pos_y_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_0,
        },
        Vertex {
            position: [rel_x, yplusone, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1,
        },
        Vertex {
            position: [1.0 + rel_x, yplusone, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_2,
        },
        Vertex {
            position: [1.0 + rel_x, yplusone, rel_z],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3,
        },
    ]
}

fn gen_face_neg_y_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_0,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1,
        },
        Vertex {
            position: [rel_x, y_f32, 1.0 + rel_z],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_2,
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3,
        },
    ]
}

fn gen_face_pos_z_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_0,
        },
        Vertex {
            position: [rel_x, y_f32, zplusone],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, zplusone],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_2,
        },
        Vertex {
            position: [1.0 + rel_x, 1.0 + y_f32, zplusone],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3,
        },
    ]
}

fn gen_face_neg_z_ao<T: BlockProvider>(
    context: &MeshGenerationContext<T>,
    tex_offset: [f32; 2],
) -> Vec<Vertex> {
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
            uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_0,
        },
        Vertex {
            position: [1.0 + rel_x, y_f32, rel_z],
            uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
            light_level: light_1,
        },
        Vertex {
            position: [rel_x, y_f32, rel_z],
            uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_2,
        },
        Vertex {
            position: [rel_x, 1.0 + y_f32, rel_z],
            uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
            light_level: light_3,
        },
    ]
}

const BLOCK_WATER_HEIGHT: f32 = 0.5;

#[inline]
fn generate_liquid<T: BlockProvider>(
    context: &mut MeshGenerationContext<T>,
    tex_offsets: [[f32; 2]; 6],
) {
    let [rel_x, y_f32, rel_z] = context.worldpos_f32();
    let yplusoff = y_f32 + BLOCK_WATER_HEIGHT;
    if !context.is_neighbor_liquid(0, 1, 0) {
        let tex_offset = tex_offsets[0];
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex {
                position: [rel_x, yplusoff, rel_z],
                uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [rel_x, yplusoff, 1.0 + rel_z],
                uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [1.0 + rel_x, yplusoff, rel_z],
                uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
        ]);
    }
    if !context.is_neighbor_liquid(0, -1, 0) && context.should_draw_face(0, -1, 0) {
        let tex_offset = tex_offsets[5];
        context.extend_indicies(&BIDIR_INDICES);
        context.vertices.append(&mut vec![
            Vertex {
                position: [rel_x, y_f32, rel_z],
                uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [rel_x, y_f32, 1.0 + rel_z],
                uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
            Vertex {
                position: [1.0 + rel_x, y_f32, rel_z],
                uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                light_level: TOP_BRIGHTNESS,
            },
        ]);
    }
    if !context.is_neighbor_liquid(0, 0, 1) && context.should_draw_face(0, 0, 1) {
        let tex_offset = tex_offsets[1];
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, 1.0 + rel_z],
                    uv: add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(1, 0, 0) && context.should_draw_face(1, 0, 0) {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [1.0 + rel_x, yplusoff, rel_z],
                    uv: add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(0, 0, -1) && context.should_draw_face(0, 0, -1) {
        let tex_offset = tex_offsets[1];
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, rel_z],
                    uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32 + 1.0, rel_z],
                    uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, rel_z],
                    uv: add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [1.0 + rel_x, yplusoff, rel_z],
                    uv: add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        }
    }
    if !context.is_neighbor_liquid(-1, 0, 0) && context.should_draw_face(-1, 0, 0) {
        let tex_offset = tex_offsets[2];
        context.extend_indicies(&BIDIR_INDICES);
        if context.is_neighbor_liquid(0, 1, 0) {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, y_f32 + 1.0, rel_z],
                    uv: add_arrs(TOP_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32 + 1.0, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        } else {
            context.vertices.append(&mut vec![
                Vertex {
                    position: [rel_x, yplusoff, rel_z],
                    uv: add_arrs(TOP_LEFT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, rel_z],
                    uv: add_arrs(BOTTOM_LEFT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, y_f32, 1.0 + rel_z],
                    uv: add_arrs(BOTTOM_RIGHT, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
                Vertex {
                    position: [rel_x, yplusoff, 1.0 + rel_z],
                    uv: add_arrs(TOP_RIGHT_WATER, tex_offset).map(f16::from_f32),
                    light_level: TOP_BRIGHTNESS,
                },
            ]);
        }
    }
}
