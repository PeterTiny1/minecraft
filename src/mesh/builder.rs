use std::f32::consts::FRAC_1_SQRT_2;

use crate::{
    block::{self, BlockType},
    mesh::{LocatedChunk, textures::get_texture_indices},
    renderer::Vertex,
    world::{CHUNK_DEPTH_I32, CHUNK_HEIGHT_I32, CHUNK_WIDTH_I32, block_index},
};

// --- Mesh Constants ---
const TOP_BRIGHTNESS: f32 = 1.0;
const BOTTOM_BRIGHTNESS: f32 = 0.6;
const SIDE_BRIGHTNESS: f32 = 0.8;
const FRONT_BRIGHTNESS: f32 = 0.9;
const BACK_BRIGHTNESS: f32 = 0.7;

const CLOSE_CORNER: f32 = 0.5 + 0.5 * FRAC_1_SQRT_2;
const FAR_CORNER: f32 = 0.5 - 0.5 * FRAC_1_SQRT_2;
const CLOSE_FLOWER_CORNER: f32 = 0.716_506_35;

const BLOCK_WATER_HEIGHT: f32 = 0.5;
const WATER_V_TOP_OFF: u8 = (16.0 - BLOCK_WATER_HEIGHT * 16.0) as u8;

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];
const FLOWER_INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0];
const GRASS_INDICES: [u32; 24] = [
    0, 1, 2, 0, 2, 3, 3, 2, 0, 2, 1, 0, 4, 5, 6, 4, 6, 7, 7, 6, 4, 6, 5, 4,
];

// --- Internal Configuration Sub-structs ---
struct SolidFaceConfig {
    dir: (i32, i32, i32),
    tex_idx: usize,
    base_brightness: f32,
    ao_offsets: [((i32, i32, i32), (i32, i32, i32), (i32, i32, i32)); 4],
    v_pos: [[f32; 3]; 4],
}

struct LiquidSideFace {
    dir: (i32, i32, i32),
    tex_idx: usize,
    p0: (f32, f32),
    p1: (f32, f32),
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

const LIQUID_SIDE_FACES: [LiquidSideFace; 4] = [
    LiquidSideFace {
        dir: (0, 0, 1),
        tex_idx: 1,
        p0: (0.0, 1.0),
        p1: (1.0, 1.0),
    },
    LiquidSideFace {
        dir: (1, 0, 0),
        tex_idx: 2,
        p0: (1.0, 0.0),
        p1: (1.0, 1.0),
    },
    LiquidSideFace {
        dir: (0, 0, -1),
        tex_idx: 1,
        p0: (0.0, 0.0),
        p1: (1.0, 0.0),
    },
    LiquidSideFace {
        dir: (-1, 0, 0),
        tex_idx: 2,
        p0: (0.0, 0.0),
        p1: (0.0, 1.0),
    },
];

// --- Main Builder Struct ---
pub struct ChunkMeshBuilder<'a> {
    center: &'a LocatedChunk,
    neighbor_grid: [[Option<&'a LocatedChunk>; 3]; 3],
    local_pos: (i32, i32, i32),
    global_pos: (i32, i32, i32),
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl<'a> ChunkMeshBuilder<'a> {
    pub fn build(
        center: &'a LocatedChunk,
        neighbors: &'a [LocatedChunk],
    ) -> (Vec<Vertex>, Vec<u32>) {
        let mut builder = Self::new(center, neighbors);
        builder.generate_mesh();
        (builder.vertices, builder.indices)
    }

    fn new(center: &'a LocatedChunk, neighbors: &'a [LocatedChunk]) -> Self {
        let mut neighbor_grid = [[None; 3]; 3];
        let [cx, cz] = center.loc;

        for chunk in neighbors {
            let dx = chunk.loc[0] - cx;
            let dz = chunk.loc[1] - cz;
            if (-1..=1).contains(&dx) && (-1..=1).contains(&dz) {
                neighbor_grid[(dx + 1) as usize][(dz + 1) as usize] = Some(chunk);
            }
        }

        Self {
            center,
            neighbor_grid,
            local_pos: (0, 0, 0),
            global_pos: (0, 0, 0),
            vertices: Vec::with_capacity(8000),
            indices: Vec::with_capacity(12000),
        }
    }

    fn generate_mesh(&mut self) {
        let base_x = self.center.loc[0] * CHUNK_WIDTH_I32;
        let base_z = self.center.loc[1] * CHUNK_DEPTH_I32;
        let contents = &self.center.data.contents;

        for y in 0..CHUNK_HEIGHT_I32 {
            for z in 0..CHUNK_DEPTH_I32 {
                for x in 0..CHUNK_WIDTH_I32 {
                    let block_type = contents[block_index(x as usize, y as usize, z as usize)];
                    if block_type == BlockType::Air {
                        continue;
                    }

                    self.local_pos = (x, y, z);
                    self.global_pos = (base_x + x, y, base_z + z);

                    let tex_indices = get_texture_indices(block_type);

                    match block_type {
                        BlockType::Flower0 => self.generate_flower(tex_indices[0]),
                        bt if bt.is_liquid() => self.generate_liquid(tex_indices),
                        bt if bt.is_grasslike() => self.generate_grass(tex_indices[0]),
                        _ => self.generate_solid(tex_indices),
                    }
                }
            }
        }
    }

    #[inline]
    fn worldpos_f32(&self) -> [f32; 3] {
        [
            self.global_pos.0 as f32,
            self.global_pos.1 as f32,
            self.global_pos.2 as f32,
        ]
    }

    #[inline]
    fn extend_indices(&mut self, base_indices: &[u32]) {
        let base_len = u32::try_from(self.vertices.len()).expect("Vertex count exceeded u32 limit");
        self.indices
            .extend(base_indices.iter().map(|i| *i + base_len));
    }

    #[inline]
    fn get_block_at_offset(&self, dx: i32, dy: i32, dz: i32) -> Option<BlockType> {
        let target_y = self.local_pos.1 + dy;
        if !(0..CHUNK_HEIGHT_I32).contains(&target_y) {
            return Some(BlockType::Air);
        }

        let target_x = self.local_pos.0 + dx;
        let target_z = self.local_pos.2 + dz;

        if (0..CHUNK_WIDTH_I32).contains(&target_x) && (0..CHUNK_DEPTH_I32).contains(&target_z) {
            return Some(
                self.center.data.contents
                    [block_index(target_x as usize, target_y as usize, target_z as usize)],
            );
        }

        let c_dx = target_x.div_euclid(CHUNK_WIDTH_I32);
        let c_dz = target_z.div_euclid(CHUNK_DEPTH_I32);

        if !(-1..=1).contains(&c_dx) || !(-1..=1).contains(&c_dz) {
            return Some(BlockType::Air);
        }

        let rem_x = target_x.rem_euclid(CHUNK_WIDTH_I32);
        let rem_z = target_z.rem_euclid(CHUNK_DEPTH_I32);

        self.neighbor_grid[(c_dx + 1) as usize][(c_dz + 1) as usize].map(|chunk| {
            chunk.data.contents[block_index(rem_x as usize, target_y as usize, rem_z as usize)]
        })
    }

    #[inline]
    fn should_draw_face(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_none_or(block::BlockType::is_transparent)
    }

    #[inline]
    fn is_neighbor_liquid(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_some_and(block::BlockType::is_liquid)
    }

    #[inline]
    fn is_neighbor_solid(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.get_block_at_offset(dx, dy, dz)
            .is_some_and(|block| !block.is_transparent())
    }

    #[inline(always)]
    fn calculate_ao_light(
        &self,
        side1: bool,
        side2: bool,
        corner: bool,
        default_brightness: f32,
    ) -> u8 {
        let ao_level = if side1 && side2 {
            0
        } else {
            3 - (side1 as u8 + side2 as u8 + corner as u8)
        };

        let ao_factor = match ao_level {
            0 => 0.4,
            1 => 0.6,
            2 => 0.8,
            _ => 1.0,
        };

        (default_brightness * ao_factor * 255.0) as u8
    }

    fn generate_solid(&mut self, tex_indices: [u8; 6]) {
        let [x, y, z] = self.worldpos_f32();
        let uv_coords = [[0, 0], [0, 16], [16, 16], [16, 0]];

        for face in &SOLID_FACES {
            let (dx, dy, dz) = face.dir;
            if !self.should_draw_face(dx, dy, dz) {
                continue;
            }

            let tex_index = tex_indices[face.tex_idx];
            self.extend_indices(&QUAD_INDICES);

            for (i, &offset) in face.v_pos.iter().enumerate() {
                let (s1, s2, c) = face.ao_offsets[i];
                let light = self.calculate_ao_light(
                    self.is_neighbor_solid(s1.0, s1.1, s1.2),
                    self.is_neighbor_solid(s2.0, s2.1, s2.2),
                    self.is_neighbor_solid(c.0, c.1, c.2),
                    face.base_brightness,
                );

                let uv = uv_coords[i];
                self.vertices.push(Vertex {
                    position: [x + offset[0], y + offset[1], z + offset[2]],
                    data: [uv[0], uv[1], tex_index, light],
                });
            }
        }
    }

    #[inline]
    fn generate_grass(&mut self, tex_index: u8) {
        let [x, y, z] = self.worldpos_f32();
        self.extend_indices(&GRASS_INDICES);

        self.vertices.extend_from_slice(&[
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
    fn generate_flower(&mut self, tex_index: u8) {
        let [x, y, z] = self.worldpos_f32();
        self.extend_indices(&FLOWER_INDICES);

        self.vertices.extend_from_slice(&[
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

    #[inline]
    fn generate_liquid(&mut self, tex_indices: [u8; 6]) {
        let [x, y, z] = self.worldpos_f32();
        let is_submerged = self.is_neighbor_liquid(0, 1, 0);

        let (top_y, side_v_top) = if is_submerged {
            (y + 1.0, 0)
        } else {
            (y + BLOCK_WATER_HEIGHT, WATER_V_TOP_OFF)
        };

        // Top face
        if !is_submerged {
            let tex = tex_indices[0];
            self.extend_indices(&QUAD_INDICES);
            self.vertices.extend_from_slice(&[
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
        if !self.is_neighbor_liquid(0, -1, 0) && self.should_draw_face(0, -1, 0) {
            let tex = tex_indices[5];
            self.extend_indices(&QUAD_INDICES);
            self.vertices.extend_from_slice(&[
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
            if !self.is_neighbor_liquid(dx, dy, dz) && self.should_draw_face(dx, dy, dz) {
                let tex = tex_indices[face.tex_idx];
                let (x0, z0) = (x + face.p0.0, z + face.p0.1);
                let (x1, z1) = (x + face.p1.0, z + face.p1.1);

                self.extend_indices(&QUAD_INDICES);
                self.vertices.extend_from_slice(&[
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
}
