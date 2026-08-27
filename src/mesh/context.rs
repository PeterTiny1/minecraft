use crate::{
    block::{self, BlockType},
    mesh::LocatedChunk,
    renderer::Vertex,
    world::{CHUNK_DEPTH_I32, CHUNK_HEIGHT_I32, CHUNK_WIDTH_I32, block_index},
};

pub struct MeshGenerationContext<'a> {
    pub center: &'a LocatedChunk,
    /// 3x3 array indexed by [dx + 1][dz + 1] relative to the center chunk
    pub neighbor_grid: [[Option<&'a LocatedChunk>; 3]; 3],
    pub local_x: i32,
    pub local_y: i32,
    pub local_z: i32,
    pub global_x: i32,
    pub global_y: i32,
    pub global_z: i32,
    pub indices: &'a mut Vec<u32>,
    pub vertices: &'a mut Vec<Vertex>,
}

impl<'a> MeshGenerationContext<'a> {
    pub fn new(
        center: &'a LocatedChunk,
        neighbors: &'a [LocatedChunk],
        indices: &'a mut Vec<u32>,
        vertices: &'a mut Vec<Vertex>,
    ) -> Self {
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
            local_x: 0,
            local_y: 0,
            local_z: 0,
            global_x: 0,
            global_y: 0,
            global_z: 0,
            indices,
            vertices,
        }
    }

    #[inline]
    pub const fn worldpos_f32(&self) -> [f32; 3] {
        [
            self.global_x as f32,
            self.global_y as f32,
            self.global_z as f32,
        ]
    }

    #[inline]
    pub fn extend_indices(&mut self, base_indices: &[u32]) {
        let base_len = u32::try_from(self.vertices.len()).expect("Vertex count exceeded u32 limit");
        self.indices
            .extend(base_indices.iter().map(|i| *i + base_len));
    }

    #[inline]
    pub fn get_block_at_offset(&self, dx: i32, dy: i32, dz: i32) -> Option<BlockType> {
        let target_y = self.local_y + dy;
        if !(0..CHUNK_HEIGHT_I32).contains(&target_y) {
            return Some(BlockType::Air);
        }

        let target_x = self.local_x + dx;
        let target_z = self.local_z + dz;

        // Fast path: fully within center chunk
        if (0..CHUNK_WIDTH_I32).contains(&target_x) && (0..CHUNK_DEPTH_I32).contains(&target_z) {
            return Some(
                self.center.data.contents
                    [block_index(target_x as usize, target_y as usize, target_z as usize)],
            );
        }

        // Slow path: neighbor lookup with proper 2D wrapping
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
    pub fn should_draw_face(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_none_or(block::BlockType::is_transparent)
    }

    #[inline]
    pub fn is_neighbor_liquid(&self, offset_x: i32, offset_y: i32, offset_z: i32) -> bool {
        self.get_block_at_offset(offset_x, offset_y, offset_z)
            .is_some_and(block::BlockType::is_liquid)
    }

    #[inline]
    pub fn is_neighbor_solid(&self, dx: i32, dy: i32, dz: i32) -> bool {
        self.get_block_at_offset(dx, dy, dz)
            .is_some_and(|block| !block.is_transparent())
    }
}
