use crate::{
    BlockType, block,
    chunk::{CHUNK_DEPTH_I32, CHUNK_HEIGHT_I32, CHUNK_WIDTH_I32, block_index},
    mesh::LocatedChunk,
    renderer::Vertex,
};

pub struct MeshGenerationContext<'a> {
    pub center: &'a LocatedChunk,
    pub neighbors: &'a [LocatedChunk],
    pub local_x: i32,
    pub local_y: i32,
    pub local_z: i32,
    pub global_x: i32,
    pub global_y: i32,
    pub global_z: i32,
    pub indices: &'a mut Vec<u32>,
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
    pub fn extend_indices(&mut self, base_indices: &[u32]) {
        let len_index = u32::try_from(self.vertices.len()).expect("mesh count exceeded u32 limit");
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
