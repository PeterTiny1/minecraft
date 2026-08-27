use std::{collections::HashMap, sync::Arc};
use vek::Vec3;

use crate::{
    block::BlockType,
    world::{
        math::resolve_block_target,
        types::{BlockProvider, ChunkDataStorage},
    },
};

/// Chunks affected by a block modification that require re-meshing.
#[derive(Debug, Default)]
pub struct BlockUpdateOutcome {
    pub dirty_chunks: Vec<[i32; 2]>,
}

#[derive(Default)]
pub struct WorldStorage {
    pub data: ChunkDataStorage,
}

impl WorldStorage {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    #[must_use]
    pub fn get_block(&self, pos: Vec3<i32>) -> Option<BlockType> {
        self.data.get_block(pos.x, pos.y, pos.z)
    }

    pub fn set_block(
        &mut self,
        target_pos: Vec3<i32>,
        block: BlockType,
    ) -> Option<BlockUpdateOutcome> {
        let (chunk_loc, idx, neighbor_offsets) = resolve_block_target(target_pos)?;
        let chunk_arc = self.data.get_mut(&chunk_loc)?;
        let chunk = Arc::make_mut(chunk_arc);

        if chunk.contents[idx] == block {
            return None;
        }

        chunk.contents[idx] = block;

        let mut outcome = BlockUpdateOutcome {
            dirty_chunks: Vec::with_capacity(1 + neighbor_offsets.len()),
        };

        // Self always re-meshes
        outcome.dirty_chunks.push(chunk_loc);

        // Re-mesh direct AND diagonal neighbor chunks affected by corner AO
        for [offset_x, offset_z] in neighbor_offsets.into_iter().flatten() {
            let n_loc = [chunk_loc[0] + offset_x, chunk_loc[1] + offset_z];
            if self.data.contains_key(&n_loc) {
                outcome.dirty_chunks.push(n_loc);
            }
        }

        Some(outcome)
    }
}
