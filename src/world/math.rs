use vek::{Aabb, Vec3};

use crate::chunk::{CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH, CHUNK_WIDTH_I32};

#[inline(always)]
pub const fn block_index(x: usize, y: usize, z: usize) -> usize {
    // Y-first indexing optimizes vertical terrain column iteration
    y + CHUNK_HEIGHT * (x + CHUNK_WIDTH * z)
}

#[inline(always)]
pub const fn world_to_chunk_pos(x: i32, y: i32, z: i32) -> ([i32; 2], [usize; 3]) {
    let chunk_loc = [x.div_euclid(CHUNK_WIDTH_I32), z.div_euclid(CHUNK_DEPTH_I32)];
    let local_pos = [
        x.rem_euclid(CHUNK_WIDTH_I32) as usize,
        y as usize,
        z.rem_euclid(CHUNK_DEPTH_I32) as usize,
    ];
    (chunk_loc, local_pos)
}

#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn chunkcoord_to_aabb(coord: [i32; 2]) -> Aabb<f32> {
    let min = Vec3::new(
        (coord[0] * CHUNK_WIDTH_I32) as f32,
        0.0,
        (coord[1] * CHUNK_DEPTH_I32) as f32,
    );
    Aabb {
        min,
        max: min + Vec3::new(CHUNK_WIDTH as f32, CHUNK_HEIGHT as f32, CHUNK_DEPTH as f32),
    }
}
