use vek::{Aabb, Vec3};

use crate::{
    RENDER_DISTANCE, camera,
    renderer::cuboid_intersects_frustum,
    world::chunk::{
        CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH, CHUNK_WIDTH_I32, ChunkDataStorage,
    },
};

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
const MAX_DISTANCE_X: i32 = RENDER_DISTANCE as i32 / CHUNK_WIDTH_I32 + 1;
const MAX_DISTANCE_Y: i32 = RENDER_DISTANCE as i32 / CHUNK_DEPTH_I32 + 1;
const RENDER_DISTANCE_CHUNKS: i32 = if MAX_DISTANCE_X > MAX_DISTANCE_Y {
    MAX_DISTANCE_X
} else {
    MAX_DISTANCE_Y
};
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn nearest_visible_unloaded(
    generated_chunks: &ChunkDataStorage,
    camera: &camera::Camera,
) -> Option<[i32; 2]> {
    let cam_pos = camera.get_position();
    let chunk_x = (cam_pos.x as i32).div_euclid(CHUNK_WIDTH_I32);
    let chunk_z = (cam_pos.z as i32).div_euclid(CHUNK_DEPTH_I32);

    let r_squared = RENDER_DISTANCE_CHUNKS * RENDER_DISTANCE_CHUNKS;

    let mut nearest_chunk = None;
    let mut shortest_distance = i32::MAX;

    for i in -MAX_DISTANCE_X..=MAX_DISTANCE_X {
        for j in -MAX_DISTANCE_Y..=MAX_DISTANCE_Y {
            let distance = i * i + j * j;

            // 1. Quick distance check first (cheapest operation)
            if distance > r_squared || distance >= shortest_distance {
                continue;
            }

            let location = [i + chunk_x, j + chunk_z];

            // 2. HashMap lookup (medium cost)
            if generated_chunks.contains_key(&location) {
                continue;
            }

            // 3. Frustum intersection check (most expensive)
            if cuboid_intersects_frustum(&chunkcoord_to_aabb(location), camera) {
                shortest_distance = distance;
                nearest_chunk = Some(location);
            }
        }
    }

    nearest_chunk
}
