use std::{collections::HashSet, sync::OnceLock};

use vek::{Aabb, Vec3};

use crate::{
    RENDER_DISTANCE, camera,
    world::{
        CHUNK_HEIGHT_I32,
        types::{
            CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_WIDTH, CHUNK_WIDTH_I32,
            ChunkDataStorage,
        },
    },
};

const MAX_DISTANCE_X: i32 = RENDER_DISTANCE as i32 / CHUNK_WIDTH_I32 + 1;
const MAX_DISTANCE_Y: i32 = RENDER_DISTANCE as i32 / CHUNK_DEPTH_I32 + 1;
const RENDER_DISTANCE_CHUNKS: i32 = if MAX_DISTANCE_X > MAX_DISTANCE_Y {
    MAX_DISTANCE_X
} else {
    MAX_DISTANCE_Y
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

/// Returns chunk relative offsets sorted by distance squared from [0, 0].
/// Initialized once on first request.
fn chunk_search_offsets() -> &'static [[i32; 2]] {
    static OFFSETS: OnceLock<Vec<[i32; 2]>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        let r = RENDER_DISTANCE_CHUNKS;
        let r_sq = r * r;
        let mut offsets = Vec::new();

        for x in -r..=r {
            for z in -r..=r {
                let dist_sq = x * x + z * z;
                if dist_sq <= r_sq {
                    offsets.push([x, z]);
                }
            }
        }

        // Sort ascending by distance squared so nearest offsets come first
        offsets.sort_unstable_by_key(|&[x, z]| x * x + z * z);
        offsets
    })
}

/// Returns up to `max_to_fetch` missing chunk positions around the camera,
/// ordered from nearest to furthest.
#[must_use]
pub fn nearest_unloaded_chunks(
    generated_chunks: &ChunkDataStorage,
    pending_chunks: &HashSet<[i32; 2]>,
    camera: &camera::Camera,
    max_to_fetch: usize,
) -> Vec<[i32; 2]> {
    if max_to_fetch == 0 {
        return Vec::new();
    }

    let cam_pos = camera.get_position();
    let center_x = (cam_pos.x as i32).div_euclid(CHUNK_WIDTH_I32);
    let center_z = (cam_pos.z as i32).div_euclid(CHUNK_DEPTH_I32);

    let mut results = Vec::with_capacity(max_to_fetch);

    for &[dx, dz] in chunk_search_offsets() {
        let location = [center_x + dx, center_z + dz];

        // Skip chunks that are already loaded or currently generating
        if !generated_chunks.contains_key(&location) && !pending_chunks.contains(&location) {
            results.push(location);
            if results.len() >= max_to_fetch {
                break;
            }
        }
    }

    results
}

/// Returns the target chunk location, array index, and relative chunk offset vectors
/// for surrounding neighbors that are affected by AO at this position.
#[must_use]
pub fn resolve_block_target(pos: Vec3<i32>) -> Option<([i32; 2], usize, [Option<[i32; 2]>; 3])> {
    if pos.y < 0 || pos.y >= CHUNK_HEIGHT_I32 {
        return None;
    }

    let (chunk_loc, [lx, ly, lz]) = world_to_chunk_pos(pos.x, pos.y, pos.z);
    let idx = block_index(lx, ly, lz);

    let max_x = (CHUNK_WIDTH_I32 - 1) as usize;
    let max_z = (CHUNK_DEPTH_I32 - 1) as usize;

    let dx = if lx == 0 {
        -1
    } else if lx == max_x {
        1
    } else {
        0
    };
    let dz = if lz == 0 {
        -1
    } else if lz == max_z {
        1
    } else {
        0
    };

    let neighbors = match (dx, dz) {
        (0, 0) => [None, None, None],
        (x, 0) => [Some([x, 0]), None, None],
        (0, z) => [Some([0, z]), None, None],
        (x, z) => [Some([x, 0]), Some([0, z]), Some([x, z])],
    };

    Some((chunk_loc, idx, neighbors))
}
