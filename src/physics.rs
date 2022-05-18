use vek::Vec3;

// pub fn is_collision_with_block(
//     camera_pos: Vec3<f32>,
//     block_x: i32,
//     block_y: i32,
//     block_z: i32,
// ) -> bool {
//     camera_pos.x + 0.3 > block_x as f32
//         && camera_pos.x - 0.3 < block_x as f32 + 1.0
//         && camera_pos.z + 0.3 > block_z as f32
//         && camera_pos.z - 0.3 < block_z as f32 + 1.0
//         && camera_pos.y > block_y as f32
//         && camera_pos.y < block_y as f32 + 1.0 + 1.5
// }

pub fn _get_distance_to_block(
    camera_pos: Vec3<f32>,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> (f32, f32, f32) {
    let block_x = block_x as f32;
    let block_y = block_y as f32;
    let block_z = block_z as f32;
    let dx = if camera_pos.x - 0.3 < block_x {
        block_x - (camera_pos.x + 0.3)
    } else if camera_pos.x - 0.3 > block_x {
        camera_pos.x - 0.3 - (block_x + 1.0)
    } else {
        0.0
    };
    let dy = if camera_pos.y - 0.3 < block_y {
        block_y - (camera_pos.y + 0.3)
    } else if camera_pos.y - 0.3 > block_y {
        camera_pos.y - 0.3 - (block_y + 1.0)
    } else {
        0.0
    };
    let dz = if camera_pos.z - 0.3 < block_z {
        block_z - (camera_pos.z + 0.3)
    } else if camera_pos.z - 0.3 > block_z {
        camera_pos.z - 0.3 - (block_z + 1.0)
    } else {
        0.0
    };
    (dx, dy, dz)
}
