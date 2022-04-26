use vek::Vec3;

pub fn is_collision_with_block(
    camera_pos: Vec3<f32>,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> bool {
    camera_pos.x + 0.3 > block_x as f32
        && camera_pos.x - 0.3 < block_x as f32 + 1.0
        && camera_pos.z + 0.3 > block_z as f32
        && camera_pos.z - 0.3 < block_z as f32 + 1.0
        && camera_pos.y > block_y as f32
        && camera_pos.y < block_y as f32 + 1.0 + 1.5
}
