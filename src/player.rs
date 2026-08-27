use std::{
    f32::consts::FRAC_PI_2,
    time::{Duration, Instant},
};
use vek::{Aabb, Vec2, Vec3};

use crate::{
    block::BlockType,
    camera::CameraData,
    direction::Direction,
    input::InputState,
    ray,
    world::{ChunkManager, WorldStorage},
};

const GRAVITY: f32 = 30.0;
const FRICTION: f32 = 30.0;
const MAX_FALL_SPEED: f32 = 54.0;
const PLAYER_HEIGHT: f32 = 1.8;
const PLAYER_WIDTH_HALF: f32 = 0.3;
const EYE_HEIGHT: f32 = 1.6;

#[derive(Debug)]
pub struct Player {
    pub position: Vec3<f32>,
    pub velocity: Vec3<f32>,
    pub is_grounded: bool,
    pub speed: f32,
    pub sensitivity: f32,

    half_extents: Vec3<f32>,
    looking_at_block: Option<(Vec3<i32>, Direction)>,
    last_break_time: Instant,
    last_place_time: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    X,
    Y,
    Z,
}

/// Calculates continuous collision detection between `moving` with `displacement`
/// against a stationary `obstacle`.
fn swept_aabb(
    moving: Aabb<f32>,
    displacement: Vec3<f32>,
    obstacle: Aabb<f32>,
) -> Option<(f32, Axis)> {
    // 1. Calculate entry and exit distances along each axis
    let (x_entry_dist, x_exit_dist) = if displacement.x > 0.0 {
        (obstacle.min.x - moving.max.x, obstacle.max.x - moving.min.x)
    } else {
        (obstacle.max.x - moving.min.x, obstacle.min.x - moving.max.x)
    };

    let (y_entry_dist, y_exit_dist) = if displacement.y > 0.0 {
        (obstacle.min.y - moving.max.y, obstacle.max.y - moving.min.y)
    } else {
        (obstacle.max.y - moving.min.y, obstacle.min.y - moving.max.y)
    };

    let (z_entry_dist, z_exit_dist) = if displacement.z > 0.0 {
        (obstacle.min.z - moving.max.z, obstacle.max.z - moving.min.z)
    } else {
        (obstacle.max.z - moving.min.z, obstacle.min.z - moving.max.z)
    };

    // 2. Calculate entry and exit times for each axis (scaled 0.0 to 1.0)
    let (x_entry, x_exit) = if displacement.x == 0.0 {
        // If already overlapping on X, entry is immediate (-inf) and exit is +inf.
        // If not overlapping at all on X, entry/exit are both invalid.
        if moving.max.x <= obstacle.min.x || moving.min.x >= obstacle.max.x {
            return None;
        }
        (-f32::INFINITY, f32::INFINITY)
    } else {
        (x_entry_dist / displacement.x, x_exit_dist / displacement.x)
    };

    let (y_entry, y_exit) = if displacement.y == 0.0 {
        if moving.max.y <= obstacle.min.y || moving.min.y >= obstacle.max.y {
            return None;
        }
        (-f32::INFINITY, f32::INFINITY)
    } else {
        (y_entry_dist / displacement.y, y_exit_dist / displacement.y)
    };

    let (z_entry, z_exit) = if displacement.z == 0.0 {
        if moving.max.z <= obstacle.min.z || moving.min.z >= obstacle.max.z {
            return None;
        }
        (-f32::INFINITY, f32::INFINITY)
    } else {
        (z_entry_dist / displacement.z, z_exit_dist / displacement.z)
    };

    // 3. Find overall entry and exit time
    let entry_time = x_entry.max(y_entry).max(z_entry);
    let exit_time = x_exit.min(y_exit).min(z_exit);

    // 4. Determine collision validity:
    // Notice `entry_time >= exit_time` (exclusive bounds) prevents snagging on zero-thickness edges.
    if entry_time >= exit_time
        || (x_entry < 0.0 && y_entry < 0.0 && z_entry < 0.0)
        || entry_time > 1.0
        || entry_time < 0.0
    {
        return None;
    }

    // 5. Identify normal hit axis
    let hit_axis = if x_entry >= y_entry && x_entry >= z_entry {
        Axis::X
    } else if y_entry >= x_entry && y_entry >= z_entry {
        Axis::Y
    } else {
        Axis::Z
    };

    Some((entry_time, hit_axis))
}
impl Player {
    #[must_use]
    pub fn new(position: Vec3<f32>, speed: f32, sensitivity: f32) -> Self {
        Self {
            position,
            velocity: Vec3::zero(),
            is_grounded: false,
            speed,
            sensitivity,
            half_extents: Vec3::new(PLAYER_WIDTH_HALF, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH_HALF),
            looking_at_block: None,
            last_break_time: Instant::now(),
            last_place_time: Instant::now(),
        }
    }

    /// Calculates the player's Axis-Aligned Bounding Box (AABB).
    /// `self.position` is considered the bottom-center of the player.
    #[must_use]
    pub fn aabb(&self) -> Aabb<f32> {
        let center = self.position + Vec3::new(0.0, self.half_extents.y, 0.0);
        Aabb {
            min: center - self.half_extents,
            max: center + self.half_extents,
        }
    }

    /// Handles mouse look, movement input, gravity, drag, collisions, and targeting raycast.
    pub fn update_physics(
        &mut self,
        dt: f32,
        world: &WorldStorage,
        input: &InputState,
        camera_data: &mut CameraData,
    ) {
        // --- 0. Update Camera Rotation ---
        camera_data.yaw += input.mouse_dx * self.sensitivity;
        camera_data.pitch = (camera_data.pitch - input.mouse_dy * self.sensitivity)
            .clamp(-FRAC_PI_2 + 0.001, FRAC_PI_2 - 0.001);

        // --- 1. Movement Direction (Aligned Directly to Camera View) ---
        let cam_forward = camera_data.get_forward_vector();
        let forward = Vec3::new(cam_forward.x, 0.0, cam_forward.z)
            .try_normalized()
            .unwrap_or(Vec3::unit_z());

        let right = Vec3::new(-forward.z, 0.0, forward.x);

        let mut move_input = Vec2::<f32>::zero();
        if input.forward {
            move_input.x += 1.0;
        }
        if input.backward {
            move_input.x -= 1.0;
        }
        if input.right {
            move_input.y += 1.0;
        }
        if input.left {
            move_input.y -= 1.0;
        }

        let input_dir = move_input.try_normalized().unwrap_or(Vec2::zero());
        let wish_dir = forward * input_dir.x + right * input_dir.y;

        // --- 2. Ground & Air Friction ---
        let friction_rate = if self.is_grounded {
            FRICTION
        } else {
            FRICTION * 0.2
        };

        let current_xz = Vec2::new(self.velocity.x, self.velocity.z);
        let current_speed = current_xz.magnitude();

        if wish_dir == Vec3::zero() && current_speed > 0.0 {
            let new_speed = (current_speed - friction_rate * dt).max(0.0);
            let scale = new_speed / current_speed;
            self.velocity.x *= scale;
            self.velocity.z *= scale;
        }

        // --- 3. Vector-Based Linear Acceleration ---
        const ACCELERATION: f32 = 80.0;
        if wish_dir != Vec3::zero() {
            let target_vel_2d = Vec2::new(wish_dir.x, wish_dir.z) * self.speed;
            let current_vel_2d = Vec2::new(self.velocity.x, self.velocity.z);

            let delta = target_vel_2d - current_vel_2d;
            let delta_dist = delta.magnitude();

            if delta_dist > 0.0 {
                let step = (ACCELERATION * dt).min(delta_dist);
                let new_vel = current_vel_2d + (delta / delta_dist) * step;

                self.velocity.x = new_vel.x;
                self.velocity.z = new_vel.y;
            }
        }

        // Vertical / Flight input
        if input.up {
            self.velocity.y += self.speed * dt * 5.0;
        } else if input.down {
            self.velocity.y -= self.speed * dt * 5.0;
        }

        // Gravity
        self.velocity.y = (self.velocity.y - GRAVITY * dt).max(-MAX_FALL_SPEED);

        // --- 4. Swept AABB Collision Resolution ---
        self.is_grounded = false;

        // Move per axis sequentially to enable wall sliding
        self.move_and_slide_axis(world, Axis::X, dt);
        self.move_and_slide_axis(world, Axis::Y, dt);
        self.move_and_slide_axis(world, Axis::Z, dt);

        // Void safety reset
        if self.position.y < -64.0 {
            self.position.y = 128.0;
            self.velocity = Vec3::zero();
        }

        // --- 5. Target Raycast ---
        let eye_level_position = self.get_camera_position();
        let looking_direction = camera_data.get_forward_vector();

        self.looking_at_block = ray::Ray::new(eye_level_position, looking_direction, 5.0)
            .find(|(e, _)| matches!(world.get_block(*e), Some(b) if b != BlockType::Air));
    }

    /// Moves the player along a single axis using Swept AABB Continuous Collision Detection.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn move_and_slide_axis(&mut self, world: &WorldStorage, axis: Axis, dt: f32) {
        let displacement = match axis {
            Axis::X => Vec3::new(self.velocity.x * dt, 0.0, 0.0),
            Axis::Y => Vec3::new(0.0, self.velocity.y * dt, 0.0),
            Axis::Z => Vec3::new(0.0, 0.0, self.velocity.z * dt),
        };

        if displacement == Vec3::zero() {
            return;
        }

        let player_aabb = self.aabb();
        let destination_aabb = Aabb {
            min: player_aabb.min + displacement,
            max: player_aabb.max + displacement,
        };

        // Construct a broadphase AABB spanning from start position to end position using map2
        let swept_box = Aabb {
            min: player_aabb.min.map2(destination_aabb.min, f32::min),
            max: player_aabb.max.map2(destination_aabb.max, f32::max),
        };

        let min_x = swept_box.min.x.floor() as i32;
        let max_x = swept_box.max.x.floor() as i32;
        let min_y = swept_box.min.y.floor() as i32;
        let max_y = swept_box.max.y.floor() as i32;
        let min_z = swept_box.min.z.floor() as i32;
        let max_z = swept_box.max.z.floor() as i32;

        let mut earliest_hit = 1.0f32;
        let mut hit_occurred = false;

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    let pos = Vec3::new(x, y, z);
                    if let Some(block) = world.get_block(pos)
                        && block.is_solid()
                    {
                        let block_aabb = Aabb {
                            min: Vec3::new(x as f32, y as f32, z as f32),
                            max: Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                        };

                        if let Some((hit_time, _)) =
                            swept_aabb(player_aabb, displacement, block_aabb)
                        {
                            if hit_time < earliest_hit {
                                earliest_hit = hit_time;
                                hit_occurred = true;
                            }
                        }
                    }
                }
            }
        }

        if hit_occurred {
            // Advance player position precisely up to the block boundary face (no skin needed)
            match axis {
                Axis::X => {
                    self.position.x += displacement.x * earliest_hit;
                    self.velocity.x = 0.0;
                }
                Axis::Y => {
                    self.position.y += displacement.y * earliest_hit;
                    if self.velocity.y <= 0.0 {
                        self.is_grounded = true;
                    }
                    self.velocity.y = 0.0;
                }
                Axis::Z => {
                    self.position.z += displacement.z * earliest_hit;
                    self.velocity.z = 0.0;
                }
            }
        } else {
            // Unobstructed path: full displacement applied
            match axis {
                Axis::X => self.position.x += displacement.x,
                Axis::Y => self.position.y += displacement.y,
                Axis::Z => self.position.z += displacement.z,
            }
        }
    }

    /// Evaluates block breaking and placing actions based on input state and cooldown timers.
    pub fn update_blocks(&mut self, input_state: &InputState, chunk_manager: &mut ChunkManager) {
        let Some((location, previous_step)) = self.looking_at_block else {
            return;
        };
        let now = Instant::now();

        // 250ms cooldown on breaking
        if input_state.left_click_just_pressed
            || input_state.left_click_held
                && now - self.last_break_time > Duration::from_millis(250)
        {
            self.last_break_time = now;
            chunk_manager.set_block(location, BlockType::Air);
        }

        // 250ms cooldown on placing
        if input_state.right_click_just_pressed
            || input_state.right_click_held
                && now - self.last_place_time > Duration::from_millis(250)
        {
            self.last_place_time = now;
            let place_pos = location + previous_step.opposite().offset();

            let target_aabb = Aabb {
                min: Vec3::new(place_pos.x as f32, place_pos.y as f32, place_pos.z as f32),
                max: Vec3::new(
                    (place_pos.x + 1) as f32,
                    (place_pos.y + 1) as f32,
                    (place_pos.z + 1) as f32,
                ),
            };

            // Prevent placing block inside player bounding box
            let intersects_player = self.aabb().collides_with_aabb(target_aabb);

            if chunk_manager.get_block(place_pos) == Some(BlockType::Air) && !intersects_player {
                chunk_manager.set_block(place_pos, BlockType::Stone);
            }
        }
    }

    #[must_use]
    pub fn get_camera_position(&self) -> Vec3<f32> {
        self.position + Vec3::new(0.0, EYE_HEIGHT, 0.0)
    }
}
