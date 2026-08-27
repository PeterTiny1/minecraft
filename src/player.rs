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

fn get_axis_overlap(a: Aabb<f32>, b: Aabb<f32>, axis: Axis) -> f32 {
    let (a_min, a_max, b_min, b_max) = match axis {
        Axis::X => (a.min.x, a.max.x, b.min.x, b.max.x),
        Axis::Y => (a.min.y, a.max.y, b.min.y, b.max.y),
        Axis::Z => (a.min.z, a.max.z, b.min.z, b.max.z),
    };

    (a_max.min(b_max) - a_min.max(b_min)).max(0.0)
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

        // Perpendicular right vector on XZ plane (guarantees exact 90-degree alignment)
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

        // --- 4. Sub-Stepped Collision Resolution ---
        let desired_displacement = self.velocity * dt;
        self.is_grounded = false;

        // Subdivide movement step into chunks smaller than half a block to stop tunneling
        let max_substep = 0.4;

        let dx_steps = (desired_displacement.x.abs() / max_substep).ceil().max(1.0) as usize;
        let step_x = desired_displacement.x / dx_steps as f32;
        for _ in 0..dx_steps {
            self.position.x += step_x;
            if self.resolve_collisions_on_axis(world, Axis::X) {
                break;
            }
        }

        let dy_steps = (desired_displacement.y.abs() / max_substep).ceil().max(1.0) as usize;
        let step_y = desired_displacement.y / dy_steps as f32;
        for _ in 0..dy_steps {
            self.position.y += step_y;
            if self.resolve_collisions_on_axis(world, Axis::Y) {
                break;
            }
        }

        let dz_steps = (desired_displacement.z.abs() / max_substep).ceil().max(1.0) as usize;
        let step_z = desired_displacement.z / dz_steps as f32;
        for _ in 0..dz_steps {
            self.position.z += step_z;
            if self.resolve_collisions_on_axis(world, Axis::Z) {
                break;
            }
        }

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

            // Only place if target position is currently empty/air
            if chunk_manager.get_block(place_pos) == Some(BlockType::Air) {
                chunk_manager.set_block(place_pos, BlockType::Stone);
            }
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn resolve_collisions_on_axis(&mut self, world: &WorldStorage, axis: Axis) -> bool {
        let skin = 0.001;
        let player_aabb = self.aabb();

        let min_x = player_aabb.min.x.floor() as i32;
        let max_x = player_aabb.max.x.floor() as i32;
        let min_y = player_aabb.min.y.floor() as i32;
        let max_y = player_aabb.max.y.floor() as i32;
        let min_z = player_aabb.min.z.floor() as i32;
        let max_z = player_aabb.max.z.floor() as i32;

        let mut max_penetration = 0.0f32;
        let mut target_block: Option<Aabb<f32>> = None;

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

                        if player_aabb.collides_with_aabb(block_aabb) {
                            let penetration = get_axis_overlap(player_aabb, block_aabb, axis);
                            if penetration > max_penetration {
                                max_penetration = penetration;
                                target_block = Some(block_aabb);
                            }
                        }
                    }
                }
            }
        }

        if let Some(block_aabb) = target_block {
            self.handle_collision(axis, block_aabb, skin);
            true
        } else {
            false
        }
    }

    fn handle_collision(&mut self, axis: Axis, block_aabb: Aabb<f32>, skin: f32) {
        match axis {
            Axis::X => {
                if self.velocity.x > 0.0 {
                    self.position.x = block_aabb.min.x - self.half_extents.x - skin;
                } else if self.velocity.x < 0.0 {
                    self.position.x = block_aabb.max.x + self.half_extents.x + skin;
                }
                self.velocity.x = 0.0;
            }
            Axis::Y => {
                if self.velocity.y <= 0.0 {
                    self.position.y = block_aabb.max.y + skin;
                    self.is_grounded = true;
                } else {
                    self.position.y = block_aabb.min.y - PLAYER_HEIGHT - skin;
                }
                self.velocity.y = 0.0;
            }
            Axis::Z => {
                if self.velocity.z > 0.0 {
                    self.position.z = block_aabb.min.z - self.half_extents.z - skin;
                } else if self.velocity.z < 0.0 {
                    self.position.z = block_aabb.max.z + self.half_extents.z + skin;
                }
                self.velocity.z = 0.0;
            }
        }
    }

    #[must_use]
    pub fn get_camera_position(&self) -> Vec3<f32> {
        self.position + Vec3::new(0.0, EYE_HEIGHT, 0.0)
    }
}
