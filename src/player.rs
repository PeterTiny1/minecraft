use std::f32::consts::FRAC_PI_2;
use vek::{Aabb, Vec2, Vec3};

use crate::{
    block::BlockType,
    camera::{self, CameraData},
    chunk::{BlockProvider, ChunkDataStorage},
    ray,
};

const GRAVITY: f32 = 30.0;
const FRICTION: f32 = 10.0;
const MAX_FALL_SPEED: f32 = 54.0;
const PLAYER_HEIGHT: f32 = 1.8;
const PLAYER_WIDTH_HALF: f32 = 0.3;
const EYE_HEIGHT: f32 = 1.6;

#[derive(Debug)]
pub struct Player {
    pub position: Vec3<f32>,
    velocity: Vec3<f32>,
    pub is_grounded: bool,
    half_extents: Vec3<f32>,
    pub looking_at_block: Option<(Vec3<i32>, usize)>,
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

impl Player {
    #[must_use] 
    pub fn new(position: Vec3<f32>) -> Self {
        Self {
            position,
            velocity: Vec3::zero(),
            is_grounded: false,
            half_extents: Vec3::new(PLAYER_WIDTH_HALF, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH_HALF),
            looking_at_block: None,
        }
    }

    /// Calculates the player's Axis-Aligned Bounding Box (AABB).
    /// `self.position` is considered the bottom-center of the player.
    #[must_use] 
    pub fn aabb(&self) -> Aabb<f32> {
        // The center of the AABB is halfway up the player's height from their position.
        let center = self.position + Vec3::new(0.0, self.half_extents.y, 0.0);
        Aabb {
            min: center - self.half_extents,
            max: center + self.half_extents,
        }
    }

    /// This is the new, merged update function
    pub fn update_physics(
        &mut self,
        dt: f32,
        world: &ChunkDataStorage,
        controller: &camera::PlayerController,
        camera_data: &CameraData,
    ) {
        // --- 1. Apply Movement Input ---
        let forward = (camera_data.yaw.cos() * camera_data.pitch.cos()) * Vec3::unit_x()
            + (camera_data.yaw.sin() * camera_data.pitch.cos()) * Vec3::unit_z();
        let right = (camera_data.yaw - FRAC_PI_2).cos() * Vec3::unit_x()
            + (camera_data.yaw - FRAC_PI_2).sin() * Vec3::unit_z();

        let forward_force = controller.amount_forward - controller.amount_backward;
        let right_force = controller.amount_left - controller.amount_right;
        let up_force = controller.amount_up - controller.amount_down;

        let direction = (forward * forward_force + right * right_force).normalized();

        if direction.x.is_finite() && direction.y.is_finite() && direction.z.is_finite() {
            self.velocity.x += direction.x * controller.get_speed() * dt;
            self.velocity.z += direction.z * controller.get_speed() * dt;
        }

        self.velocity.y += up_force * controller.get_speed() * dt * 5.0; // Multiplier 5.0 for faster vertical movement

        // --- 2. Apply Gravity & Drag  ---
        self.velocity.y -= GRAVITY * dt;
        self.velocity.y = self.velocity.y.max(-MAX_FALL_SPEED);

        // Air friction
        let air_friction_decay = 0.95_f32.powf(dt);
        self.velocity.x *= air_friction_decay;
        self.velocity.z *= air_friction_decay;

        // Ground friction
        if self.is_grounded {
            let velocity_xz = Vec2::new(self.velocity.x, self.velocity.z);
            if velocity_xz.magnitude_squared() > (FRICTION * dt).powi(2) {
                let friction = velocity_xz.normalized() * FRICTION * dt;
                self.velocity.x -= friction.x;
                self.velocity.z -= friction.y;
            } else {
                self.velocity.x = 0.0;
                self.velocity.z = 0.0;
            }
        }

        // --- 3. Collision Detection ---
        let desired_displacement = self.velocity * dt;
        self.is_grounded = false; // Reset grounded state each frame

        self.position.x += desired_displacement.x;
        self.resolve_collisions_on_axis(world, Axis::X);

        self.position.y += desired_displacement.y;
        self.resolve_collisions_on_axis(world, Axis::Y);

        self.position.z += desired_displacement.z;
        self.resolve_collisions_on_axis(world, Axis::Z);

        // Handle falling off world
        if self.position.y < -64.0 {
            self.position.y = 128.0;
            self.velocity = Vec3::zero();
        }

        // --- 4. Update Raycast ---
        // Get camera position and direction
        let eye_level_position = self.get_camera_position();
        let looking_direction = camera_data.get_forward_vector();

        self.looking_at_block = ray::Ray::new(eye_level_position, looking_direction, 5.0).find(
            |(e, _)| matches!(world.get_block(e.x, e.y, e.z), Some(b) if b != BlockType::Air),
        );
    }

    fn resolve_collisions_on_axis(&mut self, world: &ChunkDataStorage, axis: Axis) {
        let skin = 0.001;

        for _ in 0..5 {
            let player_aabb_current = self.aabb();

            let min_x = (player_aabb_current.min.x - skin).floor() as i32;
            let max_x = (player_aabb_current.max.x + skin).floor() as i32;
            let min_y = (player_aabb_current.min.y - skin).floor() as i32;
            let max_y = (player_aabb_current.max.y + skin).floor() as i32;
            let min_z = (player_aabb_current.min.z - skin).floor() as i32;
            let max_z = (player_aabb_current.max.z + skin).floor() as i32;

            let mut collision_found = false;

            for x in min_x..=max_x {
                for y in min_y..=max_y {
                    for z in min_z..=max_z {
                        if let Some(block_type) = world.get_block(x, y, z) {
                            if block_type.is_solid() {
                                let block_aabb = Aabb {
                                    min: Vec3::new(x as f32, y as f32, z as f32),
                                    max: Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                                };

                                if player_aabb_current.collides_with_aabb(block_aabb) {
                                    collision_found = true;
                                    self.handle_collision(axis, block_aabb, skin);
                                    break;
                                }
                            }
                        }
                    }
                    if collision_found {
                        break;
                    }
                }
                if collision_found {
                    break;
                }
            }

            if !collision_found {
                break;
            }
        }
    }

    fn handle_collision(&mut self, axis: Axis, block_aabb: Aabb<f32>, skin: f32) {
        match axis {
            Axis::X => {
                if self.aabb().center().x < block_aabb.center().x {
                    self.position.x = block_aabb.min.x - self.half_extents.x - skin;
                } else {
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
                if self.aabb().center().z < block_aabb.center().z {
                    self.position.z = block_aabb.min.z - self.half_extents.z - skin;
                } else {
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

    #[must_use] 
    pub const fn get_looking_at(&self) -> Option<(Vec3<i32>, usize)> {
        self.looking_at_block
    }
}
