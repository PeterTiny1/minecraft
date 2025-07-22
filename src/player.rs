use std::collections::HashMap;
use std::f32::consts::FRAC_PI_2; // Moved to top of file for better style

use vek::{Aabb, Vec2, Vec3};

use crate::{
    chunk::{get_block, BlockType, ChunkData},
    ray::Ray,
};

const GRAVITY: f32 = 30.0;
const FRICTION: f32 = 10.0;
const MAX_FALL_SPEED: f32 = 54.0;
const PLAYER_HEIGHT: f32 = 1.8; // Total height of player's collision box
const PLAYER_WIDTH_HALF: f32 = 0.3; // Half width/depth of player's collision box

pub struct Player {
    pub position: Vec3<f32>, // Bottom-center of the player AABB
    pub velocity: Vec3<f32>,
    pub pitch: f32,
    pub yaw: f32,
    pub is_grounded: bool,
    pub half_extents: Vec3<f32>, // Half width, half height, half depth of the AABB
    pub looking_at_block: Option<(Vec3<i32>, usize)>,
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

impl Player {
    pub fn new(position: Vec3<f32>) -> Self {
        Self {
            position,
            velocity: Vec3::zero(),
            pitch: 0.0,
            yaw: 0.0,
            is_grounded: false,
            // half_extents for an AABB: (width/2, height/2, depth/2)
            half_extents: Vec3::new(PLAYER_WIDTH_HALF, PLAYER_HEIGHT / 2.0, PLAYER_WIDTH_HALF),
            looking_at_block: None,
        }
    }

    /// Calculates the player's Axis-Aligned Bounding Box (AABB).
    /// `self.position` is considered the bottom-center of the player.
    pub fn aabb(&self) -> Aabb<f32> {
        // The center of the AABB is halfway up the player's height from their position.
        let center = self.position + Vec3::new(0.0, self.half_extents.y, 0.0);
        Aabb {
            min: center - self.half_extents,
            max: center + self.half_extents,
        }
    }

    pub fn apply_movement_input(
        &mut self,
        forward_amount: f32,
        right_amount: f32,
        up_down_amount: f32, // For creative mode up/down or jump impulse
        speed: f32,
        dt: f32,
    ) {
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();
        // FIX: Removed redundant .normalized() calls
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin);
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos);

        // Apply horizontal input to velocity
        self.velocity += forward * forward_amount * speed * dt;
        self.velocity += right * right_amount * speed * dt;

        // Apply vertical "creative" input
        self.velocity.y += up_down_amount * speed * dt * 5.0; // Multiplier 5.0 for faster vertical movement
    }

    pub fn apply_rotation_input(
        &mut self,
        rotate_horizontal: f32,
        rotate_vertical: f32,
        sensitivity: f32,
    ) {
        self.yaw += rotate_horizontal * sensitivity / 10.0;
        self.pitch += rotate_vertical * sensitivity / 10.0;
        self.pitch = self.pitch.clamp(-FRAC_PI_2, FRAC_PI_2);
    }

    pub fn update_physics(&mut self, dt: f32, world: &HashMap<[i32; 2], ChunkData>) {
        // 1. Apply Gravity
        self.velocity.y -= GRAVITY * dt;
        self.velocity.y = self.velocity.y.max(-MAX_FALL_SPEED); // Apply terminal velocity

        // 2. Apply Friction
        // FIX: Air friction (drag) should only apply to horizontal velocity.
        let air_friction_decay = 0.95_f32.powf(dt);
        self.velocity.x *= air_friction_decay;
        self.velocity.z *= air_friction_decay;

        // Ground friction (only if grounded)
        if self.is_grounded {
            let velocity_xz = Vec2::new(self.velocity.x, self.velocity.z);
            if velocity_xz.magnitude_squared() > (FRICTION * dt).powi(2) {
                let friction = velocity_xz.normalized() * FRICTION * dt;
                self.velocity.x -= friction.x;
                self.velocity.z -= friction.y;
            } else {
                // Stop horizontal movement if below friction threshold
                self.velocity.x = 0.0;
                self.velocity.z = 0.0;
            }
        }

        // --- 3. Collision Detection and Resolution ---
        let desired_displacement = self.velocity * dt;
        self.is_grounded = false; // Reset grounded state each frame

        // Resolve collisions on each axis independently
        self.position.x += desired_displacement.x;
        self.resolve_collisions_on_axis(world, Axis::X);

        self.position.y += desired_displacement.y;
        self.resolve_collisions_on_axis(world, Axis::Y);

        self.position.z += desired_displacement.z;
        self.resolve_collisions_on_axis(world, Axis::Z);

        // Handle falling off world (teleport back up)
        if self.position.y < -64.0 {
            self.position.y = 128.0;
            self.velocity = Vec3::zero();
        }

        // 4. Raycasting for looking_at_block
        let (pitch_sin, pitch_cos) = self.pitch.sin_cos();
        let (yaw_sin, yaw_cos) = self.yaw.sin_cos();

        let looking_direction =
            Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalized();
        let eye_level_position = self.get_camera_position();

        self.looking_at_block = Ray::new(eye_level_position, looking_direction, 5.0).find(
            |(e, _)| matches!(get_block(world, e.x, e.y, e.z), Some(b) if b != BlockType::Air),
        );
    }

    fn resolve_collisions_on_axis(&mut self, world: &HashMap<[i32; 2], ChunkData>, axis: Axis) {
        let skin = 0.001;

        for _ in 0..5 {
            // Use a for loop for a fixed number of iterations
            let player_aabb_current = self.aabb();

            // FIX: Use floor() and an inclusive range (..=) to correctly check blocks at boundaries.
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
                        if let Some(block_type) = get_block(world, x, y, z) {
                            if block_type.is_solid() {
                                let block_aabb = Aabb {
                                    min: Vec3::new(x as f32, y as f32, z as f32),
                                    max: Vec3::new((x + 1) as f32, (y + 1) as f32, (z + 1) as f32),
                                };

                                if player_aabb_current.collides_with_aabb(block_aabb) {
                                    collision_found = true;
                                    self.handle_collision(axis, block_aabb, skin);
                                    // After one collision, re-evaluate AABB and check all blocks again
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
                break; // No collisions found in this pass, resolution for this axis is done.
            }
        }
    }

    /// Helper function to contain the collision response logic.
    fn handle_collision(&mut self, axis: Axis, block_aabb: Aabb<f32>, skin: f32) {
        match axis {
            Axis::X => {
                if self.aabb().center().x < block_aabb.center().x {
                    // Player is on the -X side of the block, push them to -X
                    self.position.x = block_aabb.min.x - self.half_extents.x - skin;
                } else {
                    // Player is on the +X side of the block, push them to +X
                    self.position.x = block_aabb.max.x + self.half_extents.x + skin;
                }
                self.velocity.x = 0.0;
            }
            Axis::Y => {
                if self.velocity.y <= 0.0 {
                    // Moving down or still
                    // Resolve collision by placing player on top of the block.
                    self.position.y = block_aabb.max.y + skin;
                    self.is_grounded = true;
                } else {
                    // Moving up
                    // Resolve collision by placing player below the block (hitting a ceiling).
                    // position.y is the bottom of the AABB.
                    self.position.y = block_aabb.min.y - PLAYER_HEIGHT - skin;
                }
                self.velocity.y = 0.0;
            }
            Axis::Z => {
                // FIX: Removed copy-paste error. This logic is now correct.
                if self.aabb().center().z < block_aabb.center().z {
                    // Player is on the -Z side of the block, push them to -Z
                    self.position.z = block_aabb.min.z - self.half_extents.z - skin;
                } else {
                    // Player is on the +Z side of the block, push them to +Z
                    self.position.z = block_aabb.max.z + self.half_extents.z + skin;
                }
                self.velocity.z = 0.0;
            }
        }
    }

    pub fn get_camera_position(&self) -> Vec3<f32> {
        self.position + Vec3::new(0.0, PLAYER_HEIGHT * 0.9, 0.0)
    }
}
