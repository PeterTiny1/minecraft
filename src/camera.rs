use std::{collections::HashMap, f32::consts::FRAC_PI_2, time::Duration};

use vek::{Mat4, Quaternion, Vec2, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::{
    chunk::{self, get_block, BlockType, ChunkData},
    ray::Ray,
};

const GRAVITY: f32 = 9.807;

#[derive(Debug)]
pub struct CameraData {
    position: Vec3<f32>,
    pitch: f32,
    yaw: f32,
    quaternion: Quaternion<f32>,
}

fn to_quaternion(heading: f32, attitude: f32) -> Quaternion<f32> {
    let (s1, c1) = (heading / 2.0).sin_cos();
    let (s2, c2) = (attitude / 2.0).sin_cos();
    let (s3, c3) = (0.0, 1.0);
    Quaternion::from_xyzw(
        (s1 * s2).mul_add(c3, c1 * c2 * s3),
        (s1 * c2).mul_add(c3, c1 * s2 * s3),
        (c1 * s2).mul_add(c3, -s1 * c2 * s3),
        (c1 * c2).mul_add(c3, -s1 * s2 * s3),
    )
}

impl CameraData {
    pub fn new<V: Into<Vec3<f32>>>(position: V, yaw: f32, pitch: f32) -> Self {
        Self {
            position: position.into(),
            pitch,
            yaw,
            quaternion: to_quaternion(-yaw, pitch).normalized(),
        }
    }

    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::look_at_rh(
            self.position,
            self.quaternion * Vec3::unit_x() + self.position,
            self.quaternion * Vec3::unit_y(),
        )
    }
}

pub struct Projection {
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<f32>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::perspective_rh_no(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

pub struct Controller {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
    looking_at_block: Option<(Vec3<i32>, usize)>,
    velocity: Vec3<f32>,
}

const CORNER0: Vec3<f32> = Vec3 {
    x: 0.3,
    y: 1.5,
    z: 0.3,
};
const CORNER1: Vec3<f32> = Vec3 {
    x: 0.3,
    y: 1.5,
    z: -0.3,
};
const CORNER2: Vec3<f32> = Vec3 {
    x: -0.3,
    y: 1.5,
    z: -0.3,
};
const CORNER3: Vec3<f32> = Vec3 {
    x: -0.3,
    y: 1.5,
    z: 0.3,
};

fn next_float(x: f32) -> f32 {
    let bits = x.to_bits();
    let next_bits = if x.is_sign_positive() {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next_bits)
}

fn handle_collision(
    velocity: &mut Vec3<f32>,
    position: &mut Vec3<f32>,
    corner_offset: Vec3<f32>,
    dt: f32,
    world: &HashMap<[i32; 2], ChunkData>,
) {
    while let Some(collision) = find_collision(*velocity, *position - corner_offset, dt, world) {
        match collision.1 {
            0 | 1 => velocity.x = 0.0,
            2 => {
                velocity.y = 0.0;
                position.y = next_float((position.y - 1.5).floor() + 1.5);
            }
            3 => {
                velocity.y = 0.0;
            }
            4 | 5 => {
                velocity.z = 0.0;
            }
            _ => {}
        }
    }
}

const FRICTION: f32 = 1.0;

impl Controller {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
            looking_at_block: None,
            velocity: Vec3::default(),
        }
    }

    pub fn process_keyboard(&mut self, key: PhysicalKey, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        match key {
            PhysicalKey::Code(keycode) => match keycode {
                KeyCode::KeyW | KeyCode::ArrowUp => {
                    self.amount_forward = amount;
                    true
                }
                KeyCode::KeyS | KeyCode::ArrowDown => {
                    self.amount_backward = amount;
                    true
                }
                KeyCode::KeyA | KeyCode::ArrowLeft => {
                    self.amount_left = amount;
                    true
                }
                KeyCode::KeyD | KeyCode::ArrowRight => {
                    self.amount_right = amount;
                    true
                }
                KeyCode::Space => {
                    self.amount_up = amount;
                    true
                }
                KeyCode::ShiftLeft => {
                    self.amount_down = amount;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        self.rotate_horizontal = dx as f32;
        self.rotate_vertical = -dy as f32;
    }

    pub fn process_scroll(&mut self, delta: f32) {
        self.scroll = delta * 200.0;
    }

    fn update_camera(
        &mut self,
        camera: &mut CameraData,
        dt: Duration,
        world: &HashMap<[i32; 2], ChunkData>,
    ) {
        let dt = dt.as_secs_f32();
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalized();
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalized();
        self.velocity += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        self.velocity += right * (self.amount_right - self.amount_left) * self.speed * dt;
        self.velocity.y -= GRAVITY * dt;
        self.velocity.y *= 0.99_f32.powf(-dt);
        let velocity_xz = Vec2::new(self.velocity.x, self.velocity.z);
        if velocity_xz.magnitude() > FRICTION * dt {
            let friction = velocity_xz.normalized() * FRICTION * dt;
            self.velocity.x -= friction.x;
            self.velocity.z -= friction.y;
        } else {
            self.velocity.x = 0.0;
            self.velocity.z = 0.0;
        }
        handle_collision(&mut self.velocity, &mut camera.position, CORNER0, dt, world);
        handle_collision(&mut self.velocity, &mut camera.position, CORNER1, dt, world);
        handle_collision(&mut self.velocity, &mut camera.position, CORNER2, dt, world);
        handle_collision(&mut self.velocity, &mut camera.position, CORNER3, dt, world);
        camera.position += self.velocity * dt;
        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let looking_direction =
            Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalized();
        let looking_at_block = Ray::new(camera.position, looking_direction, 5.0).find(
            |(e, _)| matches!(get_block(world, e.x, e.y, e.z), Some(b) if b != BlockType::Air),
        );
        self.looking_at_block = looking_at_block;
        if camera.position.y < -64.0 {
            camera.position.y = 64.0;
        }
        self.scroll = 0.0;

        // camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;
        self.velocity.y += (self.amount_up - self.amount_down) * self.speed * dt * 5.0;

        camera.yaw += (self.rotate_horizontal) * self.sensitivity / 10.0;
        camera.pitch += (self.rotate_vertical) * self.sensitivity / 10.0;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        camera.pitch = camera.pitch.clamp(-FRAC_PI_2, FRAC_PI_2);

        camera.quaternion = to_quaternion(-camera.yaw, camera.pitch).normalized();
    }
}

fn find_collision(
    velocity: Vec3<f32>,
    origin: Vec3<f32>,
    dt: f32,
    world: &HashMap<[i32; 2], ChunkData>,
) -> Option<(Vec3<i32>, usize)> {
    Ray::new(origin, velocity.normalized(), (velocity * dt).magnitude())
        .find(|(b, _)| get_block(world, b.x, b.y, b.z).map_or(false, BlockType::is_solid))
}

pub struct Camera {
    pub data: CameraData,
    pub projection: Projection,
    pub controller: Controller,
}

impl Camera {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    pub fn update(&mut self, dt: Duration, world: &HashMap<[i32; 2], chunk::ChunkData>) {
        self.controller.update_camera(&mut self.data, dt, world);
    }

    pub const fn get_position(&self) -> Vec3<f32> {
        self.data.position
    }

    pub const fn get_looking_at(&self) -> Option<(Vec3<i32>, usize)> {
        self.controller.looking_at_block
    }

    pub fn calc_matrix(&self) -> Mat4<f32> {
        self.data.calc_matrix()
    }

    pub fn calc_projection_matrix(&self) -> Mat4<f32> {
        self.projection.calc_matrix()
    }
}
