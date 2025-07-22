use std::{collections::HashMap, time::Duration};

use vek::{Mat4, Quaternion, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::{chunk, player::Player};

#[derive(Debug)]
pub struct CameraData {
    pub position: Vec3<f32>,
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

    pub fn update_from_player(&mut self, player: &Player) {
        self.position = player.get_camera_position();
        self.pitch = player.pitch;
        self.yaw = player.yaw;
        self.quaternion = to_quaternion(-self.yaw, self.pitch).normalized();
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

pub struct PlayerController {
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
}

impl PlayerController {
    pub const fn new(speed: f32, sensitivity: f32) -> Self {
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
        }
    }

    pub const fn process_keyboard(&mut self, key: PhysicalKey, pressed: bool) -> bool {
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
            PhysicalKey::Unidentified(_) => false,
        }
    }

    pub fn process_mouse(&mut self, dx: f64, dy: f64) {
        self.rotate_horizontal = dx as f32;
        self.rotate_vertical = -dy as f32;
    }

    pub fn process_scroll(&mut self, delta: f32) {
        self.scroll = delta * 200.0;
    }

    pub fn update_player(&mut self, player: &mut Player, dt: std::time::Duration) {
        let dt_secs = dt.as_secs_f32();

        player.apply_movement_input(
            self.amount_forward - self.amount_backward,
            self.amount_right - self.amount_left,
            self.amount_up - self.amount_down,
            self.speed,
            dt_secs,
        );

        player.apply_rotation_input(
            self.rotate_horizontal,
            self.rotate_vertical,
            self.sensitivity,
        );

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
        self.scroll = 0.0;
    }
}

pub struct Camera {
    pub data: CameraData,
    pub projection: Projection,
    pub controller: PlayerController,
    pub player: Player,
}

impl Camera {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    pub fn update(&mut self, dt: Duration, world: &HashMap<[i32; 2], chunk::ChunkData>) {
        self.controller.update_player(&mut self.player, dt);

        self.player.update_physics(dt.as_secs_f32(), world);

        self.data.update_from_player(&self.player);
    }

    pub const fn get_position(&self) -> Vec3<f32> {
        self.data.position
    }

    pub const fn get_looking_at(&self) -> Option<(Vec3<i32>, usize)> {
        self.player.looking_at_block
    }

    pub fn get_transformation(&self) -> Mat4<f32> {
        self.projection.calc_matrix() * self.data.calc_matrix()
    }
}
