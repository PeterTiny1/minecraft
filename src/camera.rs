use std::{f32::consts::FRAC_PI_2, time::Duration};

use sdl2::keyboard::Keycode;
use vek::{Mat4, Quaternion, Vec3};

#[derive(Debug)]
pub struct Camera {
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
        s1 * s2 * c3 + c1 * c2 * s3,
        s1 * c2 * c3 + c1 * s2 * s3,
        c1 * s2 * c3 - s1 * c2 * s3,
        c1 * c2 * c3 - s1 * s2 * s3,
    )
}

impl Camera {
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
        self.aspect = width as f32 / height as f32
    }

    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::perspective_rh_no(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

pub struct CameraController {
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

impl CameraController {
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
        }
    }

    pub fn process_keyboard(&mut self, key: Keycode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        match key {
            Keycode::W | Keycode::Up => {
                self.amount_forward = amount;
                true
            }
            Keycode::S | Keycode::Down => {
                self.amount_backward = amount;
                true
            }
            Keycode::A | Keycode::Left => {
                self.amount_left = amount;
                true
            }
            Keycode::D | Keycode::Right => {
                self.amount_right = amount;
                true
            }
            Keycode::Space => {
                self.amount_up = amount;
                true
            }
            Keycode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = -mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: i32) {
        self.scroll = (delta * 200) as f32;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();
        let (yaw_sin, yaw_cos) = camera.yaw.sin_cos();
        let forward = Vec3::new(yaw_cos, 0.0, yaw_sin).normalized();
        let right = Vec3::new(-yaw_sin, 0.0, yaw_cos).normalized();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        let (pitch_sin, pitch_cos) = camera.pitch.sin_cos();
        let scrollward =
            Vec3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalized();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        camera.yaw += (self.rotate_horizontal) * self.sensitivity / 10.0;
        camera.pitch += (self.rotate_vertical) * self.sensitivity / 10.0;

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        if camera.pitch < -FRAC_PI_2 {
            camera.pitch = -FRAC_PI_2;
        } else if camera.pitch > (FRAC_PI_2) {
            camera.pitch = FRAC_PI_2;
        }
        // TODO: implement placing/breaking blocks

        camera.quaternion = to_quaternion(-camera.yaw, camera.pitch).normalized();
    }
}
