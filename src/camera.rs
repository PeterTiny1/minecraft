use std::{f32::consts::FRAC_PI_2, time::Duration};

use vek::{Mat4, Quaternion, Vec3};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, MouseScrollDelta, VirtualKeyCode},
};

#[derive(Debug)]
pub struct Camera {
    pub position: Vec3<f32>,
    quaternion: Quaternion<f32>,
}

impl Camera {
    pub fn new<V: Into<Vec3<f32>>>(position: V, yaw: f32, pitch: f32) -> Self {
        Self {
            position: position.into(),
            quaternion: {
                let mut quaternion = Quaternion::identity().rotated_y(-yaw);
                quaternion.rotate_3d(pitch, quaternion * Vec3::unit_z());
                quaternion
            },
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

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
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

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = match delta {
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();
        let scrollward = camera.quaternion * Vec3::unit_x();
        let forward = Vec3::new(scrollward.x, 0.0, scrollward.z);
        let right = Quaternion::rotation_y(-FRAC_PI_2) * forward;
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        let right = camera.quaternion * Vec3::unit_z();
        camera
            .quaternion
            .rotate_y((-self.rotate_horizontal) * self.sensitivity / 100.0);
        camera
            .quaternion
            .rotate_3d((self.rotate_vertical) * self.sensitivity / 100.0, right);

        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
    }

    pub fn release_all(&mut self) {
        self.amount_backward = 0.0;
        self.amount_forward = 0.0;
        self.amount_left = 0.0;
        self.amount_right = 0.0;
        self.amount_up = 0.0;
        self.amount_down = 0.0;
    }
}
