use vek::{Mat4, Vec3};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct Camera {
    pub data: CameraData,
    pub projection: Projection,
}

impl Camera {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    pub const fn get_position(&self) -> Vec3<f32> {
        self.data.position
    }

    pub fn get_transformation(&self) -> Mat4<f32> {
        self.projection.calc_matrix() * self.data.calc_matrix()
    }

    // This is the function we need to add for the block raycast
    pub fn get_forward_vector(&self) -> Vec3<f32> {
        (self.data.yaw.cos() * self.data.pitch.cos()) * Vec3::unit_x()
            + self.data.pitch.sin() * Vec3::unit_y()
            + (self.data.yaw.sin() * self.data.pitch.cos()) * Vec3::unit_z()
    }
}

#[derive(Debug)]
pub struct CameraData {
    pub position: Vec3<f32>,
    pub yaw: f32,
    pub pitch: f32,
}

impl CameraData {
    pub fn new(position: (f32, f32, f32), yaw: f32, pitch: f32) -> Self {
        Self {
            position: position.into(),
            yaw,
            pitch,
        }
    }
    pub fn get_forward_vector(&self) -> Vec3<f32> {
        (self.yaw.cos() * self.pitch.cos()) * Vec3::unit_x()
            + self.pitch.sin() * Vec3::unit_y()
            + (self.yaw.sin() * self.pitch.cos()) * Vec3::unit_z()
    }
    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::look_at_rh(
            self.position,
            self.position + self.get_forward_vector(),
            Vec3::unit_y(),
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
    pub fn new(width: u32, height: u32, fovy: f32, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy,
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::perspective_rh_zo(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug)]
pub struct PlayerController {
    pub amount_left: f32,
    pub amount_right: f32,
    pub amount_forward: f32,
    pub amount_backward: f32,
    pub amount_up: f32,
    pub amount_down: f32,

    // These are now private, only this controller manages them
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

    pub fn update_camera(&mut self, camera: &mut CameraData, dt: std::time::Duration) {
        let dt_secs = dt.as_secs_f32();

        // Apply rotation
        camera.yaw += self.rotate_horizontal * self.sensitivity * dt_secs;
        camera.pitch += self.rotate_vertical * self.sensitivity * dt_secs;

        // Clamp pitch
        camera.pitch = camera.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.001,
            std::f32::consts::FRAC_PI_2 - 0.001,
        );

        // Reset mouse deltas after applying
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
        self.scroll = 0.0;
    }
    pub const fn get_speed(&self) -> f32 {
        self.speed
    }
}
