use vek::{Mat4, Vec3};

pub struct Camera {
    pub data: CameraData,
    pub projection: Projection,
}

impl Camera {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.projection.resize(width, height);
    }

    #[must_use]
    pub const fn get_position(&self) -> Vec3<f32> {
        self.data.position
    }

    #[must_use]
    pub fn get_transformation(&self) -> Mat4<f32> {
        self.projection.calc_matrix() * self.data.calc_matrix()
    }

    // This is the function we need to add for the block raycast
    #[must_use]
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
    #[must_use]
    pub fn new(position: (f32, f32, f32), yaw: f32, pitch: f32) -> Self {
        Self {
            position: position.into(),
            yaw,
            pitch,
        }
    }
    #[must_use]
    pub fn get_forward_vector(&self) -> Vec3<f32> {
        (self.yaw.cos() * self.pitch.cos()) * Vec3::unit_x()
            + self.pitch.sin() * Vec3::unit_y()
            + (self.yaw.sin() * self.pitch.cos()) * Vec3::unit_z()
    }
    #[must_use]
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
    #[must_use]
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

    #[must_use]
    pub fn calc_matrix(&self) -> Mat4<f32> {
        Mat4::perspective_rh_zo(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

