use vek::Vec3;

/// 3D Voxel Raycaster (Amanatides-Woo Fast Voxel Traversal)
pub struct Ray {
    step: Vec3<i32>,
    t_max: Vec3<f32>,
    t_delta: Vec3<f32>,
    block_position: Vec3<i32>,
    max_len: f32,
    hit_face: usize,
    first_step: bool,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3<f32>, direction: Vec3<f32>, max_len: f32) -> Self {
        // Ensure unit vector direction for consistent length checks
        let dir = direction.normalized();

        let block_position = origin.map(|x| x.floor() as i32);

        // 1. Determine step direction per axis (+1, -1, or 0)
        let step = dir.map(|d| {
            if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            }
        });

        // 2. Distance to cross 1 full voxel along each axis (avoiding div by 0 using INFINITY)
        let t_delta = dir.map(|d| {
            if d == 0.0 {
                f32::INFINITY
            } else {
                (1.0 / d).abs()
            }
        });

        // 3. Distance from origin to first grid boundary (voxel face)
        let t_max = origin.map2(dir, |o, d| {
            if d > 0.0 {
                (o.floor() + 1.0 - o) / d
            } else if d < 0.0 {
                (o - o.floor()) / -d
            } else {
                f32::INFINITY
            }
        });

        Self {
            step,
            t_max,
            t_delta,
            block_position,
            max_len,
            hit_face: 0,
            first_step: true,
        }
    }
}

impl Iterator for Ray {
    type Item = (Vec3<i32>, usize); // (block_position, face_index)

    fn next(&mut self) -> Option<Self::Item> {
        // 1. First iteration yields the starting voxel (face is dummy 0)
        if self.first_step {
            self.first_step = false;
            return Some((self.block_position, self.hit_face));
        }

        // 2. Pick axis with smallest distance to next boundary
        let axis = if self.t_max.x < self.t_max.y {
            if self.t_max.x < self.t_max.z { 0 } else { 2 }
        } else if self.t_max.y < self.t_max.z {
            1
        } else {
            2
        };

        let current_t = self.t_max[axis];

        // 3. Stop if distance exceeds ray range
        if current_t > self.max_len {
            return None;
        }

        // 4. Step along the winning axis & update boundary distance
        self.block_position[axis] += self.step[axis];
        self.t_max[axis] += self.t_delta[axis];

        // 5. Calculate entry face (-X:0, +X:1, -Y:2, +Y:3, -Z:4, +Z:5)
        self.hit_face = axis * 2 + usize::from(self.step[axis] < 0);

        Some((self.block_position, self.hit_face))
    }
}
