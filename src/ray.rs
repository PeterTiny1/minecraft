use vek::Vec3;

pub struct Ray {
    step: Vec3<i32>,
    t_max: Vec3<f32>,
    t_delta: Vec3<f32>,
    block_position: Vec3<i32>,
    max_len: f32,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3<f32>, direction: Vec3<f32>, max_len: f32) -> Self {
        // 1. Calculate step per axis: +1 if dir > 0, -1 if dir < 0, 0 if dir == 0
        let step = direction.map(|d| d.signum() as i32);

        // 2. Target plane calculation: floor() + 1.0 for positive, floor() for negative
        let target_plane = origin.map2(direction, |o, d| {
            if d > 0.0 {
                o.floor() + 1.0
            } else if d < 0.0 {
                o.ceil() - 1.0
            } else {
                f32::INFINITY
            }
        });

        // 3. Distance t to first plane: (target - origin) / direction
        let t_max = (target_plane - origin) / direction;

        // 4. Distance t to cross 1 full voxel along ray
        let t_delta = direction.map(|d| (1.0 / d).abs());

        Self {
            step,
            t_max,
            t_delta,
            block_position: origin.map(|x| x.floor() as i32),
            max_len,
        }
    }
}

impl Iterator for Ray {
    type Item = (Vec3<i32>, usize); // (block_position, face_index)

    fn next(&mut self) -> Option<Self::Item> {
        // 1. Find which axis has the smallest t_max (0 for X, 1 for Y, 2 for Z)
        let (axis, &current_t) = self
            .t_max
            .as_slice()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;

        // 2. Check if we've exceeded the ray's maximum length
        if current_t > self.max_len {
            return None;
        }

        // 3. Step along the winning axis
        self.block_position[axis] += self.step[axis];
        self.t_max[axis] += self.t_delta[axis];

        // 4. Calculate hit face index (0-5: -X, +X, -Y, +Y, -Z, +Z)
        // If step is positive (+1), face is 2*axis + 1. If negative (-1), 2*axis.
        let is_positive = self.step[axis] > 0;
        let face = axis * 2 + usize::from(is_positive);

        Some((self.block_position, face))
    }
}
