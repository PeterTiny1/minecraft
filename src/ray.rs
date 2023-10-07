use vek::{num_traits::Zero, Vec3};

const DIRECTION_OFFSETS: [Vec3<i32>; 6] = [
    Vec3 { x: -1, y: 0, z: 0 },
    Vec3 { x: 1, y: 0, z: 0 },
    Vec3 { x: 0, y: -1, z: 0 },
    Vec3 { x: 0, y: 1, z: 0 },
    Vec3 { x: 0, y: 0, z: -1 },
    Vec3 { x: 0, y: 0, z: 1 },
];

pub struct Ray {
    origin: Vec3<f32>,
    direction: Vec3<f32>,
    position: Vec3<f32>,
    block_position: Vec3<i32>,
    max_len: f32,
}

impl Ray {
    #[must_use]
    pub fn new(origin: Vec3<f32>, direction: Vec3<f32>, max_len: f32) -> Self {
        Self {
            origin,
            direction,
            position: origin,
            block_position: origin.map(|x| x.floor() as i32),
            max_len,
        }
    }
    fn magnitude(&self) -> f32 {
        self.origin.distance(self.position)
    }
}

fn calculate_delta(
    axis_position: f32,
    direction: Vec3<f32>,
    axis_direction: f32,
) -> (bool, Vec3<f32>) {
    let positive_direction = axis_direction.is_sign_positive();
    let delta = {
        let possible = (if positive_direction {
            axis_direction.ceil()
        } else {
            axis_position.floor()
        } - axis_position)
            / axis_direction
            * direction;
        if possible.is_zero() {
            assert!(axis_direction != 0.0);
            assert!(!direction.is_zero());
            1.0 / axis_direction.abs() * direction
        } else {
            possible
        }
    };
    (positive_direction, delta)
}

impl Iterator for Ray {
    type Item = (Vec3<i32>, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let positive_x = self.direction.x.is_sign_positive();
        let dx: Vec3<f32> = {
            let possible = (if positive_x {
                self.position.x.ceil()
            } else {
                self.position.x.floor()
            } - self.position.x)
                / self.direction.x
                * self.direction;

            if possible.is_zero() {
                1.0 / self.direction.x.abs() * self.direction
            } else {
                possible
            }
        };
        let positive_y = self.direction.y.is_sign_positive();
        let dy: Vec3<f32> = {
            let possible = (if positive_y {
                self.position.y.ceil()
            } else {
                self.position.y.floor()
            } - self.position.y)
                / self.direction.y
                * self.direction;
            if possible.is_zero() {
                1.0 / self.direction.y.abs() * self.direction
            } else {
                possible
            }
        };
        let positive_z = self.direction.z.is_sign_positive();
        let dz: Vec3<f32> = {
            let possible = (if positive_z {
                self.position.z.ceil()
            } else {
                self.position.z.floor()
            } - self.position.z)
                / self.direction.z
                * self.direction;
            if possible.is_zero() {
                1.0 / self.direction.z.abs() * self.direction
            } else {
                possible
            }
        };
        let (direction, &real_change) = [dx, dy, dz]
            .iter()
            .enumerate()
            .reduce(|acc, item| {
                if item.1.magnitude() < acc.1.magnitude() {
                    item
                } else {
                    acc
                }
            })
            .unwrap();
        self.position += real_change;
        let direction = direction * 2 + usize::from([positive_x, positive_y, positive_z][direction]);
        self.block_position += DIRECTION_OFFSETS[direction];
        if self.magnitude() < self.max_len {
            Some((self.block_position, direction))
        } else {
            None
        }
    }
}
