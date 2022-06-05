use vek::{num_traits::Zero, Vec3};

pub struct Ray {
    origin: Vec3<f32>,
    direction: Vec3<f32>,
    position: Vec3<f32>,
    block_position: Vec3<i32>,
    max_len: f32,
}

impl Ray {
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

impl Iterator for Ray {
    type Item = Vec3<i32>;
    fn next(&mut self) -> Option<Self::Item> {
        let changex = {
            let positive = self.direction.x > 0.0;
            let possible = (if positive {
                self.position.x.ceil()
            } else {
                self.position.x.floor()
            } - self.position.x)
                / self.direction.x
                * self.direction;
            if possible.is_zero() {
                1.0 / self.direction.x.abs() * self.direction
            } else if self.direction.x == 0.0 || possible.iter().any(|i| i.abs() > 1.0) {
                Vec3 {
                    x: f32::INFINITY,
                    y: f32::INFINITY,
                    z: f32::INFINITY,
                }
            } else {
                possible
            }
        };
        let changey = {
            let positive = self.direction.y > 0.0;
            let possible = (if positive {
                self.position.y.ceil()
            } else {
                self.position.y.floor()
            } - self.position.y)
                / self.direction.y
                * self.direction;
            if possible.is_zero() {
                1.0 / self.direction.y.abs() * self.direction
            } else if self.direction.y == 0.0 || possible.iter().any(|i| i.abs() > 1.0) {
                Vec3 {
                    x: f32::INFINITY,
                    y: f32::INFINITY,
                    z: f32::INFINITY,
                }
            } else {
                possible
            }
        };
        let changez = {
            let positive = self.direction.z > 0.0;
            let possible = (if positive {
                self.position.z.ceil()
            } else {
                self.position.z.floor()
            } - self.position.z)
                / self.direction.z
                * self.direction;
            if possible.is_zero() {
                1.0 / self.direction.z.abs() * self.direction
            } else if self.direction.z == 0.0 || possible.iter().any(|i| i.abs() > 1.0) {
                Vec3 {
                    x: f32::INFINITY,
                    y: f32::INFINITY,
                    z: f32::INFINITY,
                }
            } else {
                possible
            }
        };
        let real_change = *[changex, changey, changez]
            .iter()
            .reduce(|acc, item| {
                if item.magnitude() < acc.magnitude() {
                    item
                } else {
                    acc
                }
            })
            .unwrap();
        self.position += real_change;
        if real_change == changex {
            self.block_position.x += 1;
        } else if real_change == changey {
            self.block_position.y += 1;
        } else {
            self.block_position.z += 1;
        }
        if self.magnitude() < self.max_len {
            Some(self.block_position)
        } else {
            None
        }
    }
}
