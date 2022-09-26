use vek::{num_traits::Zero, Vec3};

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

impl Iterator for Ray {
    type Item = (Vec3<i32>, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let positive_x = self.direction.x > 0.0;
        let changex = {
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
        let positive_y = self.direction.y > 0.0;
        let changey = {
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
        let positive_z = self.direction.z > 0.0;
        let changez = {
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
        let (direction, &real_change) = [changex, changey, changez]
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
        let direction = direction * 2
            + match direction {
                0 => positive_x,
                1 => positive_y,
                2 => positive_z,
                _ => false,
            } as usize;
        match direction {
            0 => self.block_position.x -= 1,
            1 => self.block_position.x += 1,
            2 => self.block_position.y -= 1,
            3 => self.block_position.y += 1,
            4 => self.block_position.z -= 1,
            5 => self.block_position.z += 1,
            _ => (),
        }
        if self.magnitude() < self.max_len {
            Some((self.block_position, direction))
        } else {
            None
        }
    }
}
