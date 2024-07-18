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
        #[allow(clippy::cast_possible_truncation)]
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

#[inline]
fn calculate_delta(
    pos_on_axis: f32,
    direction: Vec3<f32>,
    amount_in_direction: f32,
) -> (bool, Vec3<f32>) {
    let direction_positive = amount_in_direction.is_sign_positive();
    let delta = calculate_delta_(
        direction_positive,
        pos_on_axis,
        amount_in_direction,
        direction,
    );
    (direction_positive, delta)
}

fn calculate_delta_(
    direction_positive: bool,
    pos_on_axis: f32,
    amount_in_direction: f32,
    direction: Vec3<f32>,
) -> Vec3<f32> {
    let possible_delta = (if direction_positive {
        pos_on_axis.ceil()
    } else {
        pos_on_axis.floor()
    } - pos_on_axis)
        / amount_in_direction
        * direction;
    if possible_delta.is_zero() {
        1.0 / amount_in_direction.abs() * direction
    } else {
        possible_delta
    }
}

impl Iterator for Ray {
    type Item = (Vec3<i32>, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let (positive_x, dx) = calculate_delta(self.position.x, self.direction, self.direction.x);
        let (positive_y, dy) = calculate_delta(self.position.y, self.direction, self.direction.y);
        let (positive_z, dz) = calculate_delta(self.position.z, self.direction, self.direction.z);

        let (direction, &real_change) = [dx, dy, dz]
            .iter()
            .enumerate()
            .reduce(|acc, item| {
                if item.1.magnitude_squared() < acc.1.magnitude_squared() {
                    item
                } else {
                    acc
                }
            })
            .unwrap();
        self.position += real_change;
        let direction =
            direction * 2 + usize::from([positive_x, positive_y, positive_z][direction]);
        self.block_position += DIRECTION_OFFSETS[direction];
        if self.magnitude() < self.max_len {
            Some((self.block_position, direction))
        } else {
            None
        }
    }
}
