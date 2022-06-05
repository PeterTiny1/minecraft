use vek::Vec3;

struct Ray {
    origin: Vec3<f32>,
    direction: Vec3<f32>,
    position: Vec3<f32>,
    block_position: Vec3<i32>,
    max_len: f32,
}

impl Ray {
    fn new(origin: Vec3<f32>, direction: Vec3<f32>, max_len: f32) -> Self {
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
        let changex = (if self.direction.x > 0.0 {
            self.position.x.ceil()
        } else {
            self.position.x.floor()
        } - self.position.x)
            / self.direction.x
            * self.direction;
        let changey = (if self.direction.y > 0.0 {
            self.position.y.ceil()
        } else {
            self.position.y.floor()
        } - self.position.y)
            / self.direction.y
            * self.direction;
        let changez = (if self.direction.z > 0.0 {
            self.position.z.ceil()
        } else {
            self.position.z.floor()
        } - self.position.z)
            / self.direction.z
            * self.direction;
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
