use vek::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Direction {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl Direction {
    #[inline]
    pub const fn offset(self) -> Vec3<i32> {
        match self {
            Direction::NegX => Vec3::new(-1, 0, 0),
            Direction::PosX => Vec3::new(1, 0, 0),
            Direction::NegY => Vec3::new(0, -1, 0),
            Direction::PosY => Vec3::new(0, 1, 0),
            Direction::NegZ => Vec3::new(0, 0, -1),
            Direction::PosZ => Vec3::new(0, 0, 1),
        }
    }

    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Direction::NegX => Direction::PosX,
            Direction::PosX => Direction::NegX,
            Direction::NegY => Direction::PosY,
            Direction::PosY => Direction::NegY,
            Direction::NegZ => Direction::PosZ,
            Direction::PosZ => Direction::NegZ,
        }
    }

    #[inline]
    pub fn from_axis_and_step(axis: usize, step_positive: bool) -> Self {
        match (axis, step_positive) {
            (0, false) => Direction::NegX,
            (0, true)  => Direction::PosX,
            (1, false) => Direction::NegY,
            (1, true)  => Direction::PosY,
            (2, false) => Direction::NegZ,
            (2, true)  => Direction::PosZ,
            _ => panic!("axis out of range: {}", axis),
        }
    }
}
