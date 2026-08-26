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
            Self::NegX => Vec3::new(-1, 0, 0),
            Self::PosX => Vec3::new(1, 0, 0),
            Self::NegY => Vec3::new(0, -1, 0),
            Self::PosY => Vec3::new(0, 1, 0),
            Self::NegZ => Vec3::new(0, 0, -1),
            Self::PosZ => Vec3::new(0, 0, 1),
        }
    }

    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Self::NegX => Self::PosX,
            Self::PosX => Self::NegX,
            Self::NegY => Self::PosY,
            Self::PosY => Self::NegY,
            Self::NegZ => Self::PosZ,
            Self::PosZ => Self::NegZ,
        }
    }

    #[inline]
    pub fn from_axis_and_step(axis: usize, step_positive: bool) -> Self {
        match (axis, step_positive) {
            (0, false) => Self::NegX,
            (0, true) => Self::PosX,
            (1, false) => Self::NegY,
            (1, true) => Self::PosY,
            (2, false) => Self::NegZ,
            (2, true) => Self::PosZ,
            _ => panic!("axis out of range: {}", axis),
        }
    }
}
