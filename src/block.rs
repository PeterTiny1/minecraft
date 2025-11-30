use bincode::{Decode, Encode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Encode, Decode)]
pub enum BlockType {
    #[default]
    Air,
    Stone,
    GrassBlock0,
    GrassBlock1,
    GrassBlock2,
    Grass0,
    Grass1,
    Grass2,
    Flower0,
    Flower1,
    Flower2,
    Wood,
    BirchWood,
    DarkWood,
    DarkLeaf,
    Leaf,
    BirchLeaf,
    Water,
    Sand,
    Dirt,
}

impl BlockType {
    #[inline]
    #[must_use] 
    pub const fn is_solid(self) -> bool {
        !self.is_transparent() || matches!(self, Self::Leaf | Self::BirchLeaf | Self::DarkLeaf)
    }

    #[inline]
    #[must_use] 
    pub const fn is_transparent(self) -> bool {
        matches!(
            self,
            Self::Air | Self::Leaf | Self::BirchLeaf | Self::DarkLeaf
        ) || self.is_liquid()
            || self.is_grasslike()
    }

    #[inline]
    #[must_use] 
    pub const fn is_liquid(self) -> bool {
        matches!(self, Self::Water)
    }

    #[inline]
    #[must_use] 
    pub const fn is_grasslike(self) -> bool {
        matches!(
            self,
            Self::Flower0
                | Self::Flower1
                | Self::Flower2
                | Self::Grass0
                | Self::Grass1
                | Self::Grass2
        )
    }
}
