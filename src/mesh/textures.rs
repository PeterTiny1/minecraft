use crate::BlockType;

#[inline]
pub const fn get_texture_indices(block_type: BlockType) -> [u8; 6] {
    match block_type {
        BlockType::Stone => [0; 6],
        BlockType::Dirt => [1; 6],
        BlockType::GrassBlock0 => [2, 3, 3, 3, 3, 1],
        BlockType::GrassBlock1 => [4, 5, 5, 5, 5, 1],
        BlockType::GrassBlock2 => [6, 7, 7, 7, 7, 1],
        BlockType::BirchWood => [8, 9, 9, 9, 9, 8],
        BlockType::Wood => [10, 11, 11, 11, 11, 10],
        BlockType::DarkWood => [12, 13, 13, 13, 13, 12],
        BlockType::BirchLeaf => [14; 6],
        BlockType::Leaf => [15; 6],
        BlockType::DarkLeaf => [16; 6],
        BlockType::Grass0 => [17; 6],
        BlockType::Grass1 => [18; 6],
        BlockType::Grass2 => [19; 6],
        BlockType::Flower0 => [20; 6],
        BlockType::Flower1 => [21; 6],
        BlockType::Flower2 => [22; 6],
        BlockType::Sand => [23; 6],
        BlockType::Water => [24, 25, 25, 25, 25, 24],
        BlockType::Air => [0; 6],
    }
}
