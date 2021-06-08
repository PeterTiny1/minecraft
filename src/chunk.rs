use crate::Vertex;

#[derive(Clone, Debug)]
pub struct Chunk {
    pub location: [i32; 2],
    pub contents: [[[u16; 16]; 256]; 16],
    pub mesh: Vec<Vertex>,
}
