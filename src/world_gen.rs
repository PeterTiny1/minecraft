use crate::{
    block::BlockType,
    chunk::{
        CHUNK_DEPTH, CHUNK_DEPTH_I32, CHUNK_HEIGHT, CHUNK_SIZE, CHUNK_WIDTH, CHUNK_WIDTH_I32, Chunk,
    },
    world::block_index,
};
use noise::{NoiseFn, OpenSimplex};
#[derive(Clone, Copy)]
enum Biome {
    BirchFalls,
    GreenGrove,
    DarklogForest,
    // PineHills,
    // SnowDesert,
}

#[allow(clippy::cast_possible_truncation)]
fn noise_at(
    noise: &OpenSimplex,
    x: i32,
    z: i32,
    chunk_location: [i32; 2],
    scale: f64,
    offset: f64,
) -> f64 {
    noise.get([
        f64::from(x + (chunk_location[0] * CHUNK_WIDTH_I32)) / scale + offset,
        f64::from(z + (chunk_location[1] * CHUNK_DEPTH_I32)) / scale + offset,
    ])
}

const WATER_HEIGHT: usize = 64;
const BIOME_SCALE: f64 = 250.0;
// const SCALING_FACTOR: f64 = 0.011;

#[must_use]
pub fn generate(noise: &OpenSimplex, location: [i32; 2]) -> Chunk {
    let heightmap = generate_worldscale_heightmap(noise, location);
    let biomemap = generate_biomemap(noise, location);
    let mut contents: Chunk = vec![BlockType::Air; CHUNK_SIZE]
        .into_boxed_slice()
        .try_into()
        .expect("chunk size mismatch");

    for x in 0..CHUNK_WIDTH {
        for y in (0..CHUNK_HEIGHT).rev() {
            for z in 0..CHUNK_DEPTH {
                let terrain_height = heightmap[x][z];
                let biome = biomemap[x][z];
                contents[block_index(x, y, z)] =
                    determine_type(terrain_height, x, y, z, biome, noise);
            }
        }
    }
    let trees = generate_trees(noise, location);
    for (x, z) in trees {
        let height = heightmap[x][z];
        if height <= WATER_HEIGHT {
            continue;
        }
        let biome = biomemap[x][z];
        place_tree(biome, &mut contents, x, height, z);
    }
    contents
}

const fn place_tree(biome: Biome, contents: &mut Chunk, x: usize, height: usize, z: usize) {
    let wood_type = match biome {
        Biome::BirchFalls => BlockType::BirchWood,
        Biome::GreenGrove => BlockType::Wood,
        Biome::DarklogForest => BlockType::DarkWood,
    };
    let leaf_type = match biome {
        Biome::BirchFalls => BlockType::BirchLeaf,
        Biome::GreenGrove => BlockType::Leaf,
        Biome::DarklogForest => BlockType::DarkLeaf,
    };
    contents[block_index(x, height + 1, z)] = wood_type;
    contents[block_index(x, height + 2, z)] = wood_type;
    contents[block_index(x, height + 3, z)] = wood_type;
    contents[block_index(x, height + 4, z)] = wood_type;
    contents[block_index(x, height + 5, z)] = leaf_type;
}

#[allow(clippy::cast_possible_truncation)]
fn generate_biomemap(
    noise: &OpenSimplex,
    chunk_location: [i32; 2],
) -> [[Biome; CHUNK_DEPTH]; CHUNK_WIDTH] {
    std::array::from_fn(|x| {
        std::array::from_fn(|z| {
            let v = noise_at(noise, x as i32, z as i32, chunk_location, BIOME_SCALE, 18.9);
            determine_biome(v)
        })
    })
}

fn determine_biome(v: f64) -> Biome {
    if v > 0.2 {
        Biome::DarklogForest
    } else if v > -0.1 {
        Biome::GreenGrove
    } else {
        Biome::BirchFalls
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn generate_worldscale_heightmap(
    noise: &OpenSimplex,
    location: [i32; 2],
) -> [[usize; CHUNK_DEPTH]; CHUNK_WIDTH] {
    generate_heightmap(noise, location).map(|a| a.map(|b| b as usize))
}

const HEIGHT_SCALE: f64 = 96.0;
#[allow(clippy::cast_precision_loss)]
fn generate_heightmap(
    noise: &OpenSimplex,
    location: [i32; 2],
) -> [[f64; CHUNK_DEPTH]; CHUNK_WIDTH] {
    const OCTAVES: usize = 5;
    const PERSISTENCE: f64 = 0.5;
    const LACUNARITY: f64 = 2.0;

    let world_x_base = f64::from(location[0] * CHUNK_WIDTH_I32);
    let world_z_base = f64::from(location[1] * CHUNK_DEPTH_I32);

    std::array::from_fn(|x| {
        let wx = world_x_base + x as f64;

        std::array::from_fn(|z| {
            let wz = world_z_base + z as f64;

            let mut amplitude = 1.0;
            let mut frequency = 0.007;
            let mut noise_height = 0.8;
            let mut gradient_x = 0.0;
            let mut gradient_z = 0.0;

            for octave in 0..OCTAVES {
                let octave_z = (octave * 5) as f64;
                let sample_x = wx * frequency;
                let sample_z = wz * frequency;
                let offset = 0.1 * frequency;

                // X-axis gradient (finite differences)
                let sample_a = noise.get([sample_x - offset, sample_z, octave_z]);
                let sample_b = noise.get([sample_x + offset, sample_z, octave_z]);
                gradient_x += (sample_a - sample_b) * amplitude;

                // Z-axis gradient (finite differences)
                let sample_c = noise.get([sample_x, sample_z - offset, octave_z]);
                let sample_d = noise.get([sample_x, sample_z + offset, octave_z]);
                gradient_z += (sample_c - sample_d) * amplitude;

                let effect = 1.0 / (1.0 + gradient_x.hypot(gradient_z));
                let octave_noise = noise.get([sample_x, sample_z, octave_z]);

                noise_height = (octave_noise * amplitude).mul_add(effect, noise_height);

                amplitude *= PERSISTENCE;
                frequency *= LACUNARITY;
            }

            noise_height * HEIGHT_SCALE
        })
    })
}

#[allow(clippy::cast_precision_loss)]
#[inline]
fn determine_type(
    terrain_height: usize,
    x: usize,
    y: usize,
    z: usize,
    biome: Biome,
    noise: &OpenSimplex,
) -> BlockType {
    if y < terrain_height - 3 {
        BlockType::Stone
    } else if y < terrain_height {
        BlockType::Dirt
    } else if y == terrain_height {
        if terrain_height > WATER_HEIGHT {
            match biome {
                Biome::BirchFalls => BlockType::GrassBlock0,
                Biome::GreenGrove => BlockType::GrassBlock1,
                Biome::DarklogForest => BlockType::GrassBlock2,
            }
        } else {
            BlockType::Sand
        }
    } else if y < WATER_HEIGHT {
        BlockType::Water
    } else if terrain_height > WATER_HEIGHT
        && y == terrain_height + 1
        && noise.get([x as f64 / 4.0, z as f64 / 4.0, y as f64 / 4.0]) > 0.3
    {
        if noise.get([x as f64, y as f64, z as f64]) > 0.3 {
            match biome {
                Biome::BirchFalls => BlockType::Flower0,
                Biome::GreenGrove => BlockType::Flower1,
                Biome::DarklogForest => BlockType::Flower2,
            }
        } else {
            match biome {
                Biome::BirchFalls => BlockType::Grass0,
                Biome::GreenGrove => BlockType::Grass1,
                Biome::DarklogForest => BlockType::Grass2,
            }
        }
    } else {
        BlockType::Air
    }
}

#[allow(clippy::cast_sign_loss)]
fn generate_trees(noise: &OpenSimplex, location: [i32; 2]) -> Vec<(usize, usize)> {
    let [chunk_x, chunk_z] = location;
    let mut trees = vec![];
    for x in 0..CHUNK_WIDTH_I32 {
        for z in 0..CHUNK_DEPTH_I32 {
            let world_x = chunk_x * CHUNK_WIDTH_I32 + x;
            let world_z = chunk_z * CHUNK_DEPTH_I32 + z;

            let density_threshold =
                noise.get([f64::from(world_x) / 40.0, f64::from(world_z) / 40.0]) / 2.0 + 0.5;
            if noise.get([f64::from(world_x) * 3.0, f64::from(world_z) * 3.0]) > density_threshold {
                trees.push((x as usize, z as usize));
            }
        }
    }

    trees
}
