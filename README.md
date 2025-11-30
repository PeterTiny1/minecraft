# The Project
A voxel game engine written in Rust, utilizing wgpu for modern graphics rendering. Originally started as a playground to explore graphics programming.
# Technical Stack
- Language: Rust
- Graphics: wgpu (WebGPU for native)
- Math: vek
- Noise: noise-rs (OpenSimplex)
- Build System: Cargo
# Key Features
- Infinite Terrain Generation: Procedurally generated terrain using OpenSimplex noise.
- Meshing: Implements standard face culling to minimize geometry
- Cross-Platform: Runs on Linux and Windows.
# Acknowledgements
- [Learn Wgpu](https://sotrh.github.io/learn-wgpu/): The rendering pipeline boilerplate and initialisation code is heavily based on this excellent tutorial by @sotrh.
