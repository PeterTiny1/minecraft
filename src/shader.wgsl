struct Uniforms {
    view_proj: mat4x4<f32>,
};

struct ChunkOffset {
    offset: vec3<f32>,
    _padding: f32,
}

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;
@group(2) @binding(0)
var<storage, read> chunk_offsets: array<ChunkOffset>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) data: vec4<u32>,
};

struct CompactVertexInput {
    @location(0) packed_data: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) brightness: f32,
    @location(2) @interpolate(flat) tex_index: u32,
};
// Lookup table for the 4 corners of each of the 6 face directions
// Order: +X, -X, +Y, -Y, +Z, -Z
const FACE_OFFSETS = array<array<vec3<f32>, 4>, 6>(
    // +X (Right)
    array<vec3<f32>, 4>(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 1.0, 0.0)),
    // -X (Left)
    array<vec3<f32>, 4>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 1.0)),
    // +Y (Top)
    array<vec3<f32>, 4>(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 0.0)),
    // -Y (Bottom)
    array<vec3<f32>, 4>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 0.0)),
    // +Z (Front)
    array<vec3<f32>, 4>(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 1.0)),
    // -Z (Back)
    array<vec3<f32>, 4>(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0)),
);

// Standard quad UV coordinates mapped by corner index (0..3)
const UV_OFFSETS = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

const FACE_LIGHTNESS = array<f32, 6>(
    0.8, // +X (SIDE_BRIGHTNESS)
    0.8, // -X (SIDE_BRIGHTNESS)
    1.0, // +Y (TOP_BRIGHTNESS)
    0.6, // -Y (BOTTOM_BRIGHTNESS)
    0.9, // +Z (FRONT_BRIGHTNESS)
    0.7  // -Z (BACK_BRIGHTNESS)
);

const AO_FACTORS = array<f32, 4>(0.4, 0.6, 0.8, 1.0);

@vertex
fn vs_compact(
    in: CompactVertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    // 1. Bit-unpack fields from 32-bit u32
    let x_local    = f32(in.packed_data & 0x1Fu);         // Bits 0..4   (5 bits: 0..31)
    let y_local    = f32((in.packed_data >> 5u) & 0xFFu);  // Bits 5..12  (8 bits: 0..255)
    let z_local    = f32((in.packed_data >> 13u) & 0x1Fu); // Bits 13..17 (5 bits: 0..31)
    let v_idx      = (in.packed_data >> 18u) & 0x03u;      // Bits 18..19 (2 bits: 0..3)
    let face_idx   = (in.packed_data >> 20u) & 0x07u;      // Bits 20..22 (3 bits: 0..5)
    let ao_raw     = (in.packed_data >> 23u) & 0x03u;      // Bits 23..24 (2 bits: 0..3)
    let tex_idx    = (in.packed_data >> 25u) & 0x7Fu;      // Bits 25..31 (7 bits: 0..127)

    // 2. Compute position
    let block_pos = vec3<f32>(x_local, y_local, z_local);
    let corner_offset = FACE_OFFSETS[face_idx][v_idx];
    let chunk_offset = chunk_offsets[instance_idx].offset;

    let face_light = FACE_LIGHTNESS[face_idx];
    let ao_mult = AO_FACTORS[ao_raw];
    let final_light = face_light * ao_mult;

    let world_pos = block_pos + corner_offset + chunk_offset;

    // 3. Assemble output
    var out: VertexOutput;
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.tex_coords = UV_OFFSETS[v_idx];
    out.tex_index = tex_idx;
    out.brightness = final_light;

    return out;
}

// --- Shared Vertex Shader ---
@vertex
fn vs_explicit(
    in: VertexInput,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let chunk_offset = chunk_offsets[instance_idx].offset;
    let local_pos = in.position + chunk_offset;
    var out: VertexOutput;

    out.clip_position = uniforms.view_proj * vec4<f32>(local_pos, 1.0);

    // 1. Unpack UV coordinates
    out.tex_coords = vec2<f32>(in.data.xy) / 16.0;

    // 2. Unpack texture array index
    out.tex_index = in.data.z;

    // 3. Unpack light level (0-255 byte mapped to 0.0-1.0 float)
    out.brightness = f32(in.data.w) / 255.0;

    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d_array<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

// Shared Helper: Calculates directional lighting and fog application
fn apply_lighting_and_fog(texture_sample: vec4<f32>, brightness: f32, clip_z: f32) -> vec4<f32> {
    let shadow_color = vec3<f32>(0.06, 0.0, 0.1);
    let fog_color = vec3<f32>(0.2, 0.3, 0.4);

    // Apply chunk lighting/brightness tinting
    let lit_color = mix(shadow_color, texture_sample.rgb, brightness);

    // Distance-based fog factor
    let fog_factor = clamp((-clip_z + 1.0) * 2000.0, 0.0, 1.0);
    let final_rgb = mix(fog_color, lit_color, fog_factor);

    return vec4<f32>(final_rgb, texture_sample.a);
}

// ============================================================================
// 1. OPAQUE PASS (Dirt, Stone, Wood)
// - No discard
// - Fast execution path for the majority of scene geometry
// ============================================================================
@fragment
fn fs_opaque(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_sample = textureSample(t_diffuse, s_diffuse, in.tex_coords, in.tex_index);
    let color = apply_lighting_and_fog(texture_sample, in.brightness, in.clip_position.z);
    
    // Hardcoded fully opaque alpha
    return vec4<f32>(color.rgb, 1.0);
}

// ============================================================================
// 2. CUTOUT PASS (Flowers, Tall Grass, Leaves)
// - Discards pixels with alpha lower than threshold
// - Writes directly to Depth Buffer
// ============================================================================
@fragment
fn fs_cutout(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_sample = textureSample(t_diffuse, s_diffuse, in.tex_coords, in.tex_index);

    // Discard transparent pixels BEFORE performing expensive fog math
    if (texture_sample.a <= 0.1) {
        discard;
    }

    return apply_lighting_and_fog(texture_sample, in.brightness, in.clip_position.z);
}

// ============================================================================
// 3. TRANSLUCENT PASS (Water, Stained Glass)
// - NO discard (prevents Early-Z optimization breaks)
// - Blends with background pixels via pipeline Alpha Blending
// ============================================================================
@fragment
fn fs_translucent(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_sample = textureSample(t_diffuse, s_diffuse, in.tex_coords, in.tex_index);
    
    // Example: Option to apply custom tint or opacity multiplier to water here
    return apply_lighting_and_fog(texture_sample, in.brightness, in.clip_position.z);
}
