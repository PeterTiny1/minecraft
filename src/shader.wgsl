struct Uniforms {
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) data: vec4<u32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) brightness: f32,
    @location(2) @interpolate(flat) tex_index: u32,
};

// --- Shared Vertex Shader ---
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);

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
