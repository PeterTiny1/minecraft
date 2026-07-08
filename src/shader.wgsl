enable f16;
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
    @location(2) tex_index: u32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = uniforms.view_proj * vec4<f32>(in.position, 1.0);

    // 1. Unpack UV coordinates (e.g., 0 to 16 pixel bounds mapped to 0.0-1.0)
    // We cast the vec2<u32> to vec2<f32> and divide by our texture limit
    out.tex_coords = vec2<f32>(in.data.xy) / 16.0;

    // 2. Unpack texture array index (remains an integer, e.g., 0 to 255)
    out.tex_index = in.data.z;

    // 3. Unpack light level (maps 0-255 byte scale back to a clean 0.0-1.0 float)
    out.brightness = f32(in.data.w) / 255.0;

    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d_array<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_sample = textureSample(t_diffuse, s_diffuse, vec2<f32>(in.tex_coords), in.tex_index);
    // if (texture_sample.a <= 0.1) { discard; }
    let shadow_color = vec3<f32>(0.06, 0.0, 0.1);
    let final_color = mix(shadow_color, texture_sample.rgb, in.brightness);
    let faded = vec4<f32>(mix(vec3<f32>(0.2, 0.3, 0.4), final_color, vec3<f32>(clamp((-in.clip_position.z + 1.) * 2000.0, 0.0, 1.0))), texture_sample.a);
    if (faded.a <= 0.001) { discard; };
    return faded;
}
