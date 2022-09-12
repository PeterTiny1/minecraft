struct Uniforms {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) brightness: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) brightness: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = uniforms.view_proj * vec4<f32>(model.position, 1.0);
    out.brightness = model.brightness;
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_sample = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    if (texture_sample.a <= 0.001) { discard; }
    return mix(vec4<f32>(0.2, 0.3, 0.4, 1.0), texture_sample * vec4<f32>(vec3<f32>(in.brightness), 1.0), vec4<f32>(clamp((-in.clip_position.z + 1.) * 1000.0, 0.0, 1.0)));
}