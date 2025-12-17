struct UiUniform {
    aspect: f32,
};
@group(1) @binding(0)
var<uniform> uniforms: UiUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position * mat2x2<f32>(1.0, 0.0, 0.0, uniforms.aspect), 0.0, 1.0);
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
    return texture_sample;
}
