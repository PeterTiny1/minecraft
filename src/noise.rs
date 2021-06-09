fn interpolate(a0: f32, a1: f32, w: f32) -> f32 {
    return if 0.0 > w {
        a0
    } else if 1.0 < w {
        a1
    } else {
        (a1 - a0) * (3.0 - w * 2.0) * w * w + a0
    };
}

fn random_gradient(ix: i32, iy: i32) -> (f32, f32) {
    let random = 2920.0
        * (ix as f32 * 21942.0 + iy as f32 * 171324.0 + 8912.0).sin()
        * (ix as f32 * 23157.0 * iy as f32 * 217832.0 + 9758.0).cos();
    (random.cos(), random.sin())
}

fn dot_grid_gradient(ix: i32, iy: i32, x: f32, y: f32) -> f32 {
    let gradient = random_gradient(ix, iy);
    let dx = x - ix as f32;
    let dy = y - iy as f32;
    dx * gradient.0 + dy * gradient.1
}

pub fn perlin(x: f32, y: f32) -> f32 {
    let x0 = x as i32;
    let x1 = x0 + 1;
    let y0 = y as i32;
    let y1 = y0 + 1;

    let sx = x - x0 as f32;
    let sy = y - y0 as f32;
    let (mut n0, mut n1, ix0, ix1, value);
    n0 = dot_grid_gradient(x0, y0, x, y);
    n1 = dot_grid_gradient(x1, y0, x, y);
    ix0 = interpolate(n0, n1, sx);
    n0 = dot_grid_gradient(x0, y1, x, y);
    n1 = dot_grid_gradient(x1, y1, x, y);
    ix1 = interpolate(n0, n1, sx);

    value = interpolate(ix0, ix1, sy);
    value
}
