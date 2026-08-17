use image::{DynamicImage, GenericImageView, ImageBuffer, ImageError, Rgba};

pub struct Texture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

// =========================================================================
// Color Space & Linear Downsampling Helpers
// =========================================================================

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let clamped = c.clamp(0.0, 1.0);
    if clamped <= 0.0031308 {
        clamped * 12.92
    } else {
        1.055 * clamped.powf(1.0 / 2.4) - 0.055
    }
}

fn rgba8_to_linear_f32(img: &image::RgbaImage) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
    let (w, h) = img.dimensions();
    ImageBuffer::from_fn(w, h, |x, y| {
        let [r, g, b, a] = img.get_pixel(x, y).0;
        Rgba([
            srgb_to_linear(f32::from(r) / 255.0),
            srgb_to_linear(f32::from(g) / 255.0),
            srgb_to_linear(f32::from(b) / 255.0),
            f32::from(a) / 255.0, // Alpha is already linear
        ])
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn linear_f32_to_rgba8(src: &ImageBuffer<Rgba<f32>, Vec<f32>>) -> image::RgbaImage {
    let (w, h) = src.dimensions();
    ImageBuffer::from_fn(w, h, |x, y| {
        let Rgba([r, g, b, a]) = *src.get_pixel(x, y);
        let s_r = (linear_to_srgb(r) * 255.0).round() as u8;
        let s_g = (linear_to_srgb(g) * 255.0).round() as u8;
        let s_b = (linear_to_srgb(b) * 255.0).round() as u8;
        let s_a = (a.clamp(0.0, 1.0) * 255.0).round() as u8;
        Rgba([s_r, s_g, s_b, s_a])
    })
}

/// Downsamples an f32 linear buffer by 2x in linear color space with alpha-weighting.
fn halve_linear_image(src: &ImageBuffer<Rgba<f32>, Vec<f32>>) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
    let (width, height) = src.dimensions();
    let new_w = (width / 2).max(1);
    let new_h = (height / 2).max(1);

    ImageBuffer::from_fn(new_w, new_h, |x, y| {
        let base_x = x * 2;
        let base_y = y * 2;

        let mut weighted_r = 0.0f32;
        let mut weighted_g = 0.0f32;
        let mut weighted_b = 0.0f32;
        let mut total_alpha = 0.0f32;

        for dy in 0..2 {
            for dx in 0..2 {
                let px_x = (base_x + dx).min(width - 1);
                let px_y = (base_y + dy).min(height - 1);
                let Rgba([r, g, b, a]) = *src.get_pixel(px_x, px_y);

                weighted_r += r * a;
                weighted_g += g * a;
                weighted_b += b * a;
                total_alpha += a;
            }
        }

        if total_alpha > 0.0 {
            let avg_r = weighted_r / total_alpha;
            let avg_g = weighted_g / total_alpha;
            let avg_b = weighted_b / total_alpha;
            let avg_a = total_alpha / 4.0;
            Rgba([avg_r, avg_g, avg_b, avg_a])
        } else {
            Rgba([0.0, 0.0, 0.0, 0.0])
        }
    })
}

// =========================================================================
// Texture Implementation
// =========================================================================

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    #[must_use]
    pub fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[Self::DEPTH_FORMAT],
        };
        let texture = device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self { view, sampler }
    }

    /// # Errors
    ///
    /// If bytes cannot be loaded as an image
    pub fn from_bytes_mip_array(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[&[u8]],
        label: &str,
    ) -> Result<Self, ImageError> {
        Ok(Self::from_images_mip_array(
            device,
            queue,
            &bytes
                .iter()
                .map(|b| image::load_from_memory(b))
                .collect::<Result<Vec<DynamicImage>, _>>()?,
            Some(label),
        ))
    }

    /// # Errors
    ///
    /// If bytes cannot be loaded as an image
    pub fn from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, ImageError> {
        Ok(Self::from_image(
            device,
            queue,
            &image::load_from_memory(bytes)?,
            Some(label),
        ))
    }

    #[must_use]
    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Self {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });

        write_texture(queue, &texture, &rgba, size, 0);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self { view, sampler }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_images_mip_array(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        imgs: &[image::DynamicImage],
        label: Option<&str>,
    ) -> Self {
        assert!(
            !imgs.is_empty(),
            "Cannot create a texture array from zero images!"
        );

        // 1. Get base dimensions and assert all images match
        let (base_width, base_height) = imgs[0].dimensions();
        assert!(
            imgs.iter()
                .all(|img| img.dimensions() == (base_width, base_height)),
            "All images in a texture array must have identical dimensions!"
        );

        let num_layers = imgs.len() as u32;
        let mip_level_count = 4;

        let texture_size = wgpu::Extent3d {
            width: base_width,
            height: base_height,
            depth_or_array_layers: num_layers,
        };

        // 2. Allocate the 2D Texture Array
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: texture_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // 3. Convert all base images to linear f32 buffers once to avoid precision degradation
        let mut current_linear_buffers: Vec<ImageBuffer<Rgba<f32>, Vec<f32>>> = imgs
            .iter()
            .map(image::DynamicImage::to_rgba8)
            .map(|rgba| rgba8_to_linear_f32(&rgba))
            .collect();

        let mut current_width = base_width;
        let mut current_height = base_height;

        // 4. Upload mip levels dynamically
        for mip in 0..mip_level_count {
            for (layer_idx, linear_buffer) in current_linear_buffers.iter().enumerate() {
                // Convert back to u8 sRGB solely for GPU upload
                let rgba_u8 = linear_f32_to_rgba8(linear_buffer);

                queue.write_texture(
                    wgpu::TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_idx as u32,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba_u8,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * current_width),
                        rows_per_image: Some(current_height),
                    },
                    wgpu::Extent3d {
                        width: current_width,
                        height: current_height,
                        depth_or_array_layers: 1,
                    },
                );
            }

            // Downsample linear f32 buffers for the NEXT mip level loop
            if mip < mip_level_count - 1 {
                current_width = (current_width / 2).max(1);
                current_height = (current_height / 2).max(1);
                current_linear_buffers = current_linear_buffers
                    .iter()
                    .map(halve_linear_image)
                    .collect();
            }
        }

        // 5. Create View explicitly as a D2Array
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // 6. Hook up the sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self { view, sampler }
    }
}

fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    rgba: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    size: wgpu::Extent3d,
    mip_level: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfoBase {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height),
        },
        size,
    );
}
