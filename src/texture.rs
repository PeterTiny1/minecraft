use image::{DynamicImage, GenericImageView, ImageBuffer, ImageError, Rgba};

pub struct Texture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

fn halve_image_weighted(img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // New dimensions after halving
    let new_width = width / 2;
    let new_height = height / 2;

    // Create a new ImageBuffer to store the halved size image
    let mut new_img = ImageBuffer::new(new_width, new_height);

    // Process each 2x2 block
    for x in 0..new_width {
        for y in 0..new_height {
            // Use u32 for sums to prevent overflow
            let mut weighted_sum_r = 0;
            let mut weighted_sum_g = 0;
            let mut weighted_sum_b = 0;
            let mut sum_a = 0;
            let mut total_weight = 0;

            for dx in 0..2 {
                for dy in 0..2 {
                    let pixel = img.get_pixel(2 * x + dx, 2 * y + dy);
                    let r = pixel[0];
                    let g = pixel[1];
                    let b = pixel[2];
                    let a = pixel[3];

                    // The weight for each pixel's color is its own alpha value
                    let weight = u32::from(a);

                    // Add the weighted color values to the sums
                    weighted_sum_r += u32::from(r) * weight;
                    weighted_sum_g += u32::from(g) * weight;
                    weighted_sum_b += u32::from(b) * weight;

                    // The total weight is the sum of all alpha values
                    total_weight += weight;

                    // The new alpha will be a simple average of the source alphas
                    sum_a += u32::from(a);
                }
            }

            if let Some(div_r) = weighted_sum_r.checked_div(total_weight) {
                // Calculate the weighted average for the color channels
                let avg_r = u8::try_from(div_r).unwrap();
                let avg_g = u8::try_from(weighted_sum_g / total_weight).unwrap();
                let avg_b = u8::try_from(weighted_sum_b / total_weight).unwrap();

                // The new alpha is the simple average of the four pixels' alphas
                let avg_a = u8::try_from(sum_a / 4).unwrap();

                // Set the pixel in the new ImageBuffer
                new_img.put_pixel(x, y, Rgba([avg_r, avg_g, avg_b, avg_a]));
            } else {
                // If total_weight was 0, the new pixel is transparent black
                new_img.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }

    new_img
}

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
            mipmap_filter: wgpu::FilterMode::Nearest,
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
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self { view, sampler }
    }

    #[must_use]
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

        // 1. Get base dimensions from the first image
        let (base_width, base_height) = imgs[0].dimensions();
        let num_layers = imgs.len() as u32;
        let mip_level_count = 4; // Keeping your 4 levels, but we'll use a loop now

        let texture_size = wgpu::Extent3d {
            width: base_width,
            height: base_height,
            depth_or_array_layers: num_layers, // Look at us go, actual layers!
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

        // 3. Keep track of the current mip's image data buffers for all layers
        // Start with the raw RGBA8 buffers of your base images
        let mut current_mip_buffers: Vec<image::RgbaImage> =
            imgs.iter().map(image::DynamicImage::to_rgba8).collect();

        let mut current_width = base_width;
        let mut current_height = base_height;

        // 4. Loop through the mip levels dynamically
        for mip in 0..mip_level_count {
            // Upload each layer's data for the current mip level
            for (layer_idx, rgba_buffer) in current_mip_buffers.iter().enumerate() {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfoBase {
                        texture: &texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: layer_idx as u32, // Selects the specific array slice
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    rgba_buffer,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * current_width),
                        rows_per_image: Some(current_height),
                    },
                    wgpu::Extent3d {
                        width: current_width,
                        height: current_height,
                        depth_or_array_layers: 1, // Writing 1 layer at a time
                    },
                );
            }

            // Downsample all layers for the NEXT mip level loop
            if mip < mip_level_count - 1 {
                current_width /= 2;
                current_height /= 2;
                current_mip_buffers = current_mip_buffers
                    .iter()
                    .map(halve_image_weighted)
                    .collect();
            }
        }

        // 5. Create the View explicitly as a D2Array
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array), // Say goodbye to 2D flat view
            ..Default::default()
        });

        // 6. Hook up the sampler (Keeping your setup)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Linear,
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
