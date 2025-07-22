use image::{GenericImageView, ImageBuffer, ImageError, Rgba};

pub struct Texture {
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

fn halve_image(img: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    // New dimensions after halving
    let new_width = width / 2;
    let new_height = height / 2;

    // Create a new ImageBuffer to store the halved size image
    let mut new_img = ImageBuffer::new(new_width, new_height);

    // Process each 2x2 block
    for x in 0..new_width {
        for y in 0..new_height {
            // Compute the average color of the 2x2 block
            let mut sum_r = 0;
            let mut sum_g = 0;
            let mut sum_b = 0;
            let mut sum_a = 0;
            let mut count = 0;

            for dx in 0..2 {
                for dy in 0..2 {
                    let pixel = img.get_pixel(2 * x + dx, 2 * y + dy);
                    sum_r += u32::from(pixel[0]);
                    sum_g += u32::from(pixel[1]);
                    sum_b += u32::from(pixel[2]);
                    sum_a += u32::from(pixel[3]);
                    if pixel[3] != 0 {
                        count += 1;
                    }
                }
            }

            if count != 0 {
                // Average values
                let avg_r = u8::try_from(sum_r / count).unwrap();
                let avg_g = u8::try_from(sum_g / count).unwrap();
                let avg_b = u8::try_from(sum_b / count).unwrap();
                let avg_a = u8::try_from(sum_a / count).unwrap();

                // Set the pixel in the new ImageBuffer
                new_img.put_pixel(x, y, Rgba([avg_r, avg_g, avg_b, avg_a]));
            }
        }
    }

    new_img
}

impl Texture {
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

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

    pub fn from_bytes_mip(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, ImageError> {
        Ok(Self::from_image_mip(
            device,
            queue,
            &image::load_from_memory(bytes)?,
            Some(label),
        ))
    }

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

    pub fn from_image_mip(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> Self {
        let rgba = img.to_rgba8();
        let dimensions = img.dimensions();
        let half_size = |extent: wgpu::Extent3d| wgpu::Extent3d {
            width: extent.width / 2,
            height: extent.height / 2,
            depth_or_array_layers: extent.depth_or_array_layers,
        };
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let size1 = half_size(size);
        let size2 = half_size(size1);
        let size3 = half_size(size2);
        let rgba1 = halve_image(&rgba);
        let rgba2 = halve_image(&rgba1);
        let rgba3 = halve_image(&rgba2);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 4,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
        });

        write_texture(queue, &texture, &rgba, size, 0);
        write_texture(queue, &texture, &rgba1, size1, 1);
        write_texture(queue, &texture, &rgba2, size2, 2);
        write_texture(queue, &texture, &rgba3, size3, 3);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
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
