use image::{GenericImageView, ImageBuffer, ImageError, Rgba};

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

fn average(pixels: [&Rgba<u8>; 4]) -> Rgba<u8> {
    let total = pixels.map(|p| u16::from(p.0[3])).iter().sum::<u16>();
    if total == 0 {
        Rgba([0, 0, 0, 0])
    } else {
        Rgba({
            let mut d = pixels
                .into_iter()
                .map(|v| {
                    let alpha = u16::from(v.0[3]);
                    [
                        (u16::from(v.0[0]) * alpha / total) as u8,
                        (u16::from(v.0[1]) * alpha / total) as u8,
                        (u16::from(v.0[2]) * alpha / total) as u8,
                        0,
                    ]
                })
                .reduce(|acc, v| [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2], 0])
                .unwrap();
            d[3] = (total / 4) as u8;
            d
        })
    }
}

fn downsample(texture: ImageBuffer<Rgba<u8>, Vec<u8>>) -> impl Fn(u32, u32) -> Rgba<u8> {
    move |x, y| {
        average([
            texture.get_pixel(x * 2, y * 2),
            texture.get_pixel(x * 2 + 1, y * 2),
            texture.get_pixel(x * 2, y * 2 + 1),
            texture.get_pixel(x * 2 + 1, y * 2 + 1),
        ])
    }
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
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
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

        Self {
            texture,
            view,
            sampler,
        }
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
        let rgba1 =
            ImageBuffer::from_fn(dimensions.0 / 2, dimensions.1 / 2, downsample(rgba.clone()));
        let rgba2 = ImageBuffer::from_fn(
            dimensions.0 / 4,
            dimensions.1 / 4,
            downsample(rgba1.clone()),
        );
        let rgba3 = ImageBuffer::from_fn(
            dimensions.0 / 8,
            dimensions.1 / 8,
            downsample(rgba2.clone()),
        );
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 4,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
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
        wgpu::ImageCopyTexture {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: std::num::NonZeroU32::new(4 * size.width),
            rows_per_image: std::num::NonZeroU32::new(size.height),
        },
        size,
    );
}
