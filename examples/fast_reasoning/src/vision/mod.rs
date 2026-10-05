use candle_core::Tensor;
use image::imageops::FilterType;
use std::path::Path;

pub mod transformer;

#[derive(Debug)]
pub enum VisionError {
    Image(String),
}

impl std::fmt::Display for VisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image(message) => write!(f, "cannot process image: {message}"),
        }
    }
}
impl std::error::Error for VisionError {}

pub struct PixelValues {
    pub values: Vec<f32>,
    pub width: usize,
    pub height: usize,
}

pub struct PreparedImage {
    /// Local crops in row-major order, followed by the global thumbnail.
    pub images: Vec<PixelValues>,
    /// Zero for an unsplit image; otherwise the local-crop grid dimensions.
    pub rows: usize,
    pub cols: usize,
}

/// Aspect-preserving tiled preprocessing: scale the long edge, snap to encoder-sized
/// tiles, emit row-major crops, then add a square global thumbnail.
pub fn preprocess_image_tiles(
    path: &Path,
    encoder_edge: usize,
    longest_edge: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<PreparedImage, VisionError> {
    if encoder_edge == 0 || longest_edge == 0 {
        return Err(VisionError::Image(
            "image processor dimensions must be nonzero".into(),
        ));
    }
    let image = image::open(path)
        .map_err(|error| VisionError::Image(error.to_string()))?
        .to_rgb8();
    if std::env::var("FAST_REASONING_IMAGE_PREPROCESSOR").as_deref() == Ok("c") {
        return preprocess_image_tiles_c_compatible(
            &image,
            encoder_edge,
            longest_edge,
            mean,
            std,
        );
    }
    let (width, height) = image.dimensions();
    let (resized_height, resized_width) =
        resize_long_edge_size(height as usize, width as usize, longest_edge);
    let image = image::imageops::resize(
        &image,
        resized_width as u32,
        resized_height as u32,
        FilterType::Lanczos3,
    );

    let (height, width) = (image.height() as usize, image.width() as usize);
    let (snapped_height, snapped_width) = snap_to_tile_grid(height, width, encoder_edge);
    let image = image::imageops::resize(
        &image,
        snapped_width as u32,
        snapped_height as u32,
        FilterType::Lanczos3,
    );
    let rows = snapped_height.div_ceil(encoder_edge);
    let cols = snapped_width.div_ceil(encoder_edge);
    let (rows, cols) = if rows > 1 || cols > 1 {
        (rows, cols)
    } else {
        (0, 0)
    };

    let mut images = Vec::new();
    if rows > 0 {
        for row in 0..rows {
            for col in 0..cols {
                let crop = image::imageops::crop_imm(
                    &image,
                    (col * encoder_edge) as u32,
                    (row * encoder_edge) as u32,
                    encoder_edge as u32,
                    encoder_edge as u32,
                )
                .to_image();
                images.push(normalize_pixels(&crop, mean, std));
            }
        }
    }
    let global = image::imageops::resize(
        &image,
        encoder_edge as u32,
        encoder_edge as u32,
        FilterType::Lanczos3,
    );
    images.push(normalize_pixels(&global, mean, std));
    Ok(PreparedImage { images, rows, cols })
}

fn preprocess_image_tiles_c_compatible(
    decoded: &image::RgbImage,
    encoder_edge: usize,
    longest_edge: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<PreparedImage, VisionError> {
    let (resized_height, resized_width) = resize_long_edge_size(
        decoded.height() as usize,
        decoded.width() as usize,
        longest_edge,
    );
    let mut image = resize_lanczos3_c_compatible(
        decoded,
        resized_width as u32,
        resized_height as u32,
    )?;
    let (snapped_height, snapped_width) = snap_to_tile_grid(
        image.height() as usize,
        image.width() as usize,
        encoder_edge,
    );
    image = resize_lanczos3_c_compatible(
        &image,
        snapped_width as u32,
        snapped_height as u32,
    )?;
    let rows = snapped_height.div_ceil(encoder_edge);
    let cols = snapped_width.div_ceil(encoder_edge);
    let (rows, cols) = if rows > 1 || cols > 1 {
        (rows, cols)
    } else {
        (0, 0)
    };
    let mut images = Vec::new();
    if rows > 0 {
        for row in 0..rows {
            for col in 0..cols {
                let crop = image::imageops::crop_imm(
                    &image,
                    (col * encoder_edge) as u32,
                    (row * encoder_edge) as u32,
                    encoder_edge as u32,
                    encoder_edge as u32,
                )
                .to_image();
                images.push(normalize_pixels(&crop, mean, std));
            }
        }
    }
    let global = resize_lanczos3_c_compatible(
        &image,
        encoder_edge as u32,
        encoder_edge as u32,
    )?;
    images.push(normalize_pixels(&global, mean, std));
    Ok(PreparedImage { images, rows, cols })
}

fn resize_lanczos3_c_compatible(
    source: &image::RgbImage,
    width: u32,
    height: u32,
) -> Result<image::RgbImage, VisionError> {
    if source.width() == width && source.height() == height {
        return Ok(source.clone());
    }
    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    let output_width = width as usize;
    let output_height = height as usize;
    let horizontal_length = output_width
        .checked_mul(source_height)
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(|| VisionError::Image("resized image dimensions overflow".into()))?;
    let output_length = output_width
        .checked_mul(output_height)
        .and_then(|size| size.checked_mul(3))
        .ok_or_else(|| VisionError::Image("resized image dimensions overflow".into()))?;
    let mut horizontal = vec![0.0f32; horizontal_length];
    let mut pixels = vec![0u8; output_length];
    let source_pixels = source.as_raw();
    let scale_x = source_width as f32 / output_width as f32;
    let filter_x = scale_x.max(1.0);
    let radius_x = 3.0 * filter_x;
    for y in 0..source_height {
        for x in 0..output_width {
            let center = (x as f32 + 0.5) * scale_x - 0.5;
            let first = (center - radius_x).ceil() as isize;
            let last = (center + radius_x).floor() as isize;
            for channel in 0..3 {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for sample in first..=last {
                    let distance = (sample as f32 - center) / filter_x;
                    if distance.abs() >= 3.0 {
                        continue;
                    }
                    let weight = c_sinc(distance) * c_sinc(distance / 3.0);
                    let source_x = sample.clamp(0, source_width as isize - 1) as usize;
                    sum += source_pixels[(y * source_width + source_x) * 3 + channel] as f32
                        * weight;
                    weight_sum += weight;
                }
                horizontal[(y * output_width + x) * 3 + channel] = sum / weight_sum;
            }
        }
    }
    let scale_y = source_height as f32 / output_height as f32;
    let filter_y = scale_y.max(1.0);
    let radius_y = 3.0 * filter_y;
    for y in 0..output_height {
        for x in 0..output_width {
            let center = (y as f32 + 0.5) * scale_y - 0.5;
            let first = (center - radius_y).ceil() as isize;
            let last = (center + radius_y).floor() as isize;
            for channel in 0..3 {
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for sample in first..=last {
                    let distance = (sample as f32 - center) / filter_y;
                    if distance.abs() >= 3.0 {
                        continue;
                    }
                    let weight = c_sinc(distance) * c_sinc(distance / 3.0);
                    let source_y = sample.clamp(0, source_height as isize - 1) as usize;
                    sum += horizontal[(source_y * output_width + x) * 3 + channel] * weight;
                    weight_sum += weight;
                }
                let value = (sum / weight_sum).clamp(0.0, 255.0);
                pixels[(y * output_width + x) * 3 + channel] = value.round_ties_even() as u8;
            }
        }
    }
    image::RgbImage::from_raw(width, height, pixels)
        .ok_or_else(|| VisionError::Image("cannot construct resized RGB image".into()))
}

fn c_sinc(value: f32) -> f32 {
    if value.abs() < 1e-6 {
        return 1.0;
    }
    let angle = std::f32::consts::PI * value;
    angle.sin() / angle
}

fn resize_long_edge_size(height: usize, width: usize, max_edge: usize) -> (usize, usize) {
    if width >= height {
        let height = (max_edge * height / width).max(1);
        (height + height % 2, max_edge)
    } else {
        let width = (max_edge * width / height).max(1);
        (max_edge, width + width % 2)
    }
}

fn snap_to_tile_grid(height: usize, width: usize, edge: usize) -> (usize, usize) {
    if width >= height {
        let snapped_width = width.div_ceil(edge) * edge;
        let proportional_height = snapped_width * height / width;
        let snapped_height = (proportional_height.div_ceil(edge) * edge).max(edge);
        (snapped_height, snapped_width)
    } else {
        let snapped_height = height.div_ceil(edge) * edge;
        let proportional_width = snapped_height * width / height;
        let snapped_width = (proportional_width.div_ceil(edge) * edge).max(edge);
        (snapped_height, snapped_width)
    }
}

fn normalize_pixels(image: &image::RgbImage, mean: [f32; 3], std: [f32; 3]) -> PixelValues {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut values = vec![0_f32; 3 * width * height];
    for (x, y, pixel) in image.enumerate_pixels() {
        for channel in 0..3 {
            values[channel * width * height + y as usize * width + x as usize] =
                (pixel[channel] as f32 / 255.0 - mean[channel]) / std[channel];
        }
    }
    PixelValues {
        values,
        width,
        height,
    }
}

pub fn patch_embeddings(
    pixels: &PixelValues,
    patch_size: usize,
    projection: &Tensor,
) -> Result<Tensor, VisionError> {
    if pixels.width % patch_size != 0 || pixels.height % patch_size != 0 {
        return Err(VisionError::Image(
            "image dimensions must divide evenly into patches".into(),
        ));
    }
    let input = Tensor::from_vec(
        pixels.values.clone(),
        (3, pixels.height, pixels.width),
        projection.device(),
    )
    .map_err(|error| VisionError::Image(error.to_string()))?;
    let patches_h = pixels.height / patch_size;
    let patches_w = pixels.width / patch_size;
    let flattened = input
        .reshape((3, patches_h, patch_size, patches_w, patch_size))
        .and_then(|tensor| tensor.permute((1, 3, 0, 2, 4)))
        .and_then(|tensor| tensor.reshape((patches_h * patches_w, 3 * patch_size * patch_size)))
        .map_err(|error| VisionError::Image(error.to_string()))?;
    let output = projection.dims()[0];
    let projection = projection
        .reshape((output, 3 * patch_size * patch_size))
        .and_then(|tensor| tensor.t())
        .map_err(|error| VisionError::Image(error.to_string()))?;
    flattened
        .matmul(&projection)
        .map_err(|error| VisionError::Image(error.to_string()))
}
