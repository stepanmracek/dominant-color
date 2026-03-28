# dominant-color

A Rust library to extract the most dominant color from an image using K-Means clustering and silhouette scoring.

## Overview

This library calculates the dominant color of an image by:
1. Resizing the image for performance.
2. Running K-Means clustering across a range of possible cluster counts ($K$).
3. Using the silhouette score to find the optimal number of clusters.
4. Selecting the cluster centroid with the highest saturation as the dominant color.

## Usage

### Simple Example

```rust
use dominant_color_rs::{Settings, dominant_color};

fn main() {
    let img = image::open("path/to/image.jpg").expect("Failed to open image");
    
    // Use default settings
    let settings = Settings::default();
    
    if let Some(color) = dominant_color(&img, &settings) {
        println!("Dominant color (RGB 0.0-1.0): {:?}", color);
        println!(
            "Dominant color (RGB 0-255): [{}, {}, {}]",
            (color[0] * 255.0).round() as u8,
            (color[1] * 255.0).round() as u8,
            (color[2] * 255.0).round() as u8
        );
    }
}
```

### Custom Settings

You can tune the performance and accuracy by modifying the `Settings` struct:

```rust
use dominant_color_rs::Settings;

let settings = Settings {
    img_size: 128,      // Internal resize dimension (default: 72)
    clusters: 2..=10,   // Range of K values to test (default: 2..=6)
    max_iters: 200,     // Max iterations for K-Means (default: 100)
    eps: 1e-7,          // Convergence threshold (default: 1e-6)
};
```

## Examples

The library comes with several built-in examples. You can run them using `cargo run --example`:

- `extract_dominant`: Extract dominant color from a single image.
  ```bash
  cargo run --example extract_dominant path/to/image.jpg
  ```
- `extract_directory`: Process all images in a directory.
  ```bash
  cargo run --example extract_directory ./testimg
  ```

## License

MIT
