use dominant_color_rs::{Settings, dominant_colors};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example extract_dominant <path_to_image>");
        return;
    }

    let img_path = &args[1];
    let img = match image::open(img_path) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error opening image {}: {}", img_path, e);
            return;
        }
    };

    let colors = dominant_colors(&img, &Settings::default());
    if !colors.is_empty() {
        println!("Dominant colors found (sorted by saturation):");
        for (i, color) in colors.iter().enumerate() {
            println!("Color #{}", i + 1);
            println!(
                "  RGB (0.0 - 1.0): [{:.3}, {:.3}, {:.3}]",
                color[0], color[1], color[2]
            );
            println!(
                "  RGB (0 - 255):   [{}, {}, {}]",
                (color[0] * 255.0).round() as u8,
                (color[1] * 255.0).round() as u8,
                (color[2] * 255.0).round() as u8
            );
        }
    } else {
        println!("Could not determine dominant colors for the image.");
    }
}
