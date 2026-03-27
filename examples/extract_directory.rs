use dominant_color::{Settings, dominant_color};
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example extract_directory <directory_path>");
        return;
    }

    let dir_path = Path::new(&args[1]);
    if !dir_path.is_dir() {
        eprintln!("Error: {} is not a directory.", args[1]);
        return;
    }

    let settings = Settings::default();

    println!("Processing images in: {}", dir_path.display());
    println!("{:<40} | {:<20}", "File Name", "Dominant RGB (0-255)");
    println!("{:-<40}-+-{:-<20}", "", "");

    let mut entries: Vec<_> = fs::read_dir(dir_path)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .collect();

    // Sort entries by path for consistent output
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let file_name = path.file_name().unwrap().to_string_lossy();
        match image::open(&path) {
            Ok(img) => {
                if let Some(color) = dominant_color(&img, &settings) {
                    let r = (color[0] * 255.0).round() as u8;
                    let g = (color[1] * 255.0).round() as u8;
                    let b = (color[2] * 255.0).round() as u8;
                    println!("{:<40} | [{}, {}, {}]", file_name, r, g, b);
                } else {
                    println!("{:<40} | No dominant color found", file_name);
                }
            }
            Err(_) => {
                // If it's not an image, we just skip it for this example
            }
        }
    }
}
