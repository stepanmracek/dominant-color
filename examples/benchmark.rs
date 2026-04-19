use dominant_color_rs::{ColorSpace, KMeansInit, Settings, dominant_color};
use std::time::Instant;

fn calculate_mean_and_std_dev(times: &[f64]) -> (f64, f64) {
    let len = times.len() as f64;
    if len == 0.0 {
        return (0.0, 0.0);
    }
    let mean = times.iter().sum::<f64>() / len;
    let variance = times
        .iter()
        .map(|value| {
            let diff = mean - *value;
            diff * diff
        })
        .sum::<f64>()
        / len;
    (mean, variance.sqrt())
}

fn main() {
    let entries = std::fs::read_dir("testimg").expect("Failed to read testimg directory");
    let mut files = Vec::new();
    for entry in entries {
        let path = entry.unwrap().path();
        if path.is_file() {
            files.push(path);
        }
    }
    files.sort();

    let iterations = 100;

    let mut grand_total_mean_random = 0.0;
    let mut grand_total_mean_kmeanspp = 0.0;
    let mut grand_total_mean_rgb = 0.0;
    let mut grand_total_mean_oklab = 0.0;

    println!(
        "{:<30} | {:<22} | {:<22} | {:<22} | {:<22}",
        "Image", "Random Init", "K-Means++ Init", "RGB Space", "Oklab Space"
    );
    println!(
        "{:-<30}-+-{:-<22}-+-{:-<22}-+-{:-<22}-+-{:-<22}",
        "", "", "", "", ""
    );

    for path in files {
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        let img = image::open(&path).expect("Failed to open image");

        let mut times_random = Vec::with_capacity(iterations);
        let mut times_kmeanspp = Vec::with_capacity(iterations);
        let mut times_rgb = Vec::with_capacity(iterations);
        let mut times_oklab = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            // Random
            let settings_random = Settings {
                init: KMeansInit::Random,
                ..Default::default()
            };
            let start_random = Instant::now();
            let _color_random = dominant_color(&img, &settings_random);
            times_random.push(start_random.elapsed().as_secs_f64() * 1000.0);

            // K-Means++
            let settings_kmeanspp = Settings {
                init: KMeansInit::KMeansPlusPlus,
                ..Default::default()
            };
            let start_kmeanspp = Instant::now();
            let _color_kmeanspp = dominant_color(&img, &settings_kmeanspp);
            times_kmeanspp.push(start_kmeanspp.elapsed().as_secs_f64() * 1000.0);

            // RGB
            let settings_rgb = Settings {
                color_space: ColorSpace::Rgb,
                ..Default::default()
            };
            let start_rgb = Instant::now();
            let _color_rgb = dominant_color(&img, &settings_rgb);
            times_rgb.push(start_rgb.elapsed().as_secs_f64() * 1000.0);

            // Oklab
            let settings_oklab = Settings {
                color_space: ColorSpace::Oklab,
                ..Default::default()
            };
            let start_oklab = Instant::now();
            let _color_oklab = dominant_color(&img, &settings_oklab);
            times_oklab.push(start_oklab.elapsed().as_secs_f64() * 1000.0);
        }

        let (mean_random, std_dev_random) = calculate_mean_and_std_dev(&times_random);
        let (mean_kmeanspp, std_dev_kmeanspp) = calculate_mean_and_std_dev(&times_kmeanspp);
        let (mean_rgb, std_dev_rgb) = calculate_mean_and_std_dev(&times_rgb);
        let (mean_oklab, std_dev_oklab) = calculate_mean_and_std_dev(&times_oklab);

        grand_total_mean_random += mean_random;
        grand_total_mean_kmeanspp += mean_kmeanspp;
        grand_total_mean_rgb += mean_rgb;
        grand_total_mean_oklab += mean_oklab;

        let random_str = format!("{:.1}ms ± {:.1}", mean_random, std_dev_random);
        let kmeanspp_str = format!("{:.1}ms ± {:.1}", mean_kmeanspp, std_dev_kmeanspp);
        let rgb_str = format!("{:.1}ms ± {:.1}", mean_rgb, std_dev_rgb);
        let oklab_str = format!("{:.1}ms ± {:.1}", mean_oklab, std_dev_oklab);

        println!(
            "{:<30} | {:<22} | {:<22} | {:<22} | {:<22}",
            file_name, random_str, kmeanspp_str, rgb_str, oklab_str
        );
    }

    println!(
        "{:-<30}-+-{:-<22}-+-{:-<22}-+-{:-<22}-+-{:-<22}",
        "", "", "", "", ""
    );

    let total_random_str = format!("{:.1}ms", grand_total_mean_random);
    let total_kmeanspp_str = format!("{:.1}ms", grand_total_mean_kmeanspp);
    let total_rgb_str = format!("{:.1}ms", grand_total_mean_rgb);
    let total_oklab_str = format!("{:.1}ms", grand_total_mean_oklab);

    println!(
        "{:<30} | {:<22} | {:<22} | {:<22} | {:<22}",
        "TOTAL (Sum of Means)",
        total_random_str,
        total_kmeanspp_str,
        total_rgb_str,
        total_oklab_str
    );

    if grand_total_mean_rgb > 0.0 {
        println!(
            "\nOklab is {:.2}x slower than RGB",
            grand_total_mean_oklab / grand_total_mean_rgb
        );
    }
}
