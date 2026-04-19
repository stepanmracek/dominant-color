use dominant_color_rs::{KMeansInit, Settings, dominant_color};
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

    println!(
        "{:<30} | {:<28} | {:<28}",
        "Image", "Random Init (Mean ± SD)", "K-Means++ Init (Mean ± SD)"
    );
    println!("{:-<30}-+-{:-<28}-+-{:-<28}", "", "", "");

    for path in files {
        let file_name = path.file_name().unwrap().to_string_lossy().to_string();
        let img = image::open(&path).expect("Failed to open image");

        let mut times_random = Vec::with_capacity(iterations);
        let mut times_kmeanspp = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            // Random
            let settings_random = Settings {
                init: KMeansInit::Random,
                ..Default::default()
            };
            let start_random = Instant::now();
            let _color_random = dominant_color(&img, &settings_random);
            times_random.push(start_random.elapsed().as_secs_f64() * 1000.0); // Store in milliseconds

            // K-Means++
            let settings_kmeanspp = Settings {
                init: KMeansInit::KMeansPlusPlus,
                ..Default::default()
            };
            let start_kmeanspp = Instant::now();
            let _color_kmeanspp = dominant_color(&img, &settings_kmeanspp);
            times_kmeanspp.push(start_kmeanspp.elapsed().as_secs_f64() * 1000.0); // Store in milliseconds
        }

        let (mean_random, std_dev_random) = calculate_mean_and_std_dev(&times_random);
        let (mean_kmeanspp, std_dev_kmeanspp) = calculate_mean_and_std_dev(&times_kmeanspp);

        grand_total_mean_random += mean_random;
        grand_total_mean_kmeanspp += mean_kmeanspp;

        let random_str = format!("{:.2}ms ± {:.2}ms", mean_random, std_dev_random);
        let kmeanspp_str = format!("{:.2}ms ± {:.2}ms", mean_kmeanspp, std_dev_kmeanspp);

        println!(
            "{:<30} | {:<28} | {:<28}",
            file_name, random_str, kmeanspp_str
        );
    }

    println!("{:-<30}-+-{:-<28}-+-{:-<28}", "", "", "");

    let total_random_str = format!("{:.2}ms", grand_total_mean_random);
    let total_kmeanspp_str = format!("{:.2}ms", grand_total_mean_kmeanspp);

    println!(
        "{:<30} | {:<28} | {:<28}",
        "TOTAL (Sum of Means)", total_random_str, total_kmeanspp_str
    );
}
