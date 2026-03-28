use image::{DynamicImage, Pixel};
use rand::seq::IndexedRandom;
use std::ops::RangeInclusive;

pub struct KMeansResult<const DIMS: usize> {
    pub centroids: Vec<[f32; DIMS]>,
    pub clusters: Vec<Vec<usize>>,
}

pub fn silhouette_score<const DIMS: usize, F>(
    data: &[[f32; DIMS]],
    result: &KMeansResult<DIMS>,
    distance: F,
) -> f32
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let mut s = vec![0.0; data.len()];

    for ((cluster_index, cluster), centroid) in
        std::iter::zip(result.clusters.iter().enumerate(), result.centroids.iter())
    {
        for &point_index in cluster {
            let a = distance(&data[point_index], centroid);
            let b = result
                .centroids
                .iter()
                .enumerate()
                .filter(|(other_cluster_index, _)| *other_cluster_index != cluster_index)
                .map(|(_, other_cluster_centroid)| {
                    distance(&data[point_index], other_cluster_centroid)
                })
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            if a < b {
                s[point_index] = 1.0 - (a / b);
            } else if a > b {
                s[point_index] = (b / a) - 1.0;
            }
        }
    }

    s.iter().sum::<f32>() / data.len() as f32
}

pub fn eucl_distance_squared(first: &[f32], second: &[f32]) -> f32 {
    std::iter::zip(first, second)
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

pub fn eucl_distance(first: &[f32], second: &[f32]) -> f32 {
    eucl_distance_squared(first, second).sqrt()
}

fn calculate_centroids<const DIMS: usize>(
    data: &[[f32; DIMS]],
    clusters: &[Vec<usize>],
    old_centroids: &[[f32; DIMS]],
) -> Vec<[f32; DIMS]> {
    let mut ans = vec![];
    for (cluster, old_centroid) in std::iter::zip(clusters, old_centroids) {
        if cluster.is_empty() {
            ans.push(*old_centroid);
            continue;
        }

        let mut sum = cluster
            .iter()
            .map(|&index| data[index])
            .fold([0.0; DIMS], |mut acc, x| {
                for i in 0..DIMS {
                    acc[i] += x[i];
                }
                acc
            });

        let cluster_size = cluster.len() as f32;
        for v in sum.iter_mut() {
            *v /= cluster_size;
        }

        ans.push(sum);
    }

    ans
}

fn array_eq(first: &[f32], second: &[f32], eps: f32) -> bool {
    std::iter::zip(first, second).all(|(a, b)| (a - b).abs() <= eps)
}

fn centroids_eq<const DIMS: usize>(
    first: &Vec<[f32; DIMS]>,
    second: &Vec<[f32; DIMS]>,
    eps: f32,
) -> bool {
    std::iter::zip(first, second).all(|(a, b)| array_eq(a, b, eps))
}

pub fn kmeans<const DIMS: usize, F>(
    data: &[[f32; DIMS]],
    k: usize,
    distance: F,
    max_iters: usize,
    eps: f32,
) -> KMeansResult<DIMS>
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let mut rng = rand::rng();
    let mut centroids = data.sample(&mut rng, k).cloned().collect::<Vec<_>>();

    let mut clusters: Vec<Vec<usize>> = vec![vec![]; k];
    for _i in 0..max_iters {
        for c in clusters.iter_mut() {
            c.clear();
        }

        // Assign each point to the "closest" centroid
        for (index, point) in data.iter().enumerate() {
            let closest_centroid = centroids
                .iter()
                .map(|centroid| distance(centroid, point))
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .expect("Can't assign point to the closest centroid");
            clusters[closest_centroid].push(index);
        }

        let new_centroids = calculate_centroids(data, &clusters, &centroids);
        if centroids_eq(&new_centroids, &centroids, eps) {
            break;
        }
        centroids = new_centroids;
    }

    KMeansResult {
        centroids,
        clusters,
    }
}

pub fn saturation(point: &[f32; 3]) -> f32 {
    let max = point
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&0.0);
    let min = point
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(&0.0);

    max - min
}

pub struct Settings {
    pub img_size: u32,
    pub clusters: RangeInclusive<usize>,
    pub max_iters: usize,
    pub eps: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            img_size: 72,
            clusters: 2..=6,
            max_iters: 100,
            eps: 1e-6,
        }
    }
}

fn dominant_colors_private(img: &DynamicImage, settings: &Settings) -> Vec<([f32; 3], f32)> {
    let resized = image::imageops::resize(
        img,
        settings.img_size,
        settings.img_size,
        image::imageops::FilterType::Triangle,
    );

    let pixels: Vec<_> = resized
        .pixels()
        .map(|pixel| {
            let rgb = pixel.to_rgb();
            rgb.0.map(|v| (v as f32) / 255.0)
        })
        .collect();

    // take the kmeans_result maximizing silhouette_score
    let kmeans_result = settings
        .clusters
        .clone()
        .map(|k| {
            kmeans(
                &pixels,
                k,
                eucl_distance_squared,
                settings.max_iters,
                settings.eps,
            )
        })
        .map(|kmeans_result| {
            (
                silhouette_score(&pixels, &kmeans_result, eucl_distance),
                kmeans_result,
            )
        })
        .max_by(|(score1, _), (score2, _)| {
            score1
                .partial_cmp(score2)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, kmeans_result)| kmeans_result);

    match kmeans_result {
        Some(kmeans_result) => std::iter::zip(
            kmeans_result.centroids.iter(),
            kmeans_result.clusters.iter(),
        )
        .filter(|(_centroid, cluster)| !cluster.is_empty())
        .map(|(centroid, _cluster)| (*centroid, saturation(centroid)))
        .collect(),
        None => vec![],
    }
}

/// Calculates the dominant colors of an image.
///
/// Returns the vector of RGB colors represented as `[f32; 3]`
/// where each component is in the range `[0.0, 1.0]`
pub fn dominant_colors(img: &DynamicImage, settings: &Settings) -> Vec<[f32; 3]> {
    let mut centroids_and_saturations = dominant_colors_private(img, settings);

    // sort clusters by their saturation value (descending)
    centroids_and_saturations.sort_by(|(_, sat1), (_, sat2)| {
        sat2.partial_cmp(sat1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // return just the centroid colors
    centroids_and_saturations
        .into_iter()
        .map(|(centroid, _)| centroid)
        .collect()
}

/// Calculates the dominant color of an image.
///
/// Returns the RGB color as `[f32; 3]` where each component is in the range `[0.0, 1.0]`
pub fn dominant_color(img: &DynamicImage, settings: &Settings) -> Option<[f32; 3]> {
    // dominant color is centroid having the highest saturation
    dominant_colors_private(img, settings)
        .into_iter()
        .max_by(|(_, sat1), (_, sat2)| sat1.partial_cmp(sat2).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(centroid, _)| centroid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_chroma() {
        let dark_red = [4.0 / 255.0, 2.0 / 255.0, 2.0 / 255.0];
        let vivid_red = [187.0 / 255.0, 78.0 / 255.0, 69.0 / 255.0];

        let sat_dark = saturation(&dark_red);
        let sat_vivid = saturation(&vivid_red);
        assert!(sat_vivid > sat_dark);
    }

    #[test]
    fn test() {
        let entries = std::fs::read_dir("testimg").unwrap();
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_file() {
                let img = image::open(path).unwrap();
                dominant_color(&img, &Settings::default());
            }
        }
    }
}
