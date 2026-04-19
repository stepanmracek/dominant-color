use dominant_color_rs::{Settings, dominant_colors};
use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Dominant Color Viewer"),
        ..Default::default()
    };
    eframe::run_native(
        "Dominant Color Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

struct App {
    images: Vec<PathBuf>,
    selected_image_idx: usize,
    settings: Settings,
    min_clusters: usize,
    max_clusters: usize,
    dominant_colors: Vec<[f32; 3]>,
    texture: Option<egui::TextureHandle>,
    error_message: Option<String>,
    computation_time: Option<std::time::Duration>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut images = Vec::new();
        if let Ok(entries) = std::fs::read_dir("testimg") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if matches!(ext, "jpg" | "jpeg" | "png" | "webp") {
                        images.push(path);
                    }
                }
            }
        }
        images.sort();

        let settings = Settings::default();
        let min_clusters = *settings.clusters.start();
        let max_clusters = *settings.clusters.end();

        let mut app = Self {
            images,
            selected_image_idx: 0,
            settings,
            min_clusters,
            max_clusters,
            dominant_colors: Vec::new(),
            texture: None,
            error_message: None,
            computation_time: None,
        };

        app.refresh_colors(_cc.egui_ctx.clone());
        app
    }

    fn refresh_colors(&mut self, ctx: egui::Context) {
        if self.images.is_empty() {
            self.error_message = Some("No images found in testimg/ directory.".to_string());
            return;
        }

        let path = &self.images[self.selected_image_idx];
        match image::open(path) {
            Ok(img) => {
                self.settings.clusters = self.min_clusters..=self.max_clusters;
                let start_time = std::time::Instant::now();
                self.dominant_colors = dominant_colors(&img, &self.settings);
                self.computation_time = Some(start_time.elapsed());
                self.error_message = None;

                // Load texture
                let size = [img.width() as _, img.height() as _];
                let image_buffer = img.to_rgba8();
                let pixels = image_buffer.as_flat_samples();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
                self.texture = Some(ctx.load_texture(
                    path.file_name().unwrap().to_string_lossy(),
                    color_image,
                    Default::default(),
                ));
            }
            Err(e) => {
                self.error_message = Some(format!("Error opening image: {}", e));
                self.dominant_colors.clear();
                self.texture = None;
                self.computation_time = None;
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("settings_panel").show(ctx, |ui| {
            ui.heading("Image Selection");
            ui.add_space(8.0);
            if !self.images.is_empty() {
                let selected_text = self.images[self.selected_image_idx]
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .to_string();

                egui::ComboBox::from_label("Select Image")
                    .selected_text(selected_text)
                    .show_ui(ui, |ui| {
                        for i in 0..self.images.len() {
                            let label = self.images[i]
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .to_string();
                            if ui
                                .selectable_value(&mut self.selected_image_idx, i, label)
                                .changed()
                            {
                                self.refresh_colors(ctx.clone());
                            }
                        }
                    });
            } else {
                ui.label("No images found in testimg/");
            }

            ui.add_space(16.0);

            ui.heading("Settings");
            ui.add_space(8.0);

            if ui
                .add(egui::Slider::new(&mut self.settings.img_size, 8..=256).text("Resize Size"))
                .changed()
            {
                self.refresh_colors(ctx.clone());
            }

            if ui
                .add(egui::Slider::new(&mut self.min_clusters, 1..=10).text("Min Clusters"))
                .changed()
            {
                if self.min_clusters > self.max_clusters {
                    self.max_clusters = self.min_clusters;
                }
                self.refresh_colors(ctx.clone());
            }

            if ui
                .add(egui::Slider::new(&mut self.max_clusters, 1..=20).text("Max Clusters"))
                .changed()
            {
                if self.max_clusters < self.min_clusters {
                    self.min_clusters = self.max_clusters;
                }
                self.refresh_colors(ctx.clone());
            }

            if ui
                .add(egui::Slider::new(&mut self.settings.max_iters, 10..=500).text("Max Iters"))
                .changed()
            {
                self.refresh_colors(ctx.clone());
            }

            if ui
                .add(
                    egui::Slider::new(&mut self.settings.eps, 1e-7..=0.1)
                        .text("Epsilon")
                        .logarithmic(true),
                )
                .changed()
            {
                self.refresh_colors(ctx.clone());
            }

            if let Some(error) = &self.error_message {
                ui.add_space(16.0);
                ui.colored_label(egui::Color32::RED, error);
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Dominant Colors");
                if let Some(duration) = self.computation_time {
                    ui.label(format!("(Computed in {:.2?})", duration));
                }
            });
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                for color in &self.dominant_colors {
                    let r = (color[0] * 255.0) as u8;
                    let g = (color[1] * 255.0) as u8;
                    let b = (color[2] * 255.0) as u8;
                    let (rect, _response) =
                        ui.allocate_at_least(egui::vec2(50.0, 50.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 4.0, egui::Color32::from_rgb(r, g, b));
                }
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(16.0);

            if let Some(texture) = &self.texture {
                ui.add(egui::Image::new(texture).shrink_to_fit());
            }
        });
    }
}
