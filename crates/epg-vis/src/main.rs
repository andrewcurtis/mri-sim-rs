

use eframe::egui;

use egui::plot::{Plot, Line, PlotPoints};


struct MyApp {
    name: String,
    age: u32,
    frequency: f64,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            frequency: 1.0,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");
            ui.horizontal(|ui| {
                ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Click each year").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{}', age {}", self.name, self.age));

            ui.add(egui::Slider::new(&mut self.frequency, -10.0..=12.0).text("frequency"));

            let sin: PlotPoints = (0..1000).map(|i| {
                let x = i as f64 * 0.01; 
                [x, (self.frequency * x).sin()]
            }).collect();
            let line = Line::new(sin);
            Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| plot_ui.line(line));
        });

    }
}


fn main() {
    println!("Hello, world!");
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui app", 
        options, 
        Box::new(|_cc| Box:: new(MyApp::default())),
    );
}
