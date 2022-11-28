use num_complex::{Complex, Complex64};
use rand_distr::{Distribution, StandardNormal, UnitCircle};

use eframe::egui;

use egui::plot::{Line, Plot, PlotPoints};

use std::fmt;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use std::f64::consts::PI;

mod epg;
mod types;

use types::EPG;

#[macro_use]
extern crate approx;

struct MyApp {
    index: i32,
}

impl Default for MyApp {
    fn default() -> Self {
        Self { index: 0 }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");
            ui.add(egui::Slider::new(&mut self.index, 0..=12).text("index"));

            let sin: PlotPoints = (0..1000)
                .map(|i| {
                    let x = (i * self.index) as f64 * 0.01;
                    [x, x.sin()]
                })
                .collect();
            let line = Line::new(sin);
            Plot::new("my_plot")
                .view_aspect(2.0)
                .show(ui, |plot_ui| plot_ui.line(line));
        });
    }
}

fn main() {
    println!("Hello, world!");

    let x180 = epg::common::gen_rotation_matrix(PI, 0.0);
    println!("{x180:1.3}");

    {
        let mut epg = epg::vec::EPGVecRepresentation::new(5);
        println!("{}", epg);

        epg.excite();

        let t1 = 1.0;
        let t2 = 0.05;
        let dt = 1e-3;

        epg.grelax(dt, t1, t2, 1);
        println!("1\n{}", epg);

        epg.grelax(dt, t1, t2, 1);
        println!("2\n{}", epg);

        epg.rotate(&x180);

        epg.grelax(dt, t1, t2, 1);
        println!("3\n{}", epg);

        epg.grelax(dt, t1, t2, 1);
        println!("4\n{}", epg);
    }

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui app",
        options,
        Box::new(|_cc| Box:: new(MyApp::default())),
    );
}
