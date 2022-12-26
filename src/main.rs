use num_complex::{Complex, Complex64};

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
mod sequences;
mod types;

use types::EPG;

use crate::sequences::fse;

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

    let params = sequences::fse::FseParams {
        etl: 20,
        esp: 0.01,
        t1: 0.58,
        t2: 0.11,
        refocus_angle: PI / 2.0,
        cpmg_phase: PI / 2.0,
        debug_print: false,
    };
    let res = sequences::fse::simulate(params);

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui app",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

