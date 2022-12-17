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

struct FseParams {
    etl: usize,
    t1: f64,
    t2: f64,
    esp: f64,
    refocus_angle : f64,
    cpmg_phase : f64,
    debug_print : bool
}

fn fse_signal(params: FseParams) {
    let etl = params.etl;

    let mut epg = epg::vec::EPGVecRepresentation::new(etl / 2 + 1);

    let mut signal: Vec<Complex64> = Vec::with_capacity(etl + 1);

    let x180 = epg::common::gen_rotation_matrix(params.refocus_angle, params.cpmg_phase);

    // dt is the spacing of our events, unsed for dephasing/relaxation
    let dt = params.esp / 2.0;
    let et1d = Complex::from((-dt / params.t1).exp());
    let et2d = Complex::from((-dt / params.t2).exp());

    epg.excite();

    for _ in 0..etl {
        epg.grelax(et1d, et2d, 1);
        epg.rotate(&x180);
        epg.grelax(et1d, et2d, 1);
        if params.debug_print {
            println!("{}", epg);
        }

        signal.push(epg.read());
    }
    println!("Signal: {:?}", signal);
    println!("{:}", epg::vec::to_mxy(&epg));
    println!("{:}", epg::vec::to_mz(&epg));




}

fn main() {
    println!("Hello, world!");

    let params = FseParams {
        etl: 10,
        esp: 0.01,
        t1: 0.5,
        t2: 0.1,
        refocus_angle: PI/2.0,
        cpmg_phase: PI/2.0,
        debug_print: false
    };
    fse_signal(params);

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui app",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}
