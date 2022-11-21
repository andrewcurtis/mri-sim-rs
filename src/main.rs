use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use rand_distr::{Distribution, StandardNormal, UnitCircle};

use eframe::egui;

use egui::plot::{Line, Plot, PlotPoints};

use std::collections::VecDeque;
use std::fmt;
use std::rc::Rc;
use std::sync::Mutex;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use std::f64::consts::PI;

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

    let r1 = gen_rotation_matrix(3.1415 / 2.0, 0.0);
    println!("{}", r1);
    let r2 = gen_rotation_matrix(3.1415 / 4.0, 0.0);
    println!("{}", r2);
    let x180 = gen_rotation_matrix(3.1415, 0.0);
    println!("{x180:1.3}");

    {
        let mut epg = EPGVecRepresentation::new(5);
        println!("{}", epg);

        
        epg.excite();

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        println!("{}", epg);

        let t1 = 1.0;
        let t2= 0.05;
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

    //std::thread::spawn( || work_with_arrays() );
    // let options = eframe::NativeOptions::default();
    // eframe::run_native(
    //     "My egui app",
    //     options,
    //     Box::new(|_cc| Box:: new(MyApp::default())),
    // );
}

fn eiphi(phi: f64) -> Complex64 {
    (Complex::i() * phi).exp()
}

#[derive(Debug, PartialEq)]
struct EPGVecRepresentation {
    length: usize,
    f_p: VecDeque<Complex64>,
    f_n: VecDeque<Complex64>,
    z: VecDeque<Complex64>,
}

impl fmt::Display for EPGVecRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for ix in 0..self.length {
            write!(f, "f+ {:.3}\tf- {:.3}\tz {:.3}\n", self.f_p[ix], self.f_n[ix], self.z[ix])?;
        }
        Ok(())
    }
}

impl EPGVecRepresentation {
    fn new(n_states: usize) -> Self {
        let length = n_states;
        let mut f_p = VecDeque::with_capacity(length);
        let mut f_n = VecDeque::with_capacity(length);
        let mut z = VecDeque::with_capacity(length);

        f_p.push_back(Complex64::from(0.0));
        f_n.push_back(Complex64::from(0.0));
        z.push_back(Complex64::from(1.0));

        for _ in 1..length {
            f_p.push_back(Complex64::from(0.0));
            f_n.push_back(Complex64::from(0.0));
            z.push_back(Complex64::from(0.0));
        }

        Self {
            length,
            f_p,
            f_n,
            z,
        }
    }

    pub fn read(&self) -> Complex64 {
        self.f_p[0]
    }

    pub fn excite(&mut self) {
        let rot = gen_rotation_matrix(PI / 2.0, 0.0);
        Self::rotate(self, &rot);
    }

    pub fn rotate(&mut self, rmat: &Array<Complex64, Ix2>) {
        rf_rotation(self, rmat);
    }

    pub fn grelax(&mut self, dt: f64, t1: f64, t2: f64, ntwists: i64) {
        gradient_shift(self, ntwists);
        relaxation(self, dt, t1, t2);
    }

    pub fn delay(self: &mut Self, dt: f64, t1: f64, t2: f64) {
        relaxation(self, dt, t1, t2);
    }
}

impl Default for EPGVecRepresentation {
    fn default() -> Self {
        Self::new(3)
    }
}

fn rf_rotation(epg: &mut EPGVecRepresentation, rmat: &Array<Complex64, Ix2>) {
    for ix in 0..epg.length {
        // we don't want this copy here but we can't dot() with references.
        // option1 : rewrite own dot (not hard)
        // option2 : use arr2 as underlying structure.
        let sl = arr1(&[epg.f_p[ix], epg.f_n[ix], epg.z[ix]]);
        let rot = rmat.dot(&sl);
        epg.f_p[ix] = rot[0];
        epg.f_n[ix] = rot[1];
        epg.z[ix] = rot[2];
    }
}

fn relaxation(epg: &mut EPGVecRepresentation, dt: f64, t1: f64, t2: f64) {
    let t1d = Complex::from((-dt / t1).exp());
    let t2d = Complex::from((-dt / t2).exp());

    for x in epg.f_n.iter_mut() {
        *x = *x * t2d
    }

    for x in epg.f_p.iter_mut() {
        *x = *x * t2d
    }

    for (ix, z) in epg.z.iter_mut().enumerate() {
        if ix == 0 {
            *z = (1.0 - t1d) + (*z * t1d)
        } else {
            *z = *z * t1d
        }
    }
}

fn gen_rotation_matrix(alpha: f64, phi: f64) -> Array<Complex64, Ix2> {
    // make coefficients
    let sa = Complex::from(alpha.sin());
    let ca = Complex::from(alpha.cos());

    let j = Complex::i();

    let ca2 = Complex::from((alpha / 2.0).cos());
    let sa2 = Complex::from((alpha / 2.0).sin());

    array![
        [
            ca2 * ca2,
            sa2 * sa2 * eiphi(2.0 * phi),
            -j * eiphi(phi) * sa
        ],
        [
            eiphi(-2.0 * phi) * sa2 * sa2,
            ca2 * ca2,
            j * eiphi(-phi) * sa
        ],
        [-j / 2.0 * eiphi(-phi) * sa, j / 2.0 * eiphi(phi) * sa, ca]
    ]
}

fn gradient_shift(epg: &mut EPGVecRepresentation, ntwists: i64) {
    // Shift states.
    // nshfits represents the number of 2pi dephasing steps to shift by
    // F0 is special, since f_p [0] is f0, and f_n[0] is f0*(conj)

    match ntwists {
        n if n == 0 => return,
        n if n > 0 => {
            // f_p becomes more positive. f_m becomes less negative
            // do shift
            let _ = epg.f_n.pop_front().unwrap(); // f0c discard
            let f1 = epg.f_n.pop_front().unwrap();

            let f1c = f1.conj();
            epg.f_p.push_front(f1);
            epg.f_n.push_front(f1c);

            // we've pop'd 2 from f_n and pushed 1. So length is one less.
            // add zero to the end.
            epg.f_n.push_back(Complex64::from(0.0));

            // pop end of f_p to maintain length
            let _ = epg.f_p.pop_back().unwrap();

            // recurse
            gradient_shift(epg, n - 1);
        }
        n if n < 0 => {
            // f_p becomes less positive. f_m becomes more negative
            let _ = epg.f_n.pop_front().unwrap(); // f0c discard
                                                  // [ f0, f1, f2 ...] --> [f1, f2, ...]

            let f0 = epg.f_p.pop_front().unwrap();

            let f1 = epg.f_p.front().unwrap();

            let f1c = f1.conj();

            epg.f_n.push_front(f0);
            epg.f_n.push_front(f1c);

            // we've pop'd 2 from f_n and pushed 1. So length is one less.
            // add zero to the end.
            epg.f_p.push_back(Complex64::from(0.0));

            // pop end of f_n to maintain length
            let _ = epg.f_n.pop_back().unwrap();

            // recurse
            gradient_shift(epg, n + 1)
        }
        _ => unreachable!("Should never happen."),
    }
}

// How do we implement the algorithm?
// Gradients : moving f states. Depends on matirx vs vector storage.
// is a dequeue better than vector ?
// For RF parts:
// Option 1: 3 vectors, zip and multiple as we go
// Dense array: transpose and loop vs loop and lookup ?
// Relaxation : T2 affects all F states and F*.
//   T1 decay all Z states
// . T1 recovery just Z0 state
// recording signal, same.

fn work_with_arrays() {
    {
        let arr = array![
            [Complex64::new(1., 2.), Complex64::new(3., 4.)],
            [Complex64::new(5., 6.), Complex64::new(7., 8.)],
            [Complex64::new(9., 10.), Complex64::new(11., 12.)],
        ];
        let Complex { re, im } = arr.view().split_complex();
        assert_eq!(re, array![[1., 3.], [5., 7.], [9., 11.]]);
        assert_eq!(im, array![[2., 4.], [6., 8.], [10., 12.]]);
    }

    // rand generators
    {
        let v: [f64; 2] = UnitCircle.sample(&mut rand::thread_rng());
        println!("{:?} is from the unit circle.", v);

        let p: f64 = StandardNormal.sample(&mut rand::thread_rng());
        println!("{:?} is normally distributed!", p);
        println!("\n\n");
    }

    {
        // now sample lots
        let mut rng = rand::thread_rng();

        let v: Vec<f32> = StandardNormal.sample_iter(&mut rng).take(16).collect();
        println!("{:?} is normally distributed!", v);
        println!("\n\n");
    }

    {
        // now sample lots
        let mut rng = rand::thread_rng();

        let v: Vec<Complex<f32>> = UnitCircle
            .sample_iter(&mut rng)
            .take(16)
            .map(|x| Complex::new(x[0], x[1]))
            .collect();
        println!("{:?} is normally distributed!", v);

        for (ix, x) in v.iter().enumerate() {
            println!("{ix}: {x}");
        }

        let mags = v.iter().map(|x| x.norm()).collect::<Vec<_>>();
        println!("magnitudes of v : {:?}", mags);
        println!("\n\n");
    }
    // nd array tests
    {
        let a = ndarray::arr3(&[
            [
                [1, 2, 3], // -- 2 rows  \_
                [4, 5, 6],
            ], // --         /
            [
                [7, 8, 9], //            \_ 2 submatrices
                [10, 11, 12],
            ],
        ]); //            /
            //  3 columns ..../.../.../

        assert_eq!(a.shape(), &[2, 2, 3]);

        let owned1 = array![1, 2];
        let owned2 = array![3, 4];
        let view1 = ArrayView1::from(&[5, 6]);
        let view2 = ArrayView1::from(&[7, 8]);
        let mut mutable = array![9, 10];

        let sum1 = &view1 + &view2; // Allocates a new array. Note the explicit `&`.
                                    // let sum2 = view1 + &view2; // This doesn't work because `view1` is not an owned array.
        let sum3 = owned1 + view1; // Consumes `owned1`, updates it, and returns it.
        let sum4 = owned2 + &view2; // Consumes `owned2`, updates it, and returns it.
        mutable += &view2; // Updates `mutable` in-place.
        println!("results: {:?}, {:?}, {:?}, {:?}", sum1, sum3, sum4, mutable);
        println!("\n\n");
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    #[test]
    fn test_180_rotation() {
        let mut epg = EPGVecRepresentation::new(1);

        let r2 = gen_rotation_matrix(PI, 0.0);

        rf_rotation(&mut epg, &r2);
        println!("{:?}", epg);

        relaxation(&mut epg, 100e-3, 1.0, 0.1);
    }

    #[test]
    fn test_shift_symmetric() {
        let mut epg = EPGVecRepresentation::new(3);
        let mut epg2 = EPGVecRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg2, &r2);

        gradient_shift(&mut epg, 2);
        println!("{:?}", epg);
        gradient_shift(&mut epg, -2);
        println!("{:?}", epg);

        // assert_eq!(&epg.f_p, &epg2.f_p);
        // assert_eq!(&epg.f_n, &epg2.f_n);
        // assert_eq!(&epg.z, &epg2.z);
        assert_eq!(&epg, &epg2);
    }

    #[test]
    fn test_shift_conjs() {
        let mut epg = EPGVecRepresentation::new(3);
        let mut epg2 = EPGVecRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg2, &r2);

        gradient_shift(&mut epg, 1);
        println!("{:?}", epg);
        gradient_shift(&mut epg, -1);
        println!("{:?}", epg);

        assert_eq!(&epg, &epg2);
    }

    
    fn test_complex_close_l1(a: &Complex64, b: &Complex64, tolerance: f64) -> bool {
        let diff = a.norm() - b.norm();
        if diff.abs() <= tolerance {
            return true;
        } else {
            return false;
        }
    }
   
    
    fn epg_close(epg1 : &EPGVecRepresentation, epg2: &EPGVecRepresentation)
    {

        let tol = 1e-9;
        assert!(epg1.f_p.iter().zip(epg2.f_p.iter())
           .all(|(p1, p2)|  test_complex_close_l1(p1, p2, tol) ) ) ;

        assert!(epg1.f_n.iter().zip(epg2.f_n.iter())
           .all(|(p1, p2)|  test_complex_close_l1(p1, p2, tol) ) );

        assert!(epg1.z.iter().zip(epg2.z.iter())
           .all(|(p1, p2)|  test_complex_close_l1(p1, p2, tol) ) );
    }

    #[test]
    fn test_flipback() {
        let mut epg = EPGVecRepresentation::new(3);
        let epg2 = EPGVecRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);
        let rm2 = gen_rotation_matrix(-PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg, &rm2);


        println!("{:?}", epg);
        println!("{:?}", epg2);

        epg_close(&epg, &epg2);


    }

}
