use ndarray::prelude::*;
use num_complex::{Complex, Complex64};
use rand_distr::{Distribution, StandardNormal, UnitCircle};

use eframe::egui;

use egui::plot::{Plot, Line, PlotPoints};

use std::rc::Rc;
use std::thread;
use std::sync::{mpsc, Arc};
use std::time::Duration;
use std::sync::Mutex;

use std::f64::consts::PI;


macro_rules! zip {
    ($x: expr) => ($x);
    ($x: expr, $($y: expr), +) => (
        $x.iter().zip(
            zip!($($y), +))
    )
}

struct MyApp {
    index: i32,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            index: 0,
        }
    }
}


impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");
            ui.add(egui::Slider::new(&mut self.index, 0..=12).text("index"));

            let sin: PlotPoints = (0..1000).map(|i| {
                let x = (i*self.index) as f64 * 0.01; 
                [x, x.sin()]
            }).collect();
            let line = Line::new(sin);
            Plot::new("my_plot").view_aspect(2.0).show(ui, |plot_ui| plot_ui.line(line));
        });

    }
}


fn main() {
    println!("Hello, world!");

    let r1 = gen_rotation_matrix(3.1415/2.0, 0.0);
    println!("{}", r1);
    let r2 = gen_rotation_matrix(3.1415/4.0, 0.0);
    println!("{}", r2);
    let x180 = gen_rotation_matrix(3.1415, 0.0);
    println!("{x180:1.3}");
    
    {
        let mut epg = EPGVecRepresentation::new(2);
        println!("{:?}", epg);

        let r2 = gen_rotation_matrix(PI/4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        println!("{:?}", epg);

        relaxation(&mut epg, 100e-3, 1.0, 0.1);
        println!("Relaxed 1 {:?}", epg);

        relaxation(&mut epg, 100e-3, 1.0, 0.1);
        println!("Relaxed 2 {:?}", epg);

        relaxation(&mut epg, 500e-3, 1.0, 0.1);
        println!("Relaxed 3 {:?}", epg);
    }

    //std::thread::spawn( || work_with_arrays() );
    // let options = eframe::NativeOptions::default();
    // eframe::run_native(
    //     "My egui app", 
    //     options, 
    //     Box::new(|_cc| Box:: new(MyApp::default())),
    // );

}

fn eiphi(phi:f64) -> Complex64 {
    (Complex::i()*phi).exp()
}

#[derive(Debug)]
struct EPGVecRepresentation {
    length: usize,
    f_p : Vec<Complex64>,
    f_n : Vec<Complex64>,
    z: Vec<Complex64>,
}


impl EPGVecRepresentation {
    fn new(sz: usize) -> Self {
        let length = sz;
        let mut f_p = Vec::with_capacity(length);
        let mut f_n = Vec::with_capacity(length);
        let mut z = Vec::with_capacity(length);

        f_p.push(Complex64::from(0.0));
        f_n.push(Complex64::from(0.0));
        z.push(Complex64::from(1.0));

        for _ in 1 .. length {
            f_p.push(Complex64::from(0.0));
            f_n.push(Complex64::from(0.0));
            z.push(Complex64::from(0.0));
        }

        Self {
            length, f_p, f_n, z
        }
    }
}

impl Default for EPGVecRepresentation {
    fn default() -> Self {
        Self::new(3)
    }
}



fn rf_rotation(epg : &mut EPGVecRepresentation, rmat : &Array<Complex64, Ix2>) 
{
    for ix in 0..epg.length
    {
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

fn relaxation(epg : &mut EPGVecRepresentation, dt : f64, t1:f64, t2: f64) 
{
    let t1d = Complex::from((-dt/t1).exp());
    let t2d = Complex::from((-dt/t2).exp());
    
    for x in epg.f_n.iter_mut() {
        *x = *x * t2d
    }

    for x in epg.f_p.iter_mut() {
        *x = *x * t2d
    }

    for (ix, z) in epg.z.iter_mut().enumerate() {
       if ix == 0 {
           *z =  (1.0 - t1d) + (*z * t1d)
       }
       else {
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
        [ ca2*ca2, sa2*sa2*eiphi(2.0*phi), -j*eiphi(phi)*sa ],
        [eiphi(-2.0*phi)*sa2*sa2, ca2*ca2, j*eiphi(-phi)*sa ],
        [ -j/2.0 * eiphi(-phi)*sa, j/2.0*eiphi(phi)*sa, ca ]
    ]
}


fn gradeint_shift(n_twists: i64) {

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
    use super::*;
    #[test]
    fn test_180_rotation() {

        let mut epg = EPGVecRepresentation::new(2);
        println!("{:?}", epg);

        let r2 = gen_rotation_matrix(PI, 0.0);

        rf_rotation(&mut epg, &r2);
        println!("{:?}", epg);

        relaxation(&mut epg, 100e-3, 1.0, 0.1);
    }


}