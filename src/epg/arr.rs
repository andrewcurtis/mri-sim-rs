use ndarray::{arr1, Array, Ix2, s};
use num_complex::{Complex, Complex64};

use std::fmt;

use std::f64::consts::PI;

use super::common::gen_rotation_matrix;
use crate::types::EPG;

#[derive(Debug, PartialEq)]
pub(crate) struct EPGArrayRepresentation {
    length: usize,
    fzk: Array<Complex64, Ix2>,
}

// impl fmt::Display for EPGArrayRepresentation {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         for ix in 0..self.length {
//             write!(
//                 f,
//                 "{:.3}\n",
//                 self.fzk[ix],
//             )?;
//         }
//         Ok(())
//     }
// }

impl crate::types::EPG for EPGArrayRepresentation {
    fn new(n_states: usize) -> Self {
        let length = n_states;
        let mut fzk = Array::zeros((3, length));

        fzk[[2,0]] = Complex64::from(1.0);

        Self {
            length,
            fzk,
        }
    }

    fn read(&self) -> Complex64 {
        self.fzk[[0,0]]
    }

    fn excite(&mut self) {
        let rot = gen_rotation_matrix(PI / 2.0, 0.0);
        Self::rotate(self, &rot);
    }

    fn rotate(&mut self, rmat: &Array<Complex64, Ix2>) {
        rf_rotation(self, rmat);
    }

    fn grelax(&mut self, dt: f64, t1: f64, t2: f64, ntwists: i32) {
        gradient_shift(self, ntwists);
        relaxation(self, dt, t1, t2);
    }

    fn delay(self: &mut Self, dt: f64, t1: f64, t2: f64) {
        relaxation(self, dt, t1, t2);
    }
}

impl Default for EPGArrayRepresentation {
    fn default() -> Self {
        Self::new(3)
    }
}

// stopped ehre --- 

fn rf_rotation(epg: &mut EPGArrayRepresentation, rmat: &Array<Complex64, Ix2>) {
    // Think we want iter lanes 
    // see https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#method.lanes
    // and https://docs.rs/ndarray/latest/ndarray/iter/index.html
    let rot = rmat.dot(&epg.fzk);
    epg.fzk = rot;
}

fn relaxation(epg: &mut EPGArrayRepresentation, dt: f64, t1: f64, t2: f64) {
    let t1d = Complex::from((-dt / t1).exp());
    let t2d = Complex::from((-dt / t2).exp());

    // f pos and f - (rows 0 and 1) get attentuated by t2 decay
    epg.fzk.slice_mut(s![0, ..]).mapv_inplace(|x| x * t2d);
    epg.fzk.slice_mut(s![1, ..]).mapv_inplace(|x| x * t2d);

    // z states attenuate by t1 decay, while z0 has regrowth
    epg.fzk[[2,0]] = (1.0-t1d) + (epg.fzk[[2,0]] *t1d);
    epg.fzk.slice_mut(s![2, 1..]).mapv_inplace(|x| x * t1d);

}

fn gradient_shift(epg: &mut EPGArrayRepresentation, ntwists: i32) {
    // Shift states.
    // nshfits represents the number of 2pi dephasing steps to shift by
    // F0 is special, since f_p [0] is f0, and f_n[0] is f0*(conj)

    // can we do a[1..end] = a[0..end-1] etc? 
    match ntwists {
        n if n == 0 => return,
        n if n > 0 => {
            // f_p becomes more positive. f_m becomes less negative
            let f1 = epg.fzk.slice(s![0, ..-n]);
            let f2 = epg.fzk.slice(s![0, n..]);
            
        }
        n if n < 0 => {
            // f_p becomes less positive. f_m becomes more negative
        }
        _ => unreachable!("Should never happen."),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_180_rotation() {
        let mut epg = EPGArrayRepresentation::new(1);

        let r2 = gen_rotation_matrix(PI, 0.0);

        rf_rotation(&mut epg, &r2);
        println!("{:?}", epg);

        relaxation(&mut epg, 100e-3, 1.0, 0.1);
    }

    #[test]
    fn test_shift_symmetric() {
        let mut epg = EPGArrayRepresentation::new(3);
        let mut epg2 = EPGArrayRepresentation::new(3);

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
        let mut epg = EPGArrayRepresentation::new(3);
        let mut epg2 = EPGArrayRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg2, &r2);

        gradient_shift(&mut epg, 1);
        println!("{:?}", epg);
        gradient_shift(&mut epg, -1);
        println!("{:?}", epg);

        assert_eq!(&epg, &epg2);
    }

    #[test]
    fn test_flipback() {
        let mut epg = EPGArrayRepresentation::new(3);
        let epg2 = EPGArrayRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);
        let rm2 = gen_rotation_matrix(-PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg, &rm2);

        assert_eq!(&epg, &epg2);
    }
}
