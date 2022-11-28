use ndarray::{array, Array, Ix2, arr1};
use num_complex::{Complex, Complex64};

use std::collections::VecDeque;
use std::fmt;

use std::f64::consts::PI;

use super::common::gen_rotation_matrix;
use crate::types::EPG;


#[derive(Debug, PartialEq)]
pub(crate) struct EPGVecRepresentation {
    length: usize,
    f_p: VecDeque<Complex64>,
    f_n: VecDeque<Complex64>,
    z: VecDeque<Complex64>,
}

impl fmt::Display for EPGVecRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for ix in 0..self.length {
            write!(
                f,
                "f+ {:.3}\tf- {:.3}\tz {:.3}\n",
                self.f_p[ix], self.f_n[ix], self.z[ix]
            )?;
        }
        Ok(())
    }
}

impl crate::types::EPG for EPGVecRepresentation {
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

    fn read(&self) -> Complex64 {
        self.f_p[0]
    }

    fn excite(&mut self) {
        let rot = gen_rotation_matrix(PI / 2.0, 0.0);
        Self::rotate(self, &rot);
    }

    fn rotate(&mut self, rmat: &Array<Complex64, Ix2>) {
        rf_rotation(self, rmat);
    }

    fn grelax(&mut self, dt: f64, t1: f64, t2: f64, ntwists: i64) {
        gradient_shift(self, ntwists);
        relaxation(self, dt, t1, t2);
    }

    fn delay(self: &mut Self, dt: f64, t1: f64, t2: f64) {
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


#[cfg(test)]
mod tests {

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

    #[test]
    fn test_flipback() {
        let mut epg = EPGVecRepresentation::new(3);
        let epg2 = EPGVecRepresentation::new(3);

        let r2 = gen_rotation_matrix(PI / 4.0, 0.0);
        let rm2 = gen_rotation_matrix(-PI / 4.0, 0.0);

        rf_rotation(&mut epg, &r2);
        rf_rotation(&mut epg, &rm2);

        assert_eq!(&epg, &epg2);
    }

}
