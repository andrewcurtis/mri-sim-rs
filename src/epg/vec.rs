use ndarray::{arr1, Array, Ix2};
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
        // 90 degree excitation about Y to generate pure x state from [0 0 1] mz.
        let rot = gen_rotation_matrix(PI / 2.0, PI / 2.0);
        Self::rotate(self, &rot);
    }

    fn rotate(&mut self, rmat: &Array<Complex64, Ix2>) {
        rf_rotation(self, rmat);
    }

    fn spoil(&mut self, ntwists: i32) {
        gradient_shift(self, ntwists);
    }

    fn grelax(&mut self, et1d: Complex64, et2d: Complex64, ntwists: i32) {
        gradient_shift(self, ntwists);
        relaxation(self, et1d, et2d);
    }

    fn delay(self: &mut Self, et1d: Complex64, et2d: Complex64) {
        relaxation(self, et1d, et2d);
    }
}

impl Default for EPGVecRepresentation {
    fn default() -> Self {
        Self::new(3)
    }
}

pub(crate) fn to_mxy(epg: &EPGVecRepresentation) -> Complex64 {
    // sum every element of epg.f_p
    let pos_sum: Complex64 = epg
        .f_p
        .iter()
        .enumerate()
        .map(|(ix, x)| x * (Complex::i() * (2.0 * PI * ix as f64)).exp())
        .sum();
    // sum every element of epg.f_n, skipping first
    let neg_sum: Complex64 = epg
        .f_n
        .iter()
        .skip(1)
        .enumerate()
        .map(|(ix, x)| x * (Complex::i() * (2.0 * PI * ix as f64)).exp())
        .sum();

    pos_sum + neg_sum
}

pub(crate) fn to_mz(epg: &EPGVecRepresentation) -> Complex64 {
    let mut mz = Complex64::from(0.0);
    // sum every element of epg.z 1..
    mz = epg
        .z
        .iter()
        .skip(1)
        .enumerate()
        .map(|(ix, x)| x * (Complex::i() * (2.0 * PI * ix as f64)).exp())
        .sum();
    // double the sum (to account for duality of conjugate states that we don't allocate)
    // and add 0 element
    mz = 2.0 * mz + epg.z[0];
    mz
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

fn relaxation(epg: &mut EPGVecRepresentation, et1d: Complex64, et2d: Complex64) {
    for x in epg.f_n.iter_mut() {
        *x = *x * et2d
    }

    for x in epg.f_p.iter_mut() {
        *x = *x * et2d
    }

    for (ix, z) in epg.z.iter_mut().enumerate() {
        if ix == 0 {
            *z = (1.0 - et1d) + (*z * et1d)
        } else {
            *z = *z * et1d
        }
    }
}

// TODO: make a v2 that doesn't recurse and just shifts by multile steps
fn gradient_shift(epg: &mut EPGVecRepresentation, ntwists: i32) {
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

        let t1 = 1.0_f64;
        let t2 = 0.05_f64;
        let dt = 1e-3;

        let et1d = Complex::from((-dt / t1).exp());
        let et2d = Complex::from((-dt / t2).exp());

        relaxation(&mut epg, et1d, et2d);
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

    fn epg_close(epg1: &EPGVecRepresentation, epg2: &EPGVecRepresentation) {
        let tol = 1e-9;
        assert!(epg1
            .f_p
            .iter()
            .zip(epg2.f_p.iter())
            .all(|(p1, p2)| test_complex_close_l1(p1, p2, tol)));

        assert!(epg1
            .f_n
            .iter()
            .zip(epg2.f_n.iter())
            .all(|(p1, p2)| test_complex_close_l1(p1, p2, tol)));

        assert!(epg1
            .z
            .iter()
            .zip(epg2.z.iter())
            .all(|(p1, p2)| test_complex_close_l1(p1, p2, tol)));
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

    #[test]
    fn test_t2decay() {
        let mut epg = EPGVecRepresentation::new(3);

        epg.excite();

        // \infty t1
        let t1 = 1e9_f64;
        // t2
        let t2 = 0.10_f64;

        let dt = 0.2_f64;

        let et1d = Complex::from((-dt / t1).exp());
        let et2d = Complex::from((-dt / t2).exp());

        epg.delay(et1d, et2d);

        let signal = epg.read();

        let expected = et2d;
        println!("expected = {:?}", expected);
        println!("signal = {:?}", signal);
        assert!(test_complex_close_l1(&expected, &signal, 1e-7));
    }
    #[test]
    fn test_conjugate_states() {
        // test if f_p(k) == conj(f_n(k))

        let mut epg = EPGVecRepresentation::new(16);

        let ex_45_30 = gen_rotation_matrix(PI / 4.0, PI / 6.0);

        epg.rotate(&ex_45_30);

        let test_f_conj = epg
            .f_n
            .iter()
            .zip(epg.f_p.iter())
            .all(|(&a, &b)| test_complex_close_l1(&a, &b.conj(), 1e-7));

        assert!(test_f_conj);
    }
}
