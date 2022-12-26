use ndarray::{array, Array, Array2, Ix2};
use num_complex::{Complex, Complex64, ComplexFloat};
use std::f64::consts::PI;


fn eiphi(phi: f64) -> Complex64 {
    (Complex::i() * phi).exp()
}

pub(crate) fn gen_rotation_matrix(alpha: f64, phi: f64) -> Array<Complex64, Ix2> {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    fn matrix_close(a: Array<Complex64, Ix2>, b: Array<Complex64, Ix2>, epsilon:f64) -> bool {
        ndarray::Zip::from(&a)
        .and(&b)
        .all(|&x, &y| {
            (x - y).abs() <= epsilon  
        })
    }
    #[test]
    fn test_gen_matrix_1() {
        let t_y_90 = gen_rotation_matrix(PI / 2.0, PI / 2.0);
        let expected_t_y_90: Array2<Complex64> =
            array![ [Complex::from(0.5), Complex::from(-0.5), Complex::from(1.0)], 
                    [Complex::from(-0.5), Complex::from(0.5), Complex::from(1.0)], 
                    [Complex::from(-0.5), Complex::from(-0.5), Complex::from(0.0)]];

        println!("{}", t_y_90);
        println!("{}", expected_t_y_90);
        assert!(matrix_close(t_y_90, expected_t_y_90, 1e-8));

    }
}
