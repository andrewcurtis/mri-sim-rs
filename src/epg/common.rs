use ndarray::{array, Array, Ix2};
use num_complex::{Complex, Complex64};

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
