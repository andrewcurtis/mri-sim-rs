use num_complex::Complex64;

use crate::{epg, types::EPG};

pub struct SeParams {
    pub t1: f64,
    pub t2: f64,
    pub read_delay: f64,
    pub nreads: usize,
    pub refocus_angle: f64,
    pub refocus_phase: f64,
    pub echo_time: f64,
    pub debug_print: bool,
}

pub fn simulate(params: SeParams) -> Vec<Complex64> {
    let nreads = params.nreads;

    let mut epg = epg::vec::EPGVecRepresentation::new(nreads + 1);

    let mut signal: Vec<Complex64> = Vec::with_capacity(nreads + 1);

    let x180 = epg::common::gen_rotation_matrix(params.refocus_angle, params.cpmg_phase);

    // dt is the spacing of our events, unsed for dephasing/relaxation
    let dt = params.read_delay;
    let et1d = Complex64::from((-dt / params.t1).exp());
    let et2d = Complex64::from((-dt / params.t2).exp());

    let dt2 = params.esp / 2.0;
    let et1d2 = Complex64::from((-dt / params.t1).exp());
    let et2d2 = Complex64::from((-dt / params.t2).exp());

    epg.excite();
    epg.grelax(et1d, et2d, 1);
    epg.rotate(&x180);
    epg.grelax(et1d, et2d, 1);


    for _ in 0..nreads {
        epg.grelax(et1d, et2d, 0);
        signal.push(epg.read());
    }

    if params.debug_print {
        println!("Signal: {:?}", signal);
        println!("{:}", epg::vec::to_mxy(&epg));
        println!("{:}", epg::vec::to_mz(&epg));
    }

    return signal;
}
