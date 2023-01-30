use num_complex::Complex64;

use crate::{epg, types::EPG};

pub struct SpaceParams {
    pub etl: usize,
    pub t1: f64,
    pub t2: f64,
    pub esp: f64,
    pub refocus_angle: f64,
    pub cpmg_phase: f64,
    pub debug_print: bool,
}

pub fn simulate(params: SpaceParams) -> Vec<Complex64>{
    
    let etl = params.etl;

    let mut epg = epg::vec::EPGVecRepresentation::new(etl / 2 + 1);

    let mut signal: Vec<Complex64> = Vec::with_capacity(etl + 1);

    let x180 = epg::common::gen_rotation_matrix(params.refocus_angle, params.cpmg_phase);

    // dt is the spacing of our events, unsed for dephasing/relaxation
    let dt = params.esp / 2.0;
    let et1d = Complex64::from((-dt / params.t1).exp());
    let et2d = Complex64::from((-dt / params.t2).exp());

    epg.excite();

    for _ in 0..etl {
        epg.grelax(et1d, et2d, 1);
        epg.rotate(&x180);
        epg.grelax(et1d, et2d, 1);

        signal.push(epg.read());
    }

    if params.debug_print {
        println!("Signal: {:?}", signal);
        println!("{:}", epg::vec::to_mxy(&epg));
        println!("{:}", epg::vec::to_mz(&epg));
    }

    return signal;
}
