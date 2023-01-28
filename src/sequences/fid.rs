use num_complex::Complex64;

use crate::{epg, types::EPG};

pub struct FidParams {
    pub nreads: usize,
    pub t1: f64,
    pub t2: f64,
    pub echo_time: f64,
    pub debug_print: bool,
}

pub fn simulate(params: FidParams) -> Vec<Complex64> {
    let nreads = params.nreads;

    let mut epg = epg::vec::EPGVecRepresentation::new(nreads + 1);
    let mut signal: Vec<Complex64> = Vec::with_capacity(nreads + 1);

    let dt = params.echo_time;
    let et1d = Complex64::from((-dt / params.t1).exp());
    let et2d = Complex64::from((-dt / params.t2).exp());

    epg.excite();

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
