use pyo3::prelude::*; 

use std::f64::consts::PI;

mod epg;
mod sequences;
mod types;

use types::EPG;


#[pyfunction]
fn demo() {
    println!("Hello, world!");

    let params = sequences::fse::FseParams {
        etl: 20,
        esp: 0.01,
        t1: 0.58,
        t2: 0.11,
        refocus_angle: PI / 2.0,
        cpmg_phase: PI / 2.0,
        debug_print: false,
    };
    let res = sequences::fse::simulate(params);
    println!("{:?}", res);

}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn epg_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(demo, m)?)?;

    Ok(())
} 

