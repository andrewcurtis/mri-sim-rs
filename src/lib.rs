//! lib.rs
#[macro_use]
extern crate pyo3_built;
extern crate pyo3;

use pyo3::prelude::*; 

use std::f64::consts::PI;

mod epg;
pub mod sequences;
pub mod types;
//mod tissues;



#[pyfunction]
fn demo() {
    println!("Hello, world!");

    let params = sequences::space::SpaceParams {
        etl: 220,
        esp: 0.01,
        t1: 0.58,
        t2: 0.11,
        refocus_angle: PI,
        cpmg_phase: PI / 2.0,
        debug_print: false,
    };
    let res = sequences::space::simulate(params);
    //println!("{:?}", res);

}


#[allow(dead_code)]
mod build {
    include!(concat!(env!("OUT_DIR"), "/built.rs"));
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn epg_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(demo, m)?)?;

    m.add("__build__", pyo3_built!(_py, build))?;
    Ok(())
} 

