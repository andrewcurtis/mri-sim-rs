use ndarray::{Array, Ix2};
use num_complex::Complex64;

pub(crate) trait EPG {
    // This trait describes the interface for a mutable echo phase graph.
    fn new(n_states: usize) -> Self;
    fn read(&self) -> Complex64;
    fn excite(&mut self);
    fn rotate(&mut self, rmat: &Array<Complex64, Ix2>);
    fn spoil(&mut self, ntwists: i32);
    fn grelax(&mut self, et1d: Complex64, et2d: Complex64, ntwists: i32);
    fn delay(&mut self, et1d: Complex64, et2d: Complex64);
}


pub enum Tissue {
    WhiteMatter,
    GreyMatter,
    Caudate, 
    CerebroSpinalFluid,
    Thalamus,
    Blood
}

pub struct TissueProperties {
    pub name: &'static str,
    pub pd: f32,
    pub t1: f32,
    pub t2: f32,
    pub t2s: f32,
}