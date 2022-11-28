use ndarray::{Array, Ix2};
use num_complex::Complex64;

pub(crate) trait EPG {
    // This trait describes the interface for a mutable echo phase graph.
    fn new(n_states: usize) -> Self;
    fn read(&self) -> Complex64;
    fn excite(&mut self);
    fn rotate(&mut self, rmat: &Array<Complex64, Ix2>);
    fn grelax(&mut self, dt: f64, t1: f64, t2: f64, ntwists: i64);
    fn delay(&mut self, dt: f64, t1: f64, t2: f64);
}
