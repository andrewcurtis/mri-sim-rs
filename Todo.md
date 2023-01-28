Todo tasks
----------

- how do we use approx:: for unit tests?
- array2d supports approx over whole arrays more easily
- array2d version of EPG
  - do we want 3xN or Nx3? benchmark.
  - transpose and shift or just shift?


- try making a more efficient version of the deque using
rotate_right etc
- benchmark both implementations
- think about how we parallelize
- implement other EPG physics like MT and Diffusion.

maybe not:
(better to provide python wrapper?)
- implement plotting of signal 
- plotting of fzk states.


- sequences, on trait 
  - fid
  - se
  - fse 
- instead of directly simulating, let's have the sequence do params -> Vec<events>
  then generatic simulate function that takes   Vec<events> -> Vec<complex> etc.


- python interface : 
  - use py03 / maturin to build extension
  - more generic simulate entry point
  - evenly spaced events
  - Vec<event> how do we expose ADTs with data to python ?
