# waterTools

A collection of efficient Python and Cython tools for processing and analyzing water molecular dynamics simulations, primarily designed for **GPUMD** workflows.

## Key Tools

### File Conversion & IO

- **`exyz2gpumd.py`**: Convert extended XYZ files to GPUMD input format.
- **`lmp2gpumd.py`**: Convert LAMMPS/ASE structures to GPUMD format.
- **`writeGPUMDdump.py`**: Utilities for writing GPUMD-compatible dump files.
- **`c_parser.pyx`**: Fast Cython-based XYZ file parser.

### Analysis

- **`rdf.py`**: Calculate Radial Distribution Functions (RDF) with Numba acceleration.
- **`spectral.py`**: Spectral analysis tools for molecular dynamics trajectories.
- **`temps.py`**: Temperature calculation and analysis utilities.
- **`wignerEXC.py`**: Wigner excitation script (only one bond).
- **`wignerEXC2B.py`**: Wigner excitation script (both bonds).

### Utilities

- **`unwrap_coords.py`**: Handle coordinate unwrapping for periodic boundary conditions.

## Dependencies

- **Python 3.x**
- `numpy`
- `scipy`
- `numba`
- `ase` (Atomic Simulation Environment)
- `cython` (for compiling parser extensions)
