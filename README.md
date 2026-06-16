# Ultrafast Water and Ice

This repository contains simulation workflows, analysis tools, and processed data associated with the study:

**How Hydrogen-Bond Topology Shapes Ultrafast O–H Stretch Relaxation in Water and Ice**  
Alessandro Serra, Davide Donadio, and Riccardo Dettori

The project investigates ultrafast vibrational energy relaxation in hydrogen-bonded H2O phases: liquid water, proton-disordered ice Ih, and proton-ordered ice IX. Non-equilibrium molecular dynamics simulations are performed with GPUMD using a NEP potential, after initializing selected O–H stretching excitations through Wigner phase-space sampling.

## Repository Structure

```text
ultrafast_water_ice/
├── data/       # Processed data used for analysis
├── examples/   # Minimal workflows for Water, Ice Ih, and Ice IX
├── notebooks/  # Jupyter notebooks for generating all the plots
├── plots/      # Plots generated from the notebooks
└── tools/      # Conversion, excitation, and analysis utilities
```

The `examples/` directory contains separate workflows for:

```text
examples/Water/
examples/IceIh/
examples/IceIX/
```

Each example includes the files needed to generate or prepare the initial structure, relax it, apply the vibrational excitation, and run the production simulation.

## Requirements

The workflows require:

- [GPUMD](https://github.com/brucefan1983/GPUMD)
- [LAMMPS](https://www.lammps.org/) for liquid water sample generation
- [GenIce2](https://github.com/genice-dev/GenIce2) for ice sample generation
- Python 3 with common scientific packages such as `numpy`, `scipy`, `ase`, `numba`, `matplotlib`, and `jupyter`

## Basic Workflow

A typical workflow consists of:

1. generating or preparing an initial structure;
2. relaxing the structure with GPUMD;
3. exciting selected O–H bonds using Wigner sampling;
4. running a non-equilibrium production simulation;
5. analyzing vDOS, mode temperatures, and RDFs.

After preparing an initial structure, the relaxation is performed with:

```bash
cp relax.in run.in
cp initial.xyz model.xyz
gpumd > log.relax 2>&1
```

The vibrational excitation can then be generated with:

```bash
./tools/wignerEXC2B.py relaxed.xyz -d excited.xyz -e 0.1 -vv > log.excite 2>&1
```

where `-e 0.1` corresponds to exciting 10% of the molecules.

The production run is then carried out with:

```bash
cp excited.xyz model.xyz
cp production.in run.in
gpumd > log.production 2>&1
```

The file `excited-indexes.dat` contains the indices of the atoms excited during the Wigner initialization step.

## Tools

The `tools/` folder contains utilities for preparing simulations and analyzing trajectories.

Main scripts include:

- `exyz2gpumd.py`: converts extended XYZ files to GPUMD format;
- `lmp2gpumd.py`: converts LAMMPS/ASE structures to GPUMD format;
- `wignerEXC.py`: initializes a Wigner excitation on one O–H bond;
- `wignerEXC2B.py`: initializes Wigner excitations on both O–H bonds;
- `rdf.py`: computes radial distribution functions;
- `spectral.py`: computes spectral quantities and vDOS;
- `temps.py`: computes mode-resolved kinetic temperatures.

## Analysis

The repository provides tools and notebooks to analyze:

- time-resolved vibrational density of states;
- mode-resolved kinetic temperatures;
- O–O radial distribution functions;
- shell-resolved structural response;
- relaxation times between molecular and intermolecular degrees of freedom.

The analysis separates initially excited molecules from the unexcited environment, allowing the redistribution of vibrational energy from O–H stretching modes into bending, librational, translational, and hydrogen-bond degrees of freedom to be monitored.

## Citation

If you use this repository, please cite the associated paper:

```bibtex
TO BE ADDED
```

## Authors

- Alessandro Serra
- Davide Donadio
- Riccardo Dettori

## License

This repository is distributed under the BSD 3-Clause License. See the `LICENSE` file for details.
