# Water Workflow

## Generating Water Samples

Water samples can be generated in many ways. In this work, LAMMPS was used to create all initial samples.

```bash
cd lammps/
lmp -in in.generate
```

Since the dump file is in LAMMPS trajectory format, it must be converted to a GPUMD-compatible format.

```bash
./LAMMPSdump2XYZ.py nnp.dump -o water.xyz
```

Then, the sample needs to be relaxed.

```bash
cp relax.in run.in
cp water.xyz model.xyz
gpumd > log.relax 2>&1
```

## Production Run

Now that the water sample has been relaxed according to the NEP PES, it is ready to be excited.

```bash
../../tools/wignerEXC2B.py relaxed.xyz -d excited.xyz -e 0.1 -vv > log.excite 2>&1
```

Running `wignerEXC2B.py` with the `-h` flag will print its usage and available options.

Now the production run can be carried out.

```bash
cp excited.xyz model.xyz
cp production.in run.in
gpumd > log.production 2>&1
```

## Analysis

The tools in the `../../tools/` folder can now be used to analyse the trajectories. The file `excited-indexes.dat` contains the atom or atoms excited in the previous step.
