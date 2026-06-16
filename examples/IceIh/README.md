# Ice Ih Workflow

## Generating Ice Samples

All ice generation has been delegated to [GenIce2](https://github.com/genice-dev/GenIce2).

First, generate an ice sample suitable for GPUMD.

```bash
ICESEED="$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')"
genice2 Ih --rep 4 4 4 -f exyz -s "${ICESEED}" > ice.temp 2> log.genice
../../tools/exyz2gpumd.py ice.temp -o ice.xyz
```

Then, the sample needs to be relaxed.

```bash
cp relax.in run.in
cp ice.xyz model.xyz
gpumd > log.relax 2>&1
```

## Production Run

Now that the ice sample has been relaxed according to the NEP PES, it is ready to be excited.

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
