# Eigen Vector Following Trasition State Search Algorithm
Eigen vector following transition state search method based on PRFO method and Bofill's Hessian estimation using GAUSSIAN as calculation engine.

## Usage:
Create a `CM_TSS` object and pass the path to the `settings.json` file and the path to an initial structure. The `settings.json` file is formated as (default values in parantheses):
```
{
    "N": {number of atoms} (required),
    "charge": {charge of the system} (0),
    "spin": {spin of the system} (1),
    "N-procs": {number of processors to use} (8),
    "conv-radius":{atom position convergence radius} (1e-1),
    "conv-grad":{gradient convergence radius} (1e-6),
    "trust-radius": {trust radius for maximum displacement} (0.2)
    "max-iter": {max number of iterations} (10),
    "reset-H-every": {how many iterations between recalculating Hessian exactly from force calculations} (20)
    "working-dir": {path to directory with all the files} (required),
    "basis-f-name": {name of the file containing basis information for GAUSSIAN calculations} ("" skipped)
    "history-f-name": {name of the file to record optimization history}("history.xyz")
    "final-f-name": {name of the file to record optimized structure}("final.xyz")
    "submit-f-dir": {path to GAUSSIAN submission file} (required),
    "gaussian-f-name": {name for GAUSSIAN input file} ("in"), 
    "energy-header-calc": {GAUSSIAN header for energy calculations} ("#P wB97XD/6-31G** nosymm force"),
    "hess-header-calc": {GAUSSIAN header for Hessian calculations } ("#P b3lyp/6-31G** nosymm freq"),
}
```
A GAUSSIAN submission script is necessary. This is a system depandant file.

`basis-f-name` is the name of the file containing basis specifications if necessary and will be added to the bottom of the GAUSSIAN input file.

Input coordinates file should have the format:
```
{atomic number} {-1 for frozen 0 otherwise} {x} {y} {z}
```

There's an example calculation for HCN available under the `HCN` directory. The `EF_TSS.py` should either be added to PATH or copied to the same directory as the test script.
