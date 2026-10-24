# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DCFoil.jl is a Julia package (Julia 1.11) for fluid-structure interaction analysis and gradient-based design optimization of composite marine appendages (hydrofoils, keels, rudders) using low-order models: a composite beam FEM, lifting-line/strip hydrodynamics, and static/forced/flutter solvers with derivatives. A Python layer (`dcfoil/`) wraps the Julia code as OpenMDAO components through `juliacall` and `omjlcomps`.

## Environment

- Python: activate the venv first with `source /Users/wolfgang/.venv/base/bin/activate` (it provides `juliacall`, `openmdao`, `omjlcomps`, `jax` and `testflo`).
- `setup_juliapkg.py` copies the Project.toml deps into PythonCall's `juliapkg` so the Python side gets the same Julia environment.

## Commands

```bash
# Full Julia test suite (what CI runs)
julia --project=. -e 'using Pkg; Pkg.test()'
# or
julia --project=. test/runtests.jl

# Single test: each test/test_*.jl file includes its own src modules via relative paths,
# so run from test/ and call the test function directly (each returns an error/tolerance ratio, which passes at <= 1.0)
cd test && julia --project=.. -e 'include("test_vam.jl"); println(test_VAM_ellipse())'

# Python/OpenMDAO tests (CI: RunUnitTestsPython.yml). Run from dcfoil/ because they use jl.include("../src/...")
cd dcfoil && testflo .
cd dcfoil && testflo test_transfer_components.py   # single file

# Docs
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl

# Example runs (naming convention: runscripts/main_<job-title>.jl)
julia --project=. runscripts/main_mothtfoil.jl
```

There is no linter configured.

## Architecture

**Module assembly.** `src/DCFoil.jl` `include`s every submodule from a hard-coded list, and **the order matters**: later modules use earlier ones through relative `using ..Module` imports. When you add a file, put it in that list after the modules it depends on. Many submodules (and the tests) also `include` their own dependencies directly, so they can be loaded on their own.

**Two entry paths:**
1. **Julia API** (`init_model`, `run_model`, `evalFuncs`, `evalFuncsSens` in `src/DCFoil.jl`). Inputs are an `appendageParamsList` (design-variable dicts such as `alfa0`, `sweep`, `c`, `toc`, `theta_f`, ...) plus a `solverOptions` dict whose `"appendageList"` holds one options dict per component (`config` = `"wing"`/`"full-wing"`/`"t-foil"`, `material`, `nNodes`, ...). `InitModel` builds the FOIL/STRUT/HULL structs either from a DV dict or from LE/TE coordinates (`.dcf`/`.dat` meshes in `INPUT/`). After that, `SolveStatic`, `SolveForced` and `SolveFlutter` run, and `CostFunctions` lists the function names each solver can evaluate.
2. **OpenMDAO coupled path** (`dcfoil/coupled_analysis.py`, `run_OMDCFoil*.py`). This path loads the `*_om.jl` files (`struct/beam_om.jl`, `hydro/liftingline_om.jl`, `loadtransfer/ldtransfer_om.jl`, `solvers/solveflutter_om.jl`, `solvers/solveforced_om.jl`) into a `juliacall` module. Each file defines `OpenMDAOCore.AbstractExplicitComp`/`AbstractImplicitComp` structs with `setup`/`compute!`/`compute_partials!` (or `linearize!`), and the Python side wraps them in `JuliaExplicitComp`/`JuliaImplicitComp`. `dcfoil/transfer.py` holds JAX reference versions of the load/displacement transfer that the Julia `LDTransfer` components are checked against. `pyDCFoil.py` is the older Python class interface.

**Disciplines:**
- `struct/`: `FEMMethods` (beam FEM, default element `ELEMTYPE = "COMP2"` in `constants/SolutionConstants.jl`, 9 DOF per node), `BeamProperties`/`MaterialLibrary` (composite section properties), `VAM` (variational asymptotic cross-sectional analysis), `StrainBeam` (geometrically exact strain-based beam).
- `hydro/`: `LiftingLine` (with `VPM` airfoil panel method), `GlauertLL`, `HydroStrip` (strip theory and unsteady hydrodynamic matrices for flutter/forced), `OceanWaves`.
- `loadtransfer/LDTransfer.jl`: moves loads and displacements between hydro collocation points and beam nodes, and conserves force and virtual work.
- `solvers/`: `NewtonRaphson`, `EigenvalueProblem`, `SolverRoutines` (shared, holds many custom rrules), and the static/forced/flutter solvers.

**Derivatives.** Functions are written to be AD-compatible, which shapes how the code is written: ReverseDiff reverse mode (`"RAD"`) is the main approach, with ForwardDiff (`"FAD"`) and finite differences (`"FiDi"`) used for checks. The RAD helpers live in `adrules/ReverseDiffRules.jl` (included through `utils/Utilities.jl`):
- `rd_gradient`/`rd_jacobian` take the same arguments as Zygote's `gradient`/`jacobian` and return results in the same form; scalar inputs are allowed.
- Mutating arrays is fine, but allocate work arrays with `ad_buffer(x, deps...)`, `zeros(ad_eltype(deps...), ...)` or `ad_complextype(...)` so they can hold `TrackedReal`. A tracked value written into a `Float64` array throws an error.
- `ChainRulesCore.ignore_derivatives(x)` does **not** cut derivatives under ReverseDiff; use `nograd(x)`. The `do`-block form is still fine for printing and I/O.
- ReverseDiff ignores ChainRules `rrule`s, so call `rd_rrule(f, args...)` to use `f`'s rrule. This is required for anything that goes through LAPACK or complex special functions (eigen solves, `\`, `besselk(ν, im*k)`).
- ReverseDiff is real-only, so complex linear algebra on the derivative path is split into real/imag parts (for example, the forced solve uses the block system `[D_r -D_i; D_i D_r]`).
- The drivers pass plain arrays of `TrackedReal` (scalar-mode tracking), not `TrackedArray`. ReverseDiff's `TrackedArray` paths break on complex values and abstract eltypes. Avoid `zeros(Real, ...)`, `zeros(RealOrComplex, ...)`, `Vector{Number}` and `::Vector{RealOrComplex}`-typed locals on the derivative path. Also avoid `vcat` with a bare tracked scalar: write `vcat([x], v)`, not `vcat(x, v)`.
- Unlike Zygote, ReverseDiff runs the reverse pass through every recorded operation, even ones whose result is unused or only drives control flow. In reverse 0·Inf = NaN, so `sqrt(0)`, `abs` of a complex zero, or a `0/0` that is later replaced will poison the whole gradient. Wrap control-flow-only and output-only values in `nograd`, and use a safe division where a denominator can be zero.

Keep functions generic over `RealOrComplex`/`DTYPE` (from `constants/DataTypes.jl`) so complex-step and dual numbers still work.

**Outputs.** Solvers write to an output directory given in `solverOptions`. `io/TecplotIO.jl` and `io/ParaviewIO.jl` (VTK) write the geometry and solution files. `OUTPUT/`, `*.jld2`, `*.hst` and `*.html` (OpenMDAO N2 diagrams) are gitignored artifacts.

## Conventions

- Source files start with a header docstring (`@File`, `@Date created`, `@Author`, `@Desc`) and use `# ====` / `# ****` banner comments to split sections.
- Spatial indices use the `XDIM`/`YDIM`/`ZDIM` constants. Units are SI, and angles are in radians unless a comment says `[deg]`.
- In the Python tests, `juliacall` has to be imported before `jax`, or the process crashes.
