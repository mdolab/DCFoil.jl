
# DCFoil.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gawng.github.io/DCFoil.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gawng.github.io/DCFoil.jl/dev/)
[![Build Status](https://github.com/gawng/DCFoil.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gawng/DCFoil.jl/actions/workflows/CI.yml?query=branch%3Amain)


![Alt text](./media/logo.svg "logo-text")

Dynamic Composite Foil (DCFoil) in Julia

## Versions

<!-- We test for `macOS-latest` and `Ubuntu-latest`. -->
Check the Project.toml for version dependencies.

## Package

The code can be added with 
```
Pkg.add("DCFoil"); using DCFoil
```

## Developers Notes

### Get started as a developer

Start your docker container if using MDOLab codes.
You must be in this root directory. From the terminal you can type

```
julia --project=.
```

and in the Julia REPL, type

```
] activate .
```

and all the dependencies in the `Project.toml` should be queued to compile at runtime.
You can double-check that all the right packages are present with `status` in the Pkg REPL.
You may need to type into the Pkg REPL

`instantiate`

Now you are ready to run the solver in the Julia REPL with

`include("main.jl")`

Since this is JIT, it will be slow the first time because it needs to compile stuff first.
You can alternatively run the code from the terminal with

`julia main.jl`

NOTE:
`run_main.sh` is a convenience script for the above. Running julia using shell script `nohup` does not work but a regular `nohup` command is fine

### Conventions

Please use this coding convention:

* `camelCase` - variables
* `PascalCase` - modules, module filenames, and structs
* `snake_case` - functions (all functions should contain a verb) and non-module filenames
* `SCREAMING_SNAKE_CASE` - constants

### Data types

Only use **parametric** types for structs.
Concrete types in function arguments do not usually make the code faster; 
they only restrict usage (unless it is on purpose for multiple dispatch).
However, you should declare the struct type argument in function signatures
The DTYPE constant in the code `RealOrComplex` should be used when structs are not used in the function
https://docs.julialang.org/en/v1/manual/performance-tips/#Type-declarations

### Derivatives

#### Adding new cost functions or design variables

* For the given solver you're adding the DV or cost func to, check its `<solver-name>.evalFuncsSens()`

#### Try not to do this

* NOTE: as of February 24, 2024, `LinRange` is actually better and improves the flutter prediction accuracy. 
I wrote the custom rule with the help of the Julia slack channel.
<!-- `LinRange()` because it isn't easily differentiated. Do something like `collect((start:step:end))`  -->
* Mutating arrays that require the `Zygote.Buffer` data type. It is SUPER slow.
* Don't use `ForwardDiff` because it cannot do matrix operations and I haven't figured out the chain rules.
It also doesn't fit with the data types
* Unicode characters are nice for readability of math-heavy code, but do not use them for interface-level code
* `hypot()` function for calculating the L2 norm is slower than typing it out

#### AD Packages

* `AbstractDifferentiation` is a wrapper level tool
* `Zygote` is an RAD package

### Performance

* Don't add type annotations for function arguments unless for multiple dispatch
* Don't do ```zeros(n)```, but rather ```zeros(typeof(x), n)```

### DCFoil as a package

The Project.toml means this is a Julia package and can be added with ```Pkg.add("DCFoil"); using DCFoil```
However, in development mode, just go into julia for this directory and type ```] dev .```.

### Package Dependencies

Add package dependencies in the REPL with

```
using Pkg
Pkg.add("package-name", preserve=PRESERVE_DIRECT)
```

to keep the version of the package static.

To update all dependencies (be careful with this in case it breaks tests)

```
Pkg.update()
```

Use `Pkg.rm("<module-name>")` to remove a package.

NOTE: If this screws everything up, you can use `Pkg.undo()` to undo the last change (also see `help <your-command>`).
Chances are if there is a Pkg bug, it has to do with python, which actually is not necessary.


#### Python dependencies from the MACH framework

We use `PyCall` to use some modules from MACH, but this highly depends on what Python environment you are using.
If you're building `PyCall` for the first time, it depends on the PYTHON environment variable so if you build with the wrong python, don't forget to do a clean uninstall before rebuilding.
Once the package is built, the `venv` method of getting this to work requires
```
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
```
before the `using PyCall` import in the julia scripts, but I put this in the scripts anyways.
I have only gotten the Conda.jl method to work which requires these runs
```
Conda.pip_interop(true, Conda.ROOTENV) # allow pip installation
Conda.pip("install", ["<package-names>"], Conda.ROOTENV) # generic call to pip install a package
```
and for package names, you can install any python package that supports pip installation.
Unfortunately, you would have to reinstall all of MACH's modules if you do not use conda environment management.
The list is:
```
pyspline
pygeo
```

<!-- The MACH2DCFoil wrapper requires:
```
pip install julia
```
to install the pyjulia package and then in a python prompt
```
import julia
julia.install("<your-version>") # if multiple versions of julia are installed
``` -->

### Tests

Under the `./test/` directory, run
```
run_tests.jl
```

## Citation

TODO: paper links
For more, see the formal documentation <> and journal paper