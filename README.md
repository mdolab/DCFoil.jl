
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
Alternatively, use this code as a Python package with
```
pip install -e .
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

### Helpful macros
Use `@time` to time lines of code (accounts for garbage collection too).
`t = @elapsed` is OK
`@show`

### Conventions

Please use this coding convention:

* `camelCase` - variables
* `PascalCase` - modules, module filenames, and structs
* `snake_case` - functions (all functions should contain a verb) and non-module filenames
* `SCREAMING_SNAKE_CASE` - constants

### Data types

Only use **parametric** types for structs. Concrete types in function arguments do not usually make the code faster; they only restrict usage (unless it is on purpose for multiple dispatch). However, you should declare the struct type argument in function signatures. The `DTYPE` constant in the code `RealOrComplex` should be used when structs are not used in the function
https://docs.julialang.org/en/v1/manual/performance-tips/#Type-declarations

Data typing code that you will AD is very tricky.
You do not want to be too specific.

### Derivatives

#### Adding new cost functions

* For the given solver you're adding the cost func to, check its `<solver-name>.evalFuncsSens()`

#### Try not to do this

* Use `RealOrComplex` sparingly for when complex-step derivatives are needed
* NOTE: as of February 24, 2024, `LinRange` is actually better and improves the flutter prediction accuracy.
I wrote the custom rule with the help of the Julia slack channel.
* Careful of mutating operations because Julia is pass by reference! Use `copy()`
* Mutating arrays that require the `Zygote.Buffer` data type. It is SUPER slow.
* Don't use `ForwardDiff` because it cannot do matrix operations and I haven't figured out the chain rules.
It also doesn't fit with the data types.
* Unicode characters are nice for readability of math-heavy code, but do not use them for interface-level code
* `hypot()` function for calculating the L2 norm is slower than typing it out
* `@ignore_derivatives` macro from `ChainRulesCore` sometimes messes with the scope of operations. `ChainRulesCore.ignore_derivatives()` is better.

#### AD Packages

* `AbstractDifferentiation` is a wrapper level tool
* `Zygote` is an RAD package
* `ReverseDiff` is another RAD package that is more performant but restrictive on data types

### Performance

* Don't add type annotations for function arguments unless for multiple dispatch
* Don't do ```zeros(n)```, but rather ```zeros(typeof(x), n)``` or better ```::Abstract<type> = zeros()``` so it works with dual numbers.
* Make use of `similar()` to make uninitialized arrays or `typeof()`

### DCFoil as a package

The ```Project.toml``` means this is a Julia package and can be added with ```Pkg.add("DCFoil"); using DCFoil```. However, in development mode, just go into julia for this directory and type ```] dev .```.

### Package Dependencies and Updating

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

When there is a new version of Julia and you use `juliaup update`, make sure to `Pkg.update()`.


#### Python dependencies from the MACH framework

<!-- We use `PyCall` to use some modules from MACH, but this highly depends on what Python environment you are using.
If you're building `PyCall` for the first time, it depends on the PYTHON environment variable so if you build with the wrong python, don't forget to do a clean uninstall before rebuilding.
Once the package is built, the `venv` method of getting this to work requires running this in the Julia REPL
```
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
```
before the `using PyCall` import in the julia scripts, but I put this in the scripts anyways.

For a Linux / Docker workflow (preferred), it is as simple as running
```
import julia
julia.install()
```
in the Python REPL. You need to do this everytime you update Julia -->

`PythonCall` is the package for wrapping the Julia code.
We opted for this over PyCall because it's what's used in the OpenMDAO examples.

The package `juliapkg` handles the Julia packages.
A juliapkg.json file is in your python environment (e.g., venv or conda).
For the first time running, run the `setup_juliapkg.py` script.
For any other problems arising, look at the [juliapkg documentation](https://github.com/JuliaPy/PyJuliaPkg)

The list of dependencies is:
```
baseclasses
pyspline
prefoil
pygeo
```

You will need to set this up in a Linux environment because of the above dependencies. Sorry mac users.

### Tests

Under the `./test/` directory, run
```
run_tests.jl
```

### Debugging code

Use `@enter` from the `Debugger.jl` to step into functions and `@bp` to put breakpoints.
Wrap certain code in `@run` to put the execution in debug mode to get the breakpoints.
You can also use `@show`, `@debug` commands for pesky bugs.

Debugging AD code is a bit trickier.
`Cthulhu.jl` is a package that offers the `@descend` macro that helps step into your code.
`ascend` is also helpful and can be used on stacktraces.

## Citation

For more, see the [journal paper](https://doi.org/10.1016/j.compstruct.2024.118367). Please cite this article when using DCFoil in your research or curricula.

Ng, Galen W., Eirikur Jonsson, Sicheng He, and Joaquim RRA Martins. "Dynamic hydroelasticity of composite appendages with reverse-mode algorithmic differentiation." Composite Structures 346 (2024): 118367.

```
@Article{Ng2024,
    author      = {Galen W. Ng and Eirikur Jonsson and Sicheng He and Joaquim R.R.A. Martins},
    title       = {Dynamic hydroelasticity of composite appendages with reverse-mode algorithmic differentiation},
    doi         = {10.1016/j.compstruct.2024.118367},
    journal     = {Composite Structures},
    year        = {2024}}
```