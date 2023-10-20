# DCFoil.jl

![Alt text](./media/logo.svg "logo-text")

Dynamic Composite Foil (DCFoil) in Julia

## Versions

We test for `macOS-latest` and `Ubuntu-latest`.

v1.7.3 (stable)
v1.9.0 (latest)

## Get started

You must be in this root directory. From the terminal you can type

```
julia --project=.
```

or in the Julia REPL, type

```
] activate .
```

and all the dependencies in the `Project.toml` should be queued to compile at runtime.
You can double-check that all the right packages are present with `status` in the Pkg REPL.
You may need to type into the Pkg REPL

`instantiate()`

Now you are ready to run the solver in the Julia REPL with

`include("main.jl")`

Since this is JIT, it will be slow the first time because it needs to compile stuff first.
You can alternatively run the code from the terminal with

`julia main.jl`

NOTE:
`run_main.sh` is a convenience script for the above. Running julia using shell script `nohup` does not work but a regular `nohup` command is fine


## Developers Notes

### Conventions

Please use this coding convention:

* `camelCase` - variables
* `PascalCase` - modules and module filenames
* `snake_case` - functions (all functions should contain a verb) and non-module filenames
* `SCREAMING_SNAKE_CASE` - constants

### Sensitivites

#### Adding new cost functions or design variables

* For the given solver you're adding the DV or cost func to, check its `.evalFuncsSens()`
#### Try not to do this

* `LinRange()` because it isn't easily differentiated. Do something like `start:step:end`
* Mutating arrays

### Package Dependencies

Add package dependencies in the REPL with

```
using Pkg
Pkg.add("package-name", preserve=PRESERVE_DIRECT)
```

to keep the version of the package static.

To update all dependencies (be careful with this in case it breaks tests)

```
Pkg.update(level=UPLEVEL_PATCH)
```

Use `Pkg.rm("<module-name>")` to remove a package.

NOTE: If this screws everything up, you can use `Pkg.undo()` to undo the last change (also see `help <your-command>`).
Chances are if there is a Pkg bug, it has to do with python, which actually is not necessary.


#### Python dependencies from the MACH framework

We use `PyCall` to use some modules from MACH, but this highly depends on what Python environment you are using.
<!-- If you're building `PyCall` for the first time, it depends on the PYTHON environment variable so if you build with the wrong python, don't forget to do a clean uninstall before rebuilding.
Once the package is built, the `venv` method of getting this to work requires
```
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
```
before the `using PyCall` import in the julia scripts, but I put this in the scripts anyways. -->
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
### Tests

Under the `./test/` directory, run
```
run_tests.jl
```

## Citation
TODO: paper links
For more, see the formal documentation <> and journal paper