# dcfoil-jl
Dynamic Composite Foil (DCFoil) in Julia

## Get started
You must be in this root directory. From the terminal you can type
`julia --project=.`
OR
in the Julia REPL, type 
`] activate .`
and all the dependencies in the `Project.toml` should be queued to compiled at runtime. 
You can double check that all the right packages are present with `status` in the Pkg REPL.
You may need to type into the Julia REPL
`Pkg.instantiate()`

Now you are ready to run the solver with 
`include("main.jl")`
Since this is JIT, it will be slow the first time because it needs to compile stuff first.

For more, see the formal documentation (TODO:) and journal paper (TODO:)
## Developers Notes

### Convention
Please use this coding convention:

camelCase - variables
PascalCase - modules and module filenames
snake_case - functions (all functions should contain a verb) and non-module filenames
SCREAMING_SNAKE_CASE - constants

### Dependencies

Add dependencies in the REPL with
`Using Pkg`
`Pkg.add("package-name", preserve=PRESERVE_DIRECT)`
to keep the version of the package static.

To update all dependencies (be careful with this in case it breaks tests TODO)
`Pkg.update(level=UPLEVEL_PATCH)` 
NOTE: If this screws everything up, you can use `Pkg.undo()` to undo the last change (also see `help <your-command>`).

### Tests

Run the `run_tests.jl`. TODO: actually use the julia unittest package