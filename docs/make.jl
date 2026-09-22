using Documenter
using DCFoil

DocMeta.setdocmeta!(DCFoil, :DocTestSetup, :(using DCFoil); recursive=true)

makedocs(;
    modules=[DCFoil],
    authors="Galen Ng <nggw@umich.edu>",
    sitename="DCFoil.jl",
    # Only exported names must appear in the manual. The default (:all) fails because
    # src/hydro/VPM.jl and src/hydro/OceanWaves.jl are `include`d both into DCFoil and
    # into the LiftingLine / SolveForced submodules, leaving unexported submodule
    # docstrings that no @autodocs block renders.
    checkdocs=:exports,
    format=Documenter.HTML(;
        canonical="https://gawng.github.io/DCFoil.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gawng/DCFoil.jl.git",
    devbranch="main",
)
