push!(LOAD_PATH, "../src/")

using Documenter
include("../src/DCFoil.jl")
using .DCFoil

DocMeta.setdocmeta!(DCFoil, :DocTestSetup, :(using DCFoil); recursive=true)

makedocs(;
    modules=[DCFoil],
    authors="Galen Ng <nggw@umich.edu>",
    sitename="DCFoil.jl",
    format=Documenter.HTML(;
        canonical="https://mdolab.github.io/DCFoil.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mdolab/DCFoil.jl.git",
    devbranch="main",
)
