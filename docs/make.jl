using DCFoil
using Documenter

DocMeta.setdocmeta!(DCFoil, :DocTestSetup, :(using DCFoil); recursive=true)

makedocs(;
    modules=[DCFoil],
    authors="Galen Ng <nggw@umich.edu>",
    sitename="DCFoil.jl",
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
    repo="github.com/mdolab/DCFoil.jl.git",
    devbranch="main",
)
