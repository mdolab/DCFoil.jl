
# DCFoil.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gawng.github.io/DCFoil.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gawng.github.io/DCFoil.jl/dev/)
[![Build Status](https://github.com/gawng/DCFoil.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/gawng/DCFoil.jl/actions/workflows/CI.yml?query=branch%3Amain)


![Alt text](./media/logo.svg "logo-text")

Copyright (C) 2023-2025 The Regents of the University of Michigan
All rights reserved.

Dynamic Composite Foil (DCFoil.jl) is a composite marine appendage design optimization software written in Julia.
It analyzes the fluid-structure interaction of slender composite structures using low-order models.

## Documentation

See the `docs/` directory to build the documentation.

## Contributors

Galen W. Ng (primary developer)
Shugo Kaneko (load and displacement transfer scheme)
Eirikur Jonsson (technical advice)
Sicheng He (technical advice)
Joaquim R. R. A. Martins (technical advice)

## Versions

Check the Project.toml for version dependencies.

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