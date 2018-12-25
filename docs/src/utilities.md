# Utilities

```@contents
Pages = ["utilities.md"]
```

## Core

The most widely used functions while solving RL problems are listed below:

```@autodocs
Modules = [Ju]
Pages = ["core.jl"]
```

## Callbasks

The callbacks are used in [`train!`](@ref). Here are some of the predefined callbacks:

```@autodocs
Modules = [Ju]
Pages = ["callbacks.jl"]
```

## Decays

Decays are used to efficiently calculate some variables which change based on time step.

```@autodocs
Modules = [Ju]
Pages = ["decays.jl"]
```

## Iterators

Some iterators are very helpful while implementing traditional RL algorithms.

```@autodocs
Modules = [Ju]
Pages = ["iterators.jl"]
```

## Helper Functions

Following are some commonly used functions.

```@autodocs
Modules = [Ju]
Pages = ["helper_functions.jl"]
```

## Others

```@docs
CircularArrayBuffer
Tiling
```