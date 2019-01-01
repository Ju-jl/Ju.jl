export AbstractSpace, AbstractContinuousSpace, AbstractDiscreteSpace,
       sample

import StatsBase:sample
import Base:in, size, ==, length, eltype

"""
    AbstractSpace

Supertype of [`AbstractContinuousSpace`](@ref) and [`AbstractDiscreteSpace`](@ref).
"""
abstract type AbstractSpace end

"""
    AbstractContinuousSpace <: AbstractSpace

Supertype of different kinds of continuous spaces.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `sample(space)` | Get a random sample from the space |
| `Base.in(space, x)` | Test whether `x` is in the `space` |
| **Optional Methods** | |
| `Base.eltype(space)` | Return the type of the sample in a space |

See also: [`AbstractDiscreteSpace`](@ref)
"""
abstract type AbstractContinuousSpace <: AbstractSpace end

"""
    AbstractDiscreteSpace <: AbstractSpace

Supertype of different kinds of discrete spaces.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `sample(space)` | Get a random sample from the space |
| `Base.in(space, x)` | Test whether `x` is in the `space` |
| `Base.size(space)` | Return the size of the space in all dimensions |
| **Optional Methods** | |
| `Base.length(space)` | Return the number of elements in that space. By default it will be `*(Base.size(space))`. |
| `Base.eltype(space)` | Return the type of the sample in a space |

See also: [`AbstractContinuousSpace`](@ref)
"""
abstract type AbstractDiscreteSpace <: AbstractSpace end

length(s::AbstractDiscreteSpace) = *(size(s)...)

# Tuple Support
sample(s::Tuple{Vararg{<:AbstractSpace}}) = map(sample, s)
in(a::Tuple{Vararg{T, N} where T}, b::Tuple{Vararg{<:AbstractSpace, N}}) where N = all(map((x, y) -> in(x, y), a, b))

# Dict Support
sample(s::Dict{String}) = Dict(map((k, v) -> (k, sample(v)), s))
in(a::Dict{String}, b::Dict{String}) = length(a) == length(b) &&
    all(p -> haskey(a, p.first) ? in(a[p.first], p.second) : false,
        b)