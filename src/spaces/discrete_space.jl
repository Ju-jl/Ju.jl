"""
    struct DiscreteSpace <: AbstractDiscreteSpace
        n::Int
    end

The elements in a `DiscreteSpace` is `1:n`
"""
struct DiscreteSpace <: AbstractDiscreteSpace
    n::Int
end

size(d::DiscreteSpace) = (d.n,)
eltype(d::DiscreteSpace) = Int
sample(d::DiscreteSpace) = rand(1 : d.n)
in(x::Int, d::DiscreteSpace) = 1 ≤ x ≤ d.n
==(x::DiscreteSpace, y::DiscreteSpace) = x.n == y.n