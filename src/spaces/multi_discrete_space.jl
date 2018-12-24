"""
    struct MultiDiscreteSpace{N} <:AbstractDiscreteSpace
        counts::Array{Int, N}
    end

The element in `MultiDiscreteSpace{N}` is a multi-dimension array. 
The number of each dimension is specified by `counts`.
"""
struct MultiDiscreteSpace{N} <:AbstractDiscreteSpace
    counts::Array{Int, N}
end

size(s::MultiDiscreteSpace) = Tuple(s.counts)
eltype(s::MultiDiscreteSpace) = typeof(s.counts)
sample(s::MultiDiscreteSpace) = map(x -> rand(1:x), s.counts)
==(x::MultiDiscreteSpace, y::MultiDiscreteSpace) = x.counts == y.counts
in(xs::Array{Int}, s::MultiDiscreteSpace) = all(map((e, x) -> 1 ≤ x < e + 1 , s.counts, xs))
in(xs::NTuple{N, Int}, s::MultiDiscreteSpace{N}) where N = all(map((e, x) -> 1 ≤ x < e + 1 , s.counts, xs))