"""
    MultiContinuousSpace(low::Number, high::Number, size::Tuple{Vararg{Int}})
    MultiContinuousSpace(low::Array{<:Number}, high::Array{<:Number})

# Examples
```julia-repl
MultiContinuousSpace(-1, 1, (2,3))
MultiContinuousSpace([0, 0, 0], [1, 2, 3])
```
"""
struct MultiContinuousSpace{T <: Number,N} <: AbstractContinuousSpace
    low::Array{T,N}
    high::Array{T,N}
    MultiContinuousSpace(low::Number, high::Number, size::Tuple{Vararg{Int}}) = MultiContinuousSpace(fill(low, size), fill(high, size))
    function MultiContinuousSpace(low::Array{T, N}, high::Array{T, N}) where {T, N}
        size(low) != size(high) && throw(DimensionMismatch("size of low and high must match!"))
        new{T, N}(low, high)
    end
end


eltype(s::MultiContinuousSpace{T, N}) where {T, N} = Array{T, N}
==(x::MultiContinuousSpace, y::MultiContinuousSpace) = x.low == y.low && x.high == y.high
sample(s::MultiContinuousSpace) = map((l, h) -> l + rand() * (h - l), s.low, s.high)
in(xs::AbstractArray{T, N} where T, s::MultiContinuousSpace{T, N} where T) where N = all(map((a, b, c) -> a ≤ b ≤ c, s.low, xs, s.high))