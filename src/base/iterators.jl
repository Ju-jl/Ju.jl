export reductions, Reductions, is_first_visit, IsFirstVisit, countmap

import Base:iterate, length, size, eltype
import StatsBase:countmap

"extend the countmap in StatsBase to support general iterator"
function countmap(iter)
    res = Dict{eltype(iter), Int}()
    for x in iter
        if haskey(res, x)
            res[x] += 1
        else
            res[x] = 1
        end
    end
    res
end

"""
    struct Reductions{T<:Union{NamedTuple{(:init,)}, NamedTuple{()}}, F, I}
        op::F
        itr::I
        kw::T
    end

Construct an iterator to get all the intermediate values while calling `reduce(op, itr;kw...)`
"""
struct Reductions{T<:Union{NamedTuple{(:init,)}, NamedTuple{()}}, F, I}
    op::F
    itr::I
    kw::T
end

Reductions(op, itr) = Reductions(op, itr, NamedTuple())

"""
    reductions(op, iter; init)

Return an Iterator of the intermediate values of the reduction
(as per reduce) of `iter` by `op`.

!!! note
    You can not apply `Iterators.reverse` to `Reductions` (due to time complexity).
    If you really want to get the reversed `Reductions`, consider `collect` first and then call `reverse!`

# Example
```julia
julia> reductions(+, [2,3,4])
3-element Array{Int64,1}:
 2
 5
 9

julia> reductions(+, [2,3,4], init=3)
Reductions{NamedTuple{(:init,),Tuple{Int64}},typeof(+),Array{Int64,1}}(+, [2, 3, 4], (init = 3,))

julia> collect(reductions(+, [2,3,4], init=3))
4-element Array{Any,1}:
  3
  5
  8
 12
```
"""
reductions(op, itr; kw...) = Reductions(op, itr, kw.data)

function iterate(r::Reductions{<:NamedTuple{()}})
    y = iterate(r.itr)
    if y === nothing
        throw(ArgumentError("reductions over an empty collection without init value is not allowed"))
    end
    (v, s) = y
    (v, (v, s))
end

function iterate(r::Reductions{<:NamedTuple{(:init,)}})
    v = r.kw.init
    (v, (v,))
end

function iterate(r::Reductions, (v,)::Tuple{Any})
    y = iterate(r.itr)
    y === nothing && return nothing
    v = r.op(v, y[1])
    (v, (v, y[2]))
end

function iterate(r::Reductions, (v, s)::Tuple{Any, Any})
    y = iterate(r.itr, s)
    y === nothing && return nothing
    v = r.op(v, y[1])
    (v, (v, y[2]))
end

length(r::Reductions{<:NamedTuple{()}}) = length(r.itr)
length(r::Reductions{<:NamedTuple{(:init,)}}) = length(r.itr) + 1
size(r::Reductions{<:NamedTuple{()}}) = size(r.itr)
Base.IteratorEltype(::Reductions{<:NamedTuple{()}}) = Base.EltypeUnknown()
Base.IteratorSize(r::Reductions{<:NamedTuple{()}}) = Base.IteratorSize(r.itr)

# TODO: handle multi-dim Arrays along different dims?
reductions(op, A::AbstractArray) = Base.collect_similar(A, Reductions(op, A))

"""
    IsFirstVisit(itr)

Return an iterator which signifies whether each element in `itr` occurs for the first time.

# Example
```julia
julia> s = [1,2,3,1,4,2,5];

julia> is_first_visit(s)
7-element Array{Bool,1}:
  true
  true
  true
 false
  true
 false
  true

julia> Iterators.reverse(IsFirstVisit(s))
Base.Iterators.Reverse{IsFirstVisit{Array{Int64,1}}}(IsFirstVisit{Array{Int64,1}}([1, 2, 3, 1, 4, 2, 5]))

julia> collect(Iterators.reverse(IsFirstVisit(s)))
7-element Array{Any,1}:
  true
 false
  true
 false
  true
  true
  true
```

!!! warning
    Although `IsFirstVisit` supports `Iterators.reverse`, you should still take care of the memory usage.
    Internally we will walk through the `itr` first and calculate the count of each unique element.
"""
struct IsFirstVisit{I}
    itr::I
end

is_first_visit(itr) = IsFirstVisit(itr)

function iterate(itr::IsFirstVisit)
    y = iterate(itr.itr)
    y === nothing && return nothing
    (v, s) = y
    (true, (push!(Set(), v), s))
end

function iterate(itr::IsFirstVisit, (ss, s))
    y = iterate(itr.itr, s)
    y === nothing && return nothing
    (v, s) = y
    (!(v in ss), (push!(ss, v), s))
end

# TODO: handle multi-dim Arrays along different dims?
is_first_visit(A::AbstractArray) = Base.collect_similar(A, IsFirstVisit(A))
length(itr::IsFirstVisit) = length(itr.itr)
size(itr::IsFirstVisit) = size(itr.itr)
eltype(itr::IsFirstVisit) = Bool
Base.IteratorSize(itr::IsFirstVisit) = Base.IteratorSize(itr.itr)

function iterate(itr::Iterators.Reverse{<:IsFirstVisit})
    y = iterate(Iterators.reverse(itr.itr.itr))
    y === nothing && return nothing
    counts = countmap(itr.itr.itr)
    (v, s) = y
    counts[v] -= 1
    counts[v] == 0 && delete!(counts, v)
    (!haskey(counts, v), (counts, s))
end

function iterate(itr::Iterators.Reverse{<:IsFirstVisit}, (counts, s))
    y = iterate(Iterators.reverse(itr.itr.itr), s)
    y === nothing && return nothing
    (v, s) = y
    counts[v] -= 1
    counts[v] == 0 && delete!(counts, v)
    (!haskey(counts, v), (counts, s))
end