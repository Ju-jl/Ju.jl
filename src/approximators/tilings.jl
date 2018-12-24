import Base:length, -

"""
    Tiling(ranges::NTuple{N, Tr}) where {N, Tr}

Using a tuple of `ranges` to simulate a tiling.

# Example
```julia
julia> t = Tiling((1:2:5, 10:5:20))
Tiling{2,StepRange{Int64,Int64}}((1:2:5, 10:5:20), [1 3; 2 4])

julia> Ju.encode(t, (2, 12))  # encode into an Int
1

julia> Ju.encode(t, (2, 18))

julia> t2 = t - (1, 3)  # shift a little to get a new Tiling
Tiling{2,StepRange{Int64,Int64}}((0:2:4, 7:5:17), [1 3; 2 4])3
```

See also: [`TilingsV`](@ref), [`TilingsQ`](@ref)
"""
struct Tiling{N, Tr<:AbstractRange}
    ranges::NTuple{N, Tr}
    inds::LinearIndices{N, NTuple{N, Base.OneTo{Int}}}
    Tiling(ranges::NTuple{N, Tr}) where {N, Tr} = new{N, Tr}(ranges, LinearIndices(Tuple(length(r) - 1 for r in ranges)))
end

"""
    (-)(t::Tiling, xs)

Shift `t` along each dimension by each element in `xs`.
"""
function (-)(t::Tiling, xs)
    Tiling(Tuple(r .- x for (r,x) in zip(t.ranges, xs)))
end

length(t::Tiling) = reduce(*, (length(r)-1 for r in t.ranges))

encode(range::AbstractRange, x) = floor(Int, div(x-range[1], step(range)) + 1)
encode(t::Tiling{1}, x::Number) = encode(t.ranges[1], x)
encode(t::Tiling{1}, xs) = encode(t.ranges[1], xs[1])
encode(t::Tiling{2}, xs) = t.inds[CartesianIndex(encode(t.ranges[1], xs[1]), encode(t.ranges[2], xs[2]))]
encode(t::Tiling{3}, xs) = t.inds[CartesianIndex(encode(t.ranges[1], xs[1]), encode(t.ranges[2], xs[2]), encode(t.ranges[3], xs[3]))]

"""
    TilingsV{Tt<:Tiling} <: AbstractVApproximator{Vector{Float64}}
    TilingsV(tilings::Vector{Tt}) where Tt<:Tiling

Using a vector of `tilings` to encode state. Each tiling has an independent weight.

See more details at Section (9.5.4) on Page 217 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*

See also: [`Tiling`](@ref), [`TilingsQ`](@ref)
"""
struct TilingsV{Tt<:Tiling} <: AbstractVApproximator{Vector{Float64}}
    tilings::Vector{Tt}
    weights::Vector{Vector{Float64}}
    TilingsV(tilings::Vector{Tt}) where Tt<:Tiling = new{Tt}(tilings, [zeros(Float64, length(t)) for t in tilings])
end

function (ts::TilingsV)(s)
    v = 0.
    for (w, t) in zip(ts.weights, ts.tilings)
        v += w[encode(t, s)]
    end
    v
end

function update!(ts::TilingsV, s, e)
    for i in 1:length(ts.tilings)
        ts.weights[i][encode(ts.tilings[i], s)] += e
    end
end


"""
    TilingsQ{Tt<:Tiling} <: AbstractQApproximator{Vector{Float64}, Int}
    TilingsQ(tilings::Vector{Tt}, nactions) where Tt<:Tiling

The only difference compared to [`TilingsV`](@ref) is that now the weight of each tiling is a matrix.
"""
struct TilingsQ{Tt<:Tiling} <: AbstractQApproximator{Vector{Float64}, Int}
    tilings::Vector{Tt}
    weights::Vector{Array{Float64, 2}}
    TilingsQ(tilings::Vector{Tt}, nactions) where Tt<:Tiling = new{Tt}(tilings, [zeros(Float64, length(t), nactions) for t in tilings])
end

function (ts::TilingsQ)(s, a)
    v = 0.
    for (w, t) in zip(ts.weights, ts.tilings)
        v += w[encode(t, s), a]
    end
    v
end

function (ts::TilingsQ)(s, ::Val{:dist}) 
    dist = zeros(Float64, size(ts.weights[1],2))
    for (w,t) in zip(ts.weights, ts.tilings)
        dist .+= @view w[encode(t,s), :]
    end
    dist
end

(t::TilingsQ)(s) = t(s, Val(:dist))
(t::TilingsQ)(s, ::Val{:max}) = findmax(t(s))[1]
(t::TilingsQ)(s, ::Val{:argmax}) = sample(findallmax(t(s))[2])

function update!(ts::TilingsQ, s, a, e)
    for (w, t) in zip(ts.weights, ts.tilings)
        w[encode(t, s), a] += e
    end
end