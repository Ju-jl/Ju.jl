"""
    struct AggregationV{Tf<:Function} <: AbstractVApproximator{Int}
        table::Vector{Float64}
        f::Tf
    end

Using `a.f` to map a state `s` into an `Int`, then use `a.table` to check the corresponding state value.
"""
struct AggregationV{Tf<:Function} <: AbstractVApproximator{Int}
    table::Vector{Float64}
    f::Tf
end

(a::AggregationV)(s) = a.table[a.f(s)]

update!(a::AggregationV, s, e) = a.table[a.f(s)] += e