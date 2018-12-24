"""
    struct TabularV <: AbstractVApproximator{Int}
        table::Vector{Float64}
    end

Using a `table` of type `Vector{Float64}` to record the state values.
"""
struct TabularV <: AbstractVApproximator{Int}
    table::Vector{Float64}
end

TabularV(ns::Int, init::Float64=0.) = TabularV(fill(init, ns))

(VLearner::TabularV)(s::Int) = VLearner.table[s]

update!(VLearner::TabularV, s::Int, e::Float64) = VLearner.table[s] += e
