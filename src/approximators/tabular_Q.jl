using StatsBase:sample

"""
    struct TabularQ <: AbstractQApproximator{Int, Int}
        table::Array{Float64, 2}
    end

Using a `table` of type `Array{Float64,2}` to record the action value of each state.
"""
struct TabularQ <: AbstractQApproximator{Int, Int}
    table::Array{Float64, 2}
end

"""
    TabularQ(ns::Int, na::Int=1, init::Float64=0.)

Initial a table of size `(ns, na)` filled with value of `init`.
"""
TabularQ(ns::Int, na::Int=1, init::Float64=0.) = TabularQ(fill(init, ns, na))

(Q::TabularQ)(s::Int, ::Val{:dist}) = Q.table[s, :]
(Q::TabularQ)(s::Int, ::Val{:max}) = @views findmax(Q.table[s, :])[1]
(Q::TabularQ)(s::Int, ::Val{:argmax}) = @views sample(findallmax(Q.table[s, :])[2])
(Q::TabularQ)(s::Int, a::Int) = Q.table[s, a]
(Q::TabularQ)(s::Int) = Q(s, Val(:dist))

"""
!!! warning
    Without bound check! Unsafe!
"""
function update!(Q::TabularQ, s::Int, errors::Vector{Float64})
    for (i, e) in enumerate(errors)
        Q.table[s, i] += e
    end
end

update!(Q::TabularQ, s::Int, a::Int, e::Float64) = Q.table[s, a] += e