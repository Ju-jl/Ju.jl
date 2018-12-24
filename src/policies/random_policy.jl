using StatsBase:sample, Weights

"""
    RandomPolicy(prob::Array{Float64, 2})
    RandomPolicy(prob::Vector{Float64})

The probability of each action is predefined by `prob`.
If `prob` is a vector, then all states share the same `prob`.
"""
struct RandomPolicy{n} <: AbstractPolicy
    prob::Array{Float64, n}
    RandomPolicy(prob::Array{Float64, 2}) = new{2}(prob)
    RandomPolicy(prob::Vector{Float64}) = new{1}(prob)
end

(p::RandomPolicy{2})(s::Int) = sample(Weights(p.prob[s, :], 1.0))
(p::RandomPolicy{2})(s::Int, ::Val{:dist}) = p.prob[s, :]
(p::RandomPolicy{2})(s::Int, a::Int) = p.prob[s, a]

(p::RandomPolicy{1})(s) = sample(Weights(p.prob, 1.0))
(p::RandomPolicy{1})(s, ::Val{:dist}) = p.prob
(p::RandomPolicy{1})(s, a::Int) = p.prob[a]