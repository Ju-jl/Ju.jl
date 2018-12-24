using LinearAlgebra:dot

"""
    struct LinearV <: AbstractVApproximator{Int}
        features::Array{Float64, 2}
        weights::Vector{Float64}
    end

Using a matrix `features` to represent each state along with a vector of `weights`.

See more details at Section (9.4) on Page 205 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct LinearV <: AbstractVApproximator{Int}
    features::Array{Float64, 2}
    weights::Vector{Float64}
end

(learner::LinearV)(s::Int) = @views dot(learner.features[s, :], learner.weights)

function update!(learner::LinearV, s::Int, e::Float64)
    for i in 1:length(learner.weights)
        learner.weights[i] += learner.features[s, i] * e
    end
end