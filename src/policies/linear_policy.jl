using NNlib:softmax
using StatsBase:sample, Weights

"""
    struct LinearPolicy <: AbstractPolicy
        features::Array{Float64, 3}
        weights::Vector{Float64}
    end

The probability of each action is calculate by `features` and `weights` and then normalized by softmax.
"""
struct LinearPolicy <: AbstractPolicy
    features::Array{Float64, 3}
    weights::Vector{Float64}
end

(π::LinearPolicy)(s, ::Val{:dist}) = @view(π.features[s, :, :]) * π.weights |> softmax
(π::LinearPolicy)(s) = sample(Weights(π(s, Val(:dist)), 1.0))
(π::LinearPolicy)(s, a) = π(s, Val(:dist))[a]

function update!(π::LinearPolicy, e)
    π.weights .+= e
end