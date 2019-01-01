"""
    WeightedSample <: AbstractActionSelector
    WeightedSample()
"""
struct WeightedSample <: AbstractActionSelector end

"""
    (p::WeightedSample)(values::AbstractArray)

!!! note
    Action `values` are normalized to have a sum of 1.0
    and then used as the probability to sample a random action.
"""
(p::WeightedSample)(values::AbstractArray) = sample(weights(values))