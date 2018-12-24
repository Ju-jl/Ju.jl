export UpperConfidenceBound, EpsilonGreedySelector, WeightedSample, AlternateSelector 

using StatsBase:weights, sample

include("epsilon_greedy.jl")
include("upper_confidence_bound.jl")
include("weighted_sample.jl")
include("alternate.jl")