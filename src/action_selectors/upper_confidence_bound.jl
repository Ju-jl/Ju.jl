"""
    UpperConfidenceBound <: AbstractActionSelector
    UpperConfidenceBound(na, c=2.0, t=0)

# Arguments
- `na` is the number of actions used to create a internal counter.
- `t` is used to store current time step.
- `c` is used to control the degree of exploration.
"""
mutable struct UpperConfidenceBound <: AbstractActionSelector
    c::Float64
    t::Int
    actioncounts::Vector{Float64}
    UpperConfidenceBound(na, c=2.0, t=0) = new(c, t, fill(1e-10, na))
end

@doc raw"""
    (ucb::UpperConfidenceBound)(values::AbstractArray)

Unlike [`EpsilonGreedySelector`](@ref), uncertaintyies are considered in UCB.

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

```math
A_t = \underset{a}{\arg \max} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
```

See more details at Section (2.7) on Page 35 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function (p::UpperConfidenceBound)(values::AbstractArray)
    action = findallmax(@. values + p.c * sqrt(log(p.t+1) / p.actioncounts))[2] |> sample
    p.actioncounts[action] += 1
    p.t += 1
    action
end