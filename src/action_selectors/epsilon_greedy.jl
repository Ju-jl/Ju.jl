"""
    EpsilonGreedySelector <: AbstractActionSelector
    EpsilonGreedySelector(ϵ)
    
The best action is selected for a proportion ``1 - \\epsilon``
and a random action (with uniform probability) is selected for a proportion ``\\epsilon``.

## Example

```julia
julia> selector = EpsilonGreedySelector(0.1)
EpsilonGreedySelector(0.1)

julia> countmap(selector([1,2,1,1]) for _ in 1:1000)
Dict{Any,Int64} with 4 entries:
  4 => 37
  2 => 915
  3 => 22
  1 => 26
```
"""
mutable struct EpsilonGreedySelector <: AbstractActionSelector
    ϵ::Float64
end


"""
    (p::EpsilonGreedySelector)(values::AbstractArray{T, 1}) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
(p::EpsilonGreedySelector)(values::AbstractArray{<:Number, 1}) = rand() > p.ϵ ? sample(findallmax(values)[2]) : rand(1:length(values))