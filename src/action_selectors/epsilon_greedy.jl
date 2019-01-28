"""
    EpsilonGreedySelector <: AbstractActionSelector
    EpsilonGreedySelector(ϵ)
    
The best action is selected for a proportion ``1 - \\epsilon``
and a random action (with uniform probability) is selected for a proportion ``\\epsilon``.
ϵ can also be a decay. See the following examples.

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

julia> ϵ = exp_decay(init=1.0, λ=1.0, decay_step=500, clip=0.1)
(::getfield(Ju, Symbol("#f#34")){Float64,Float64,Int64,Float64}) (generic function with 1 method)

julia> selector = EpsilonGreedySelector(ϵ)
EpsilonGreedySelector{getfield(Ju, Symbol("#f#34")){Float64,Float64,Int64,Float64}}(getfield(Ju, Symbol("#f#34")){Float64,Float64,Int64,Float64}(1.0, 1.0, 500, 0.1, Core.Box(-1)))

julia> countmap(selector([1,2,1,1]) for _ in 1:1000)
Dict{Any,Int64} with 4 entries:
  4 => 101
  2 => 677
  3 => 106
  1 => 116
"""
mutable struct EpsilonGreedySelector{T<:Union{Float64, <:Function}} <: AbstractActionSelector
    ϵ::T
end

_epsilon_select(values, ϵ) = rand() > ϵ ? sample(findallmax(values)[2]) : rand(1:length(values))

"""
    (p::EpsilonGreedySelector)(values::AbstractArray{T, 1}) where T

!!! note
    If multiple values with the same maximum value are found.
    Then a random one will be returned!

    `NaN` will be filtered unless all the values are `NaN`.
    In that case, a random one will be returned.
"""
(p::EpsilonGreedySelector{Float64})(values) = _epsilon_select(values, p.ϵ)

(p::EpsilonGreedySelector{<:Function})(values) = _epsilon_select(values, p.ϵ())

# TODO: add `dims` argument to support higher dimension?