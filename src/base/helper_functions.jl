export findallmax, deletefirst!, onehot,
       discounted_reward, reverse_discounted_rewards,
       importance_weight, reverse_importance_weights, huber_loss

using SparseArrays
using StatsBase:mean

"""
    findallmax(A::AbstractArray)

Like `findmax`, but all the indices of the maximum value are returned.

!!! warning
    All elements of value `NaN` in `A` will be ignored, unless all elements are `NaN`.
    In that case, the returned maximum value will be `NaN` and the returned indices will be `collect(1:length(A))`

#Examples
```julia-repl
julia> findallmax([-Inf, -Inf, -Inf])
(-Inf, [1, 2, 3])

julia> findallmax([Inf, Inf, Inf])
(Inf, [1, 2, 3])

julia> findallmax([Inf, 0, Inf])
(Inf, [1, 3])

julia> findallmax([0,1,2,1,2,1,0])
(2, [3, 5])
```
"""
function findallmax(A)
    maxval = -Inf
    idxs = Vector{Int}()
    for (i, x) in enumerate(A)
        if !isnan(x)
            if x > maxval
                maxval = x
                empty!(idxs)
                push!(idxs, i)
            elseif x == maxval
                push!(idxs, i)
            end
        end
    end
    if length(idxs) == 0
        NaN, collect(1:length(A))
    else
        maxval, idxs
    end
end

"""
    deletefirst!(A::Vector, element)

Find the first `element` in `A` and delete it.
`==` is used to compare equality.
"""
function deletefirst!(A::Vector, element)
    deleteat!(A, findfirst(x->x == element, A))
end

"""
    onehot(n::Int, x::Int, t::Type=Int; isdense::Bool=true)

If `isdense` is `false`, a `SparseArray` is returned.
"""
function onehot(n::Int, x::Int, t::Type = Int; isdense::Bool = true)
    a = isdense ? zeros(t, n) : spzeros(t, n)
    a[x] = one(t)
    a
end

discounted_reward(xs, γ) = foldr((r, g)->r + γ * g, xs)

"""
    reverse_discounted_rewards(rewards, γ)

Given the `rewards` and discount ratio `γ`, the discounted reward at each time step in the reversed order is returned.
The returned object is of type [`Reductions`](@ref).
"""
reverse_discounted_rewards(xs, γ) = Reductions((g, r)->g * γ + r, Iterators.reverse(xs))

"""
    importance_weight(π, b, states, actions)

Calculate the importance weight between the target policy `π` and behavior policy `b`
given `states` and `actions`.
"""
function importance_weight(π, b, states, actions)
    ρ = 1.
    for (s, a) in zip(states, actions)
        if π(s, a) == 0
            ρ = 0.
            break
        else
            ρ *= π(s, a) / b(s, a)
        end
    end
    ρ
end

"""
    reverse_importance_weights(π, b, states, actions)

Calculate the importance weight at each time step in the **reversed** order
between the target policy `π` and behavior policy `b` given `states` and `actions`.

The returned object is of type [`Reductions`](@ref)
"""
reverse_importance_weights(π, b, states, actions) = Reductions(
    (ρ, (s, a)) -> ρ == 0. ? 0. : ρ * π(s, a) / b(s, a),
    Iterators.reverse(zip(states, actions)),
    (init=1.,))

const is_using_gpu = false

"""
    huber_loss(labels, predictions;δ = 1.0)

See [huber_loss](https://en.m.wikipedia.org/wiki/Huber_loss)
and the [implementation](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/losses/losses_impl.py#L394-L469) in TensorFlow.
"""
function huber_loss(labels, predictions;δ = 1.0)
    abs_error = abs.(predictions .- labels)
    quadratic = min.(abs_error, δ)
    linear = abs_error .- quadratic
    mean(0.5 .* quadratic .* quadratic .+ δ .* linear)
end