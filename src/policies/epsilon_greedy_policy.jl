"""
    struct EpsilonGreedyPolicy <: AbstractPolicy
        table::Vector{Int}
        nactions::Int
        ϵ::Float64
    end

Just like the [`DeterministicPolicy`](@ref), the best actions are stored in the `table`.
However the best action will only be taken at a portion of 1 - ϵ.

See also: [`EpsilonGreedySelector`](@ref)
"""
struct EpsilonGreedyPolicy <: AbstractPolicy
    table::Vector{Int}
    nactions::Int
    ϵ::Float64
end

function (p::EpsilonGreedyPolicy)(s::Int) 
    if rand() > p.ϵ
        p.table[s]
    else
        rand(1:p.nactions)
    end
end

function (p::EpsilonGreedyPolicy)(s::Int, ::Val{:dist})
    prob = fill(p.ϵ/p.nactions, p.nactions)
    prob[p.table[s]] += 1 - p.ϵ
    prob
end

(p::EpsilonGreedyPolicy)(s::Int, a::Int) = p.table[s] == a ? (1 - p.ϵ) + p.ϵ / p.nactions : p.ϵ / p.nactions

update!(p::EpsilonGreedyPolicy, s::Int, a::Int) = p.table[s] = a