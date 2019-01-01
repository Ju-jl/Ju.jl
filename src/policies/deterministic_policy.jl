using SparseArrays

"""
    struct DeterministicPolicy <: AbstractPolicy
        table::Vector{Int}
        nactions::Int
    end

The action to be adopted is stored in `table`.
"""
struct DeterministicPolicy <: AbstractPolicy
    table::Vector{Int}
    nactions::Int
end

(p::DeterministicPolicy)(s::Int) = p.table[s]
(p::DeterministicPolicy)(s::Int, ::Val{:dist}) = SparseVector(p.nactions, [p(s)], [1.0])
(p::DeterministicPolicy)(s::Int, a::Int) = p.table[s] == a ? 1.0 : 0.0
update!(p::DeterministicPolicy, s::Int, a::Int) = p.table[s] = a