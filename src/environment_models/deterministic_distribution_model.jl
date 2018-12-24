"""
   struct DeterministicDistributionModel <: AbstractDistributionModel
      table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob), Tuple{Int, Float64, Float64}}}, 2}
   end

Store all the transformations in the `table`.
"""
 struct DeterministicDistributionModel <: AbstractDistributionModel
    table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob), Tuple{Int, Float64, Float64}}}, 2}
 end

function update!(::DeterministicDistributionModel, ::AbstractTurnBuffer) end

getstates(m::DeterministicDistributionModel) = axes(m.table, 1)
getactions(m::DeterministicDistributionModel) = axes(m.table, 2)

(m::DeterministicDistributionModel)(s::Int, a::Int) =  m.table[s, a]