import StatsBase:sample

"""
      ExperienceSampleModel <: AbstractSampleModel

Generate a turn sample based on previous experiences.
"""
mutable struct ExperienceSampleModel <: AbstractSampleModel
   experiences::Dict{Any, Dict{Any, NamedTuple{(:reward, :isdone, :nextstate), Tuple{Float64, Bool, Any}}}}
   sample_count::Int
   ExperienceSampleModel() = new(Dict{Any, Dict{Any, NamedTuple{(:reward, :isdone, :nextstate), Tuple{Float64, Bool, Any}}}}(), 0)
end

update!(m::ExperienceSampleModel, buffer, learner) = update!(m, buffer)

function update!(m::ExperienceSampleModel, buffer::EpisodeSARDBuffer)
   s, a, r, d, s′ = buffer.state[end-1], buffer.action[end-1], buffer.reward[end], buffer.isdone[end], buffer.state[end]
   if haskey(m.experiences, s)
         m.experiences[s][a] = (reward=r, isdone=d, nextstate=s′)
   else
         m.experiences[s] = Dict{Any, NamedTuple{(:reward, :isdone, :nextstate), Tuple{Float64, Bool, Any}}}(a => (reward=r, isdone=d, nextstate=s′))
   end
end

function sample(model::ExperienceSampleModel)
    s = rand(keys(model.experiences))
    a = rand(keys(model.experiences[s]))
    model.sample_count += 1
    s, a, model.experiences[s][a]...
end