using DataStructures

"""
    PrioritizedSweepingSampleModel <: AbstractSampleModel
    PrioritizedSweepingSampleModel(θ::Float64=1e-4)

See more details at Section (8.4) on Page 168 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct PrioritizedSweepingSampleModel <: AbstractSampleModel
    experiences::Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}
    PQueue::PriorityQueue{Tuple{Any,Any},Float64}
    predecessors::Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}
    θ::Float64
    sample_count::Int
    PrioritizedSweepingSampleModel(θ::Float64=1e-4) = new(Dict{Tuple{Any,Any},Tuple{Float64,Bool,Any}}(),
                                                     PriorityQueue{Tuple{Any,Any},Float64}(Base.Order.Reverse),
                                                     Dict{Any,Set{Tuple{Any,Any,Float64,Bool}}}(),
                                                     θ, 0)
end

function update!(m::PrioritizedSweepingSampleModel, buffer::EpisodeSARDBuffer, learner::AbstractLearner)
    s, a, r, d, s′ = buffer.state[end - 1], buffer.action[end - 1], buffer.reward[end], buffer.isdone[end], buffer.state[end]
    m.experiences[(s, a)] = (r, d, s′)
    P = priority(learner, s, a, r, d, s′)
    if P >= m.θ
        m.PQueue[(s, a)] = P
    end
    if !haskey(m.predecessors, s′)
        m.predecessors[s′] = Set{Tuple{Any,Any,Float64,Bool}}()
    end
    push!(m.predecessors[s′], (s, a, r, d))
end

function sample(m::PrioritizedSweepingSampleModel)
    if length(m.PQueue) > 0
        m.sample_count += 1
        s, a = dequeue!(m.PQueue)
        r, d, s′ = m.experiences[(s, a)]
        s, a, r, d, s′
    else
        nothing
    end
end