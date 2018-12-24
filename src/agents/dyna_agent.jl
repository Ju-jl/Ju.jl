"""
    DynaAgent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tm<:AbstractEnvironmentModel, Tpp<:Function} <: AbstractAgent
    DynaAgent(learner::Tl, buffer::Tb, model::Tm, nsteps::Int=0, preprocessor::Tpp=identity, role=:anonymous) where {Tl, Tb, Tm, Tpp} 

See more details at Section (8.2) on Page 162 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*

See also: [`Agent`](@ref), [`AbstractLearner`](@ref), [`AbstractTurnBuffer`](@ref), [`AbstractEnvironmentModel`](@ref)
"""
struct DynaAgent{Tl<:AbstractLearner,
                 Tb<:AbstractTurnBuffer,
                 Tm<:AbstractEnvironmentModel,
                 Tpp<:Function} <: AbstractAgent
    learner::Tl
    buffer::Tb
    model::Tm
    nsteps::Int
    preprocessor::Tpp
    role::Symbol
    function DynaAgent(learner::Tl,
                       buffer::Tb,
                       model::Tm,
                       nsteps::Int=0,
                       preprocessor::Tpp=identity,
                       role=:anonymous) where {Tl, Tb, Tm, Tpp} 
        new{Tl, Tb, Tm, Tpp}(learner, buffer, model, nsteps, preprocessor, role)
    end
end

function update!(agent::DynaAgent, s, a, r, d, ns, na)
    push!(agent.buffer, s, a, r, d, ns, na)
    update!(agent.learner, agent.buffer)  # direct RL
    update!(agent.model, agent.buffer, agent.learner)  # model learning
    plan!(agent.learner, agent.model, agent.nsteps)  # planning
end

"""
    (agent::DynaAgent)(obs)

Take in an `obs` from environment and use `agent.preprocessor` to transform it into an internal `state`.
Then use `agent.learner` to get the `action`.
Return a pair of `state => action`.
"""
function (agent::DynaAgent)(obs)
    s = agent.preprocessor(obs)
    a = agent.learner(s)
    s => a
end

"""
    plan!(learner::QLearner, model::Union{TimeBasedSampleModel, ExperienceSampleModel}, nsteps::Int)

See more details at Section (8.2) on Page 164 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function plan!(learner::QLearner, model::Union{TimeBasedSampleModel, ExperienceSampleModel}, nsteps::Int)
    for _ in 1:nsteps
        update!(learner, sample(model)...)
    end
end

"""
    plan!(learner::QLearner, model::PrioritizedSweepingSampleModel, nsteps::Int)

See more details at Section (8.2) on Page 170 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function plan!(learner::QLearner, model::PrioritizedSweepingSampleModel, nsteps::Int)
    for _ in 1:nsteps
        record = sample(model)
        record == nothing && continue
        s, a, r, d, s′ = record
        update!(learner, s, a, r, d, s′)
        for (s̄, ā, r̄, d̄) in model.predecessors[s]
            P = priority(learner, s̄, ā, r̄, d̄, s)
            if P ≥ model.θ
                model.PQueue[(s̄, ā)] = P
            end
        end
    end
end