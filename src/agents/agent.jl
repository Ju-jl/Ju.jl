"""
    Agent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tpp<:Function} <: AbstractAgent 
    Agent(learner::Tl, buffer::Tb, preprocessor::Tpp=identity, role=:anonymous) where {Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tpp<:Function}

A `preprocessor` is just a normal function.
It transforms the **observation** from an environment to the internal **state**, which is then stored in the `buffer`.
`role` is a `Symbol`. Usually it is used in multi-agents environment to distinction different agents.

See also: [`AbstractLearner`](@ref), [`AbstractTurnBuffer`](@ref)
"""
struct Agent{Tl<:AbstractLearner,
             Tb<:AbstractTurnBuffer,
             Tpp<:Function} <: AbstractAgent
    learner::Tl
    buffer::Tb
    preprocessor::Tpp
    role::Symbol
    function Agent(learner::Tl,
                   buffer::Tb,
                   preprocessor::Tpp=identity,
                   role=:anonymous) where {Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tpp<:Function} 
        new{Tl, Tb, Tpp}(learner, buffer, preprocessor, role)
    end
end
 
"""
    update!(agent::Agent, s, a, r, d, ns, na)

Given a turn info `s`tate, `a`ction, `r`eward, is`d`one, `n`ext`s`tate, `n`ext`a`ction,
(usually produced in **Synchronize Environments**)
then update the `agent.buffer` and `agent.learner`.
"""
function update!(agent::Agent, s, a, r, d, ns, na)
    push!(agent.buffer, s, a, r, d, ns, na)
    update!(agent.learner, agent.buffer)
end

"""
    update!(agent::Agent, s, a, r, d, ns)

Given a turn info `s`tate, `a`ction, `r`eward, is`d`one, `n`ext`s`tate,
(usually produced in **Asynchronize Environments**)
then update the `agent.buffer` and `agent.learner`.
"""
function update!(agent::Agent)
    update!(agent.learner, agent.buffer)
end

"""
    (agent::Agent)(obs)

Take in an `obs` from environment and use `agent.preprocessor` to transform it into an internal `state`.
Then use `agent.learner` to get the `action`.
Return a pair of `state => action`.
"""
function (agent::Agent)(obs)
    s = agent.preprocessor(obs)
    a = agent.learner(s)
    s => a
end