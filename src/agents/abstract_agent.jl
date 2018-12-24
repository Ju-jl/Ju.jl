export AbstractAgent,
       update!

import DataStructures:update!

"""
    AbstractAgent

Supertype of **agents**.
Usually, an agent needs to contain at least an [`AbstractLearner`](@ref) and an [`AbstractTurnBuffer`](@ref).

| Required Methods| Brief Description |
|:----------------|:------------------|
| `agent(obs)` | `agent`, an instance of an `AbstractAgent`, must be a functional object to receive an observation as input and return a pair of state and action (s => a) |
| `update!(agent, s, a, r, d, s′[, a′])` | Update the agent after an interaction with environment |
"""
abstract type AbstractAgent end

function update! end
function buffertype(agent::AbstractAgent) 
    typeof(agent.buffer)
end