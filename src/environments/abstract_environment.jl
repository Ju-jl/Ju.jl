export AbstractEnvironment, AbstractSyncEnvironment, AbstractAsyncEnvironment,
       reset!, observe, render, actionspace, observationspace, get_next_role, isend, get_idle_action

import DataStructures:reset!

"""
    AbstractEnvironment{Tos, Tas, N}

Supertype for different kinds of environments with the observation space of type `Tos`
and action space of type `Tas`. Here the `N` means the number of **agents** the environment
can interact with.

See also: [`AbstractAsyncEnvironment`](@ref), [`AbstractSyncEnvironment`](@ref)
"""
abstract type AbstractEnvironment{Tos, Tas, N} end

"""
    AbstractSyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N}

Supertype for **synchronized** environments with the observation space of type `Tos`
and action space of type `Tas`. Here the `N` means the number of **agents** the environment can interact with.
The `Sync` means that the environment will hang up and wait for the agent's input.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `env(action)` | Each environment `env` must be a functional object that receive an action as input and return a `NamedTuple{(:observation, :reward, :isdone)}`|
| `observe(env)` | Return a `NamedTuple{(:observation, :isdone)}`|
| `reset!(env)` | Reset the environment and return a `NamedTuple{(:observation, :isdone)}`|
| `get_next_role(env)` | Required for multi-agent environments (`N > 1`). Tell the system which agent to act next |
| **Optional Methods** | |
| `observationspace(env)` | Return the observation space of the environment. See also: [`AbstractSpace`](@ref) |
| `actionspace(env)` | Return the action space of the environment.  See also: [`AbstractSpace`](@ref) |
| `render(env)` | Render the environment |
| `isend(env)` | Check whether the `env` reached an end or not. For single agent environment, `observe(env).isdone` is returned. For multi-agents environment, `get_next_role(env) == nothing` is returned |

"""
abstract type AbstractSyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N} end

"""
    AbstractAsyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N}

Opposed to [`AbstractSyncEnvironment`](@ref), the `Async` environments will not wait for
the agent's input. Instead it will keep running asynchronously.
"""
abstract type AbstractAsyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N} end

"Get current state of an environment."
function observe end

"Reset an environment to the initial state"
function reset! end

"Get the state space of an environment"
function observationspace end

"Get the action space of an environment"
function actionspace end

"Render an environment"
function render end

"""
Get the next role to act. Usually used in [`AbstractSyncEnvironment`](@ref).
Either return a role of `Symbol` or `nothing`(which means the end of the game)
"""
get_next_role(::AbstractSyncEnvironment{Tos, Tas, 1}) where {Tos, Tas} = :anonymous

"""
Get the default action for [`AbstractSyncEnvironment`](@ref)
(This interface is experimental)
TODO: Add explaination of why we need it here
"""
function get_idle_action(::AbstractEnvironment) error("unimplemented") end

"""
    isend(env::AbstractEnvironment)

Check whether the `env` has reached an end of episode.
"""
isend(env::AbstractSyncEnvironment{Tos, Tas, 1}) where {Tos, Tas} = observe(env).isdone
isend(env::AbstractSyncEnvironment{Tos, Tas, N}) where {Tos, Tas, N} = get_next_role(env) == nothing