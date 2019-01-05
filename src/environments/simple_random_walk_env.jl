"""
    SimpleRandomWalkEnv

A simple random walk environment for tutorial.

# Example
```julia
julia> env = SimpleRandomWalkEnv()
SimpleRandomWalkEnv(7, 3, 3, [-1, 1])

julia> env(1)
(observation = 2, reward = 0.0, isdone = false)

julia> reset!(env)
(observation = 3, isdone = false)
```
"""
mutable struct SimpleRandomWalkEnv <: AbstractSyncEnvironment{DiscreteSpace,DiscreteSpace,1}
    N::Int
    start::Int
    state::Int
    actions::Vector{Int}
    SimpleRandomWalkEnv(;N=7, actions=[-1, 1], start=3) = new(N, start, start, actions)
end

function (env::SimpleRandomWalkEnv)(a::Int)
    env.state = min(max(env.state + env.actions[a], 1), env.N)
    (observation = env.state,
     reward      = env.state == env.N ? 3.0 : (env.state == 1 ? 1. : 0.),
     isdone      = env.state == env.N || env.state == 1)
end

function reset!(env::SimpleRandomWalkEnv)
    env.state = env.start
    (observation = env.state, isdone = false)
end

observe(env::SimpleRandomWalkEnv) = (observation = env.state, isdone = env.state == env.N || env.state == 1)
observationspace(env::SimpleRandomWalkEnv) = DiscreteSpace(env.N)
actionspace(env::SimpleRandomWalkEnv) = DiscreteSpace(length(env.actions))