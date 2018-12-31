export train!, policy_evaluation!, policy_improvement!, policy_iteration!, value_iteration!

"""
    train!(env, agent; callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),))

Train an `agent` in `env`.
By default `callbacks` is `(stop_at_step(1),)`. So it will only train one step forward.
"""
train!(env, agent; callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),)) = train!(env, agent, buffertype(agent); callbacks=callbacks)

function train!(env::AbstractSyncEnvironment{Tss, Tas, 1} where {Tss, Tas},
                agent::AbstractAgent,
                ::Type{<:SARDBuffer};
                callbacks)
    s, a = observe(env).observation |> agent
    isstop = false
    while !isstop
        obs, r, d = env(a)
        if d
            reset!(env)
            ns, na = agent(observe(env).observation)
        else
            ns, na = agent(obs)
        end
        push!(buffer(agent), s, a, r, d, ns, na)
        update!(agent)
        s, a = ns, na
        for cb in callbacks
            res = cb(env, agent)
            if res isa Bool && res
                isstop = true
            end
        end
    end
    callbacks
end

"""
    train!(env::AbstractSyncEnvironment{Tss, Tas, N} where {Tss, Tas},
           agents::Tuple{Vararg{<:Agent{<:AbstractLearner, <:SARDSBuffer}, N}};
           callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),)) where N

TODO: Add an `AgentManager` struct to better organize `agents`.

For sync environments of mulit-agents, it becomes much more complicated compared to the single agent environments.
Here is an implementation for one of the most common cases. Each agent take an action alternately.
In every step, all agents may observe partial/complete information of the environment from their own perspective.

You may consider to overwrite this function according to the problem you want to solve.
"""
function train!(env::AbstractSyncEnvironment{Tss, Tas, N} where {Tss, Tas},
                agents::Tuple{Vararg{<:Agent{<:AbstractLearner, <:SARDBuffer}, N}};
                callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),)) where N
    named_agents = Dict((agent.role, agent) for agent in agents)

    a = nothing
    next_role = get_next_role(env)

    for agent in agents
        obs = observe(env, agent.role).observation
        if agent.role == next_role
            s, a = agent(obs)
            push!(buffer(agent), s, a)
        else
            push!(buffer(agent), agent.preprocessor(obs), get_idle_action(env))
        end
    end

    isstop = false
    while !isstop
        next_role = get_next_role(env)
        if next_role == nothing
            reset!(env)
            next_role = get_next_role(env)
            for agent in agents
                empty!(buffer(agent))
                obs = observe(env, agent.role).observation
                if agent.role == next_role
                    s, a = agent(obs)
                    push!(buffer(agent), s, a)
                else
                    push!(buffer(agent), agent.preprocessor(obs), get_idle_action(env))
                end
            end
        end
        
        env(a, next_role)  # now take action
        next_role = get_next_role(env)

        for agent in agents
            obs, isdone, reward = observe(env, agent.role)
            if agent.role == next_role
                s, a = agent(obs)
                push!(buffer(agent), reward, isdone, s, a)
            else
                push!(buffer(agent), reward, isdone, agent.preprocessor(obs), get_idle_action(env))
            end
            update!(agent)
        end

        for cb in callbacks
            res = cb(env, named_agents)
            if res isa Bool && res
                isstop = true
            end
        end
    end
    callbacks
end

"""
    policy_evaluation!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4)

See more details at Section (4.1) on Page 75 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_evaluation!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4)
    states, actions = getstates(Model), getactions(Model)
    while true
        Δ = 0.
        for s in states
            v = sum(π(s, Val(:dist)) .* (sum(p * (r + γ * V(s′)) for (s′, r, p) in Model(s, a)) for a in actions))
            error = v - V(s)
            update!(V, s, error)
            Δ = max(Δ, abs(error))
        end
        Δ < θ && break
    end
end

"""
    policy_improvement!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9)

See more details at Section (4.2) on Page 76 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_improvement!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9)
    states, actions = getstates(Model), getactions(Model)
    is_policy_stable = true
    for s in states
        old_a = π(s)
        best_action_inds = findallmax(sum(p * (r + γ * V(s′)) for (s′, r, p) in Model(s, a)) for a in actions)[2]
        new_a = actions[sample(best_action_inds)]
        if new_a != old_a 
            update!(π, s, new_a)
            is_policy_stable = false
        end
    end
    is_policy_stable
end

"""
    policy_iteration!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))

See more details at Section (4.3) on Page 80 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function policy_iteration!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))
    for i in 1:max_iter
        @debug "iteration: $i"
        policy_evaluation!(V, π, Model; γ=γ, θ=θ)
        policy_improvement!(V, π, Model; γ=γ) && break
    end
end

"""
    value_iteration!(V::AbstractVApproximator, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))

See more details at Section (4.4) on Page 83 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
function value_iteration!(V::AbstractVApproximator, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))
    states, actions = getstates(Model), getactions(Model)
    for i in 1:max_iter
        Δ = 0.
        for s in states
            v = maximum(sum(p * (r + γ * V(s′)) for (s′, r, p) in Model(s, a)) for a in actions)
            error = v - V(s)
            update!(V, s, error)
            Δ = max(Δ, abs(error))
        end
        Δ < θ && break
    end
end