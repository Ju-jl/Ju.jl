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
        obs, r, d, extra_info = env(a)
        ns, na = d ? agent(reset!(env).observation) : agent(obs)
        update!(agent, s, a, r, d, ns, na)
        s, a = ns, na
        for cb in callbacks
            res = cb(agent, extra_info)
            if res isa Bool && res
                isstop = true
            end
        end
    end
    callbacks
end

function train!(env::AbstractSyncEnvironment{Tss, Tas, N} where {Tss, Tas},
                agents::Tuple{Vararg{<:Agent{<:AbstractLearner, <:SARDSBuffer}, N}};
                callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),)) where N
    agents = Dict((agent.role, agent) for agent in agents)
    isstop = false
    while !isstop
        next_role = get_next_role(env)
        next_role == nothing && break
        agent = agents[next_role]

        s, a = observe(env, agent.role).observation |> agent
        obs, r, d, extra_info = env(a, agent.role)
        ns = d ? agent.preprocessor(reset!(env).observation) : agent.preprocessor(obs)
        update!(agent, s, a, r, d, ns)
        for cb in callbacks
            res = cb(agent, extra_info)
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