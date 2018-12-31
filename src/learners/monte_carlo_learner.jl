using StatsBase:mean
const PolicyOrSelector = Union{AbstractPolicy, AbstractActionSelector}

"""
    MonteCarloLearner(approximator::Tapp, π::Tp, γ::Float64=1., α::Float64 = 1.0, first_visit::Bool = true) where {Tapp,Tp} 

See more details at Section (5.1) on Page 92 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct MonteCarloLearner{visit,Tapp <: AbstractApproximator,Tp <:PolicyOrSelector,Tr <: Function} <: AbstractMonteCarloLearner
    approximator::Tapp
    π::Tp
    γ::Float64
    α::Float64
    returns::Tr
    function MonteCarloLearner(approximator::Tapp, π::Tp, γ::Float64=1., α::Float64 = 1.0, first_visit::Bool = true) where {Tapp,Tp} 
        returns = cached_sample_avg()
        if first_visit
            new{:FirstVisit,Tapp,Tp,typeof(returns)}(approximator, π, γ, α, returns)
        else
            new{:EveryVisit,Tapp,Tp,typeof(returns)}(approximator, π, γ, α, returns)
        end
    end
end

(learner::MonteCarloLearner{Tvisit, <:AbstractQApproximator, <:AbstractActionSelector})(s) where Tvisit = learner.approximator(s) |> learner.π
(learner::MonteCarloLearner{Tvisit, <:AbstractApproximator, <:AbstractPolicy})(s) where Tvisit = learner.π(s)
(learner::MonteCarloLearner{Tvisit, <:AbstractApproximator, <:AbstractPolicy})(s, ::Val{:dist}) where Tvisit = learner.π(s, Val(:dist))

update!(learner::MonteCarloLearner, buffer::EpisodeSARDBuffer) = isfull(buffer) && update!(learner, @view(buffer.state[1:end - 1]), @view(buffer.action[1:end - 1]), buffer.reward)
update!(learner::MonteCarloLearner, buffer::EpisodeSARDSBuffer) = isfull(buffer) && update!(learner, buffer.state, buffer.action, buffer.reward)
update!(learner::MonteCarloLearner{Tvisit,<:AbstractVApproximator}, states, actions, rewards) where Tvisit = update!(learner, states, rewards)

function update!(learner::MonteCarloLearner{:FirstVisit,<:AbstractVApproximator}, states, rewards)
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.
    for (isfirstvisit, s, r) in Iterators.reverse(zip(IsFirstVisit(states), states, rewards))
        if isfirstvisit
            G = γ * G + r
            update!(V, s, α * (Returns(s, G) - V(s)))
        end
    end
end

function update!(learner::MonteCarloLearner{:EveryVisit,<:AbstractVApproximator}, states, rewards)
    α, γ, V, Returns, G = learner.α, learner.γ, learner.approximator, learner.returns, 0.
    for (s, r) in Iterators.reverse(zip(states, rewards))
        G = γ * G + r
        update!(V, s, α * (Returns(s, G) - V(s)))
    end
end

function update!(learner::MonteCarloLearner{:FirstVisit,<:AbstractQApproximator}, states, actions, rewards)
    α, γ, Q, π, Returns, G = learner.α, learner.γ, learner.approximator, learner.π, learner.returns, 0.
    for (isfirstvisit, s, a, r) in Iterators.reverse(zip(IsFirstVisit(states), states, actions, rewards))
        G = γ * G + r
        if isfirstvisit
            update!(Q, s, a, α * (Returns((s, a), G) - Q(s, a)))
            update!(π, s, Q(s, Val(:argmax)))
        end
    end
end

function update!(learner::MonteCarloLearner{:EveryVisit, <:AbstractQApproximator}, states, actions, rewards)
    α, γ, Q, π, Returns, G = learner.α, learner.γ, learner.approximator, learner.π, learner.returns, 0.
    for (s, a, r) in Iterators.reverse(zip(states, actions, rewards))
        G = γ * G + r
        update!(Q, s, a, α * (Returns((s, a), G) - Q(s, a)))
        update!(π, s, Q(s, Val(:argmax)))
    end
end

"""
    MonteCarloExploringStartLearner(approximator::Tapp, π::Tp, π_start::RandomPolicy, γ::Float64, α::Float64 = 1.0; is_first_visit::Bool = true) where {Tapp,Tp} 

See more details at Section (5.3) on Page 99 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct MonteCarloExploringStartLearner{visit,Tapp <: AbstractApproximator,Tp <: AbstractPolicy,Tr <: Function} <: AbstractMonteCarloLearner
    approximator::Tapp
    π::Tp
    π_start::RandomPolicy
    γ::Float64
    α::Float64
    returns::Tr
    is_start::Bool
    function MonteCarloExploringStartLearner(approximator::Tapp, π::Tp, π_start::RandomPolicy, γ::Float64, α::Float64 = 1.0; is_first_visit::Bool = true) where {Tapp,Tp} 
        returns = cached_sample_avg()
        if is_first_visit
            new{:FirstVisit,Tapp,Tp,typeof(returns)}(approximator, π, π_start, γ, α, returns, true)
        else
            new{:EveryVisit,Tapp,Tp,typeof(returns)}(approximator, π, π_start, γ, α, returns, true)
        end
    end
end

(learner::MonteCarloExploringStartLearner)(s) = learner.is_start ? learner.π_start(s) : learner.π(s)

function update!(learner::MonteCarloExploringStartLearner{Tvisit,<:AbstractQApproximator}, buffer::EpisodeSARDBuffer) where Tvisit
    α, γ, Q, π, Returns = learner.α, learner.γ, learner.approximator, learner.π, learner.returns
    if isfull(buffer)
        learner.is_start = true  # reset start flag
        G, states_actions, rewards = 0, zip(@view(buffer.state[1:end - 1]), @view(buffer.action[1:end - 1])), buffer.reward
        for (isfirstvisit, (s, a), r) in Iterators.reverse(zip(IsFirstVisit(states_actions), states_actions, rewards))
            G = γ * G + r
            if Tvisit == :EveryVisit || (Tvisit == :FirstVisit && isfirstvisit)
                update!(Q, s, a, α * (Returns((s, a), G) - Q(s, a)))
                update!(π, s, Q(s, Val(:argmax)))
            end
        end
    else
        learner.is_start = false  # reset start flag
    end
end

"""
    OffPolicyMonteCarloLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Float64 = 1.0; isfirstvisit::Bool = true, sampling::Symbol = :OrdinaryImportanceSampling,) where {Tapp,Tpb,Tpt} 

See more details at Section (5.7) on Page 111 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct OffPolicyMonteCarloLearner{Tvisit,Sampling,Tapp <: AbstractApproximator,Tpb <: AbstractPolicy,Tpt <: AbstractPolicy} <: AbstractMonteCarloLearner
    approximator::Tapp
    π_behavior::Tpb
    π_target::Tpt
    γ::Float64
    α::Float64
    ρ_G::Dict{Int,Vector{Pair{Float64,Float64}}}
    function OffPolicyMonteCarloLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Float64 = 1.0; isfirstvisit::Bool = true, sampling::Symbol = :OrdinaryImportanceSampling,) where {Tapp,Tpb,Tpt} 
        sampling in [:OrdinaryImportanceSampling, :WeightedImportanceSampling] || error("unknown sampling method $method")
        if isfirstvisit
            new{:FirstVisit,sampling,Tapp,Tpb,Tpt}(approximator, π_behavior, π_target, γ, α, Dict{Int,Vector{Pair{Float64,Float64}}}())
        else
            new{:EveryVisit,sampling,Tapp,Tpb,Tpt}(approximator, π_behavior, π_target, γ, α, Dict{Int,Vector{Pair{Float64,Float64}}}())
        end
    end
end

(learner::OffPolicyMonteCarloLearner)(s) = learner.π_behavior(s)

function update!(learner::OffPolicyMonteCarloLearner{Tvisit,Tsampling,<:AbstractVApproximator}, buffer::EpisodeSARDBuffer) where {Tvisit,Tsampling}
    α, γ, V = learner.α, learner.γ, learner.approximator
    if isfull(buffer)
        G, ρ, states, actions, rewards = 0, 1., @view(buffer.state[1:end - 1]), @view(buffer.action[1:end - 1]), buffer.reward
        for (isfirstvisit, s, a, r) in Iterators.reverse(zip(IsFirstVisit(states), states, actions, rewards))
            G = γ * G + r
            ρ *= learner.π_target(s, a) / learner.π_behavior(s, a)
            if Tvisit == :EveryVisit || (Tvisit == :FirstVisit && isfirstvisit)
                if haskey(learner.ρ_G, s)
                    push!(learner.ρ_G[s], ρ => G)
                else
                    learner.ρ_G[s] = [ρ => G]
                end
            end
        end
        if Tsampling == :OrdinaryImportanceSampling
            for s in keys(learner.ρ_G)
                v = mean(ρ * G for (ρ, G) in learner.ρ_G[s])
                update!(V, s, α * (v - V(s)))
            end
        else
            for s in keys(learner.ρ_G)
                numerator = sum(ρ * G for (ρ, G) in learner.ρ_G[s])
                denominator = sum(ρ for (ρ, _) in learner.ρ_G[s])
                v = denominator == 0 ? 0 : numerator / denominator
                update!(V, s, α * (v - V(s)))
            end
        end
    end
end