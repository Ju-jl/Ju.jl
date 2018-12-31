const PolicyOrSelector = Union{AbstractPolicy, AbstractActionSelector}

"""
    TDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0) where {Tapp<:AbstractVApproximator, Tp<:PolicyOrSelector} = new{Tapp, Tp, :SRS}(approximator, π, γ, α, n)
    TDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator, Tp<:PolicyOrSelector} 

See more details at Section (7.1) on Page 142 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct TDLearner{Tapp <: AbstractApproximator, Tp <: PolicyOrSelector, method} <: AbstractModelFreeLearner 
    approximator::Tapp
    π::Tp
    γ::Float64
    α::Float64
    n::Int
    TDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0) where {Tapp<:AbstractVApproximator, Tp<:PolicyOrSelector} = new{Tapp, Tp, :SRS}(approximator, π, γ, α, n)
    function TDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator, Tp<:PolicyOrSelector} 
        supported_methods = [:SARSA, :ExpectedSARSA]
        !in(method, supported_methods) && throw(ArgumentError("Supported methods are $supported_methods , your input is $method"))
        new{Tapp, Tp, method}(approximator, π, γ, α, n)
    end
end



(learner::TDLearner{<:AbstractQApproximator, <:AbstractActionSelector})(s) = learner.approximator(s) |> learner.π
(learner::TDLearner{<:AbstractApproximator, <:AbstractPolicy})(s) = learner.π(s)
(learner::TDLearner{<:AbstractApproximator, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :SARSA}, buffer::EpisodeSARDBuffer)
    n = learner.n
    update!(learner,
            @view(buffer.state[max(1, end - n - 1) : end - 1]),
            @view(buffer.action[max(1, end - n - 1) : end - 1]),
            @view(buffer.reward[max(1, end - n) : end]),
            @view(buffer.state[max(1, end - n) : end]),
            @view(buffer.action[max(1, end - n) : end]),
            Val(buffer.isdone[end]))
end

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :ExpectedSARSA}, buffer::EpisodeSARDBuffer)
    n = learner.n
    update!(learner,
            @view(buffer.state[max(1, end - n - 1) : end - 1]),
            @view(buffer.action[max(1, end - n - 1) : end - 1]),
            @view(buffer.reward[max(1, end - n) : end]),
            @view(buffer.state[max(1, end - n) : end]),
            Val(buffer.isdone[end]))
end

function update!(learner::TDLearner{<:AbstractVApproximator, <:PolicyOrSelector, :SRS}, buffer::EpisodeSARDBuffer)
    n = learner.n
    update!(learner,
            @view(buffer.state[max(1, end - n - 1) : end - 1]),
            @view(buffer.reward[max(1, end - n) : end]),
            @view(buffer.state[max(1, end - n) : end]),
            Val(buffer.isdone[end]))
end

function update!(learner::TDLearner{<:AbstractVApproximator,<:PolicyOrSelector, :SRS}, states, rewards, nextstates, ::Val{true})
    n, γ, V, α = learner.n, learner.γ, learner.approximator, learner.α
    # Warning!!! The order of calculation is reversed here for speed. The impact is uncertain!!!
    for (G, s) in zip(reverse_discounted_rewards(rewards, γ), Iterators.reverse(states))
        update!(V, s, α * (G - V(s)))
    end
end

function update!(learner::TDLearner{<:AbstractVApproximator,<:PolicyOrSelector, :SRS}, states, rewards, nextstates, ::Val{false})
    n, γ, V, α = learner.n, learner.γ, learner.approximator, learner.α
    if length(states) ≥ n
        G = discounted_reward(rewards, γ) + γ^n * V(nextstates[end])
        s = states[1]
        update!(V, s, α * (G - V(s)))
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :SARSA}, states, actions, rewards, nextstates, nextactions, ::Val{true})
    n, γ, Q, α = learner.n, learner.γ, learner.approximator, learner.α
    # Warning!!! The order of calculation is reversed here for speed. The impact is uncertain!!!
    for (G, s, a) in zip(reverse_discounted_rewards(rewards, γ), Iterators.reverse(states), Iterators.reverse(actions))
        update!(Q, s, a, α * (G - Q(s, a)))
        update!(learner.π, s, Q(s, Val(:argmax)))
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :SARSA}, states, actions, rewards, nextstates, nextactions, ::Val{false})
    n, γ, Q, α = learner.n, learner.γ, learner.approximator, learner.α
    if length(states) ≥ n
        G = discounted_reward(rewards, γ) + γ^n * Q(nextstates[end], nextactions[end])
        s, a = states[1], actions[1]
        update!(Q, s, a, α * (G - Q(s, a)))
        update!(learner.π, s, Q(s, Val(:argmax)))
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :ExpectedSARSA}, states, actions, rewards, nextstates, ::Val{true})
    n, γ, Q, α = learner.n, learner.γ, learner.approximator, learner.α
    # Warning!!! The order of calculation is reversed here for speed. The impact is uncertain!!!
    for (G, s, a) in zip(reverse_discounted_rewards(rewards, γ), Iterators.reverse(states), Iterators.reverse(actions))
        update!(Q, s, a, α * (G - Q(s, a)))
        update!(learner.π, s, Q(s, Val(:argmax)))
    end
end

function update!(learner::TDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :ExpectedSARSA}, states, actions, rewards, nextstates, ::Val{false})
    n, γ, Q, α, π = learner.n, learner.γ, learner.approximator, learner.α, learner.π
    if length(states) ≥ n
        G = discounted_reward(rewards, γ) + γ^n * sum(Q(nextstates[end], Val(:dist)) .* π(nextstates[end], Val(:dist)))
        s, a = states[1], actions[1]
        update!(Q, s, a, α * (G - Q(s, a)))
        update!(learner.π, s, Q(s, Val(:argmax)))
    end
end

"""

    struct TDλReturnLearner{Tapp <: AbstractApproximator, Tp <: PolicyOrSelector} <: AbstractModelFreeLearner 
        approximator::Tapp
        π::Tp
        γ::Float64
        α::Float64
        λ::Float64
    end

See more details at Section (12.2) on Page 292 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct TDλReturnLearner{Tapp <: AbstractApproximator, Tp <: PolicyOrSelector} <: AbstractModelFreeLearner 
    approximator::Tapp
    π::Tp
    γ::Float64
    α::Float64
    λ::Float64
end

(learner::TDλReturnLearner)(s) = learner.π(s)
(learner::TDλReturnLearner)(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::TDλReturnLearner{<:AbstractVApproximator}, buffer::EpisodeSARDBuffer)
    λ, γ, V, α, T = learner.λ, learner.γ, learner.approximator, learner.α, length(buffer.state)
    if isfull(buffer)
        for t in 1:(T-1)
            G = 0.
            for n in 1:(T - t - 1)
                G += λ^(n-1) * (discounted_reward(@view(buffer.reward[t:t+n-1]), γ) + γ^n * V(buffer.state[t+n]))
            end
            G *= 1 - λ
            G += λ^(T-t-1) * (discounted_reward(@view(buffer.reward[t:T-1]), γ) + γ^(T-t) * V(buffer.state[T]))
            sₜ = buffer.state[t]
            update!(V, sₜ, α * (G - V(sₜ)))
        end
    end
end


"""
    OffPolicyTDLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Float64, n::Int=0, method::Symbol=:SARSA_ImportanceSampling) where {Tapp<:AbstractApproximator, Tpb<:AbstractPolicy, Tpt<:AbstractPolicy} 

See more details at Section (7.3) on Page 148 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct OffPolicyTDLearner{Tapp <: AbstractApproximator, Tpb <: PolicyOrSelector, Tpt <: PolicyOrSelector, Tα<:Union{Function, Float64}, method} <: AbstractModelFreeLearner 
    approximator::Tapp
    π_behavior::Tpb
    π_target::Tpt
    γ::Float64
    α::Tα
    n::Int
    function OffPolicyTDLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Tα, n::Int=0, method::Symbol=:SARSA_ImportanceSampling) where {Tapp<:AbstractApproximator, Tpb<:PolicyOrSelector, Tpt<:PolicyOrSelector, Tα<:Union{Function, Float64}} 
        new{Tapp, Tpb, Tpt, Tα, method}(approximator, π_behavior, π_target, γ, α, n)
    end
end

const QLearner = OffPolicyTDLearner{Tapp, Tp, Tp, Tα, :QLearning} where {Tapp<:AbstractQApproximator, Tp<:PolicyOrSelector, Tα<:Union{Function, Float64}}
QLearner(Q::AbstractQApproximator, π::PolicyOrSelector, γ::Float64, α::Union{Function, Float64}) = OffPolicyTDLearner(Q, π, π, γ, α, 0, :QLearning)

(learner::OffPolicyTDLearner{<:AbstractApproximator, <:AbstractActionSelector})(s) = learner.approximator(s) |> learner.π_behavior
(learner::OffPolicyTDLearner{<:AbstractApproximator, <:AbstractPolicy})(s) = learner.π_behavior(s)
(learner::OffPolicyTDLearner{<:AbstractApproximator, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π_behavior(s, Val(:dist))

priority(learner::QLearner{<:AbstractQApproximator, <:AbstractPolicy, <:Float64}, s, a, r, d, s′) = priority(learner, learner.α, s, a, r, d, s′)
priority(learner::QLearner{<:AbstractQApproximator, <:AbstractPolicy, <:Function}, s, a, r, d, s′) = priority(learner, learner.α((s,a)), s, a, r, d, s′)
function priority(learner::QLearner, α, s, a, r, d, s′)
    γ, Q = learner.γ, learner.Q
    error = d ? α * (r - Q(s, a)) : α * (r + γ * Q(s′, Val(:max)) - Q(s, a))
    abs(error)
end

function update!(learner::OffPolicyTDLearner{<:AbstractApproximator, <:PolicyOrSelector, <:PolicyOrSelector, <:Union{Function, Float64}, :SARSA_ImportanceSampling}, buffer::EpisodeSARDBuffer)
    n = learner.n
    update!(learner,
            @view(buffer.state[max(1, end - n - 1) : end - 1]),
            @view(buffer.action[max(1, end - n - 1) : end - 1]),
            @view(buffer.reward[max(1, end - n) : end]),
            @view(buffer.state[max(1, end - n) : end]),
            @view(buffer.action[max(1, end - n) : end]),
            Val(buffer.isdone[end]))
end

update!(learner::QLearner{<:AbstractQApproximator, <:PolicyOrSelector, <:Float64}, s, a, r, d, s′) = update!(learner, learner.α, s, a, r, d, s′)
update!(learner::QLearner{<:AbstractQApproximator, <:PolicyOrSelector, <:Function}, s, a, r, d, s′) = update!(learner, learner.α((s, a)), s, a, r, d, s′)
function update!(learner::QLearner, α, s, a, r, d, s′)
    Q, γ, π = learner.approximator, learner.γ, learner.π_target
    error = d ? α * (r - Q(s, a)) : α * (r + γ * Q(s′, Val(:max)) - Q(s, a))
    update!(Q, s, a, error)
    update!(π, s, Q(s, Val(:argmax)))
end

function update!(learner::QLearner, buffer::EpisodeSARDBuffer)
    s, a, r, d, s′ = buffer.state[end-1], buffer.action[end-1], buffer.reward[end], buffer.isdone[end], buffer.state[end]
    update!(learner, s, a, r, d, s′)
end

function update!(learner::QLearner, buffer::EpisodeSARDSBuffer)
    update!(learner, buffer.state[end], buffer.action[end], buffer.reward[end], buffer.isdone[end], buffer.nextstate[end])
end

function update!(learner::OffPolicyTDLearner{<:AbstractQApproximator, <:AbstractPolicy,  <:AbstractPolicy, <:Union{Function, Float64}, :SARSA_ImportanceSampling}, states, actions, rewards, nextstates, nextactions, ::Val{true})
    n, γ, Q, α, π, b = learner.n, learner.γ, learner.approximator, learner.α, learner.π_target, learner.π_behavior
    # Warning!!! The order of calculation is reversed here for speed. The impact is uncertain!!!
    for (G, ρ, s, a) in zip(reverse_discounted_rewards(rewards, γ),
                            reverse_importance_weights(π, b, @view(states[2:end]), @view(actions[2:end])),
                            Iterators.reverse(states),
                            Iterators.reverse(actions))
        update!(Q, s, a, α * ρ * (G - Q(s, a)))
        update!(π, s, Q(s, Val(:argmax)))
    end
end

function update!(learner::OffPolicyTDLearner{<:AbstractVApproximator, <:AbstractPolicy,  <:AbstractPolicy, <:Union{Function, Float64}, :SARSA_ImportanceSampling}, states, actions, rewards, nextstates, nextactions, ::Val{true})
    n, γ, V, α, π, b = learner.n, learner.γ, learner.approximator, learner.α, learner.π_target, learner.π_behavior
    # Warning!!! The order of calculation is reversed here for speed. The impact is uncertain!!!
    for (G, ρ, s, a) in zip(reverse_discounted_rewards(rewards, γ),
                            reverse_importance_weights(π, b, @view(states[2:end]), @view(actions[2:end])),
                            Iterators.reverse(states))
        update!(V, s, α * ρ * (G - V(s)))
    end
end

function update!(learner::OffPolicyTDLearner{<:AbstractQApproximator, <:AbstractPolicy,  <:AbstractPolicy, <:Union{Function, Float64}, :SARSA_ImportanceSampling}, states, actions, rewards, nextstates, nextactions, ::Val{false})
    n, γ, Q, α, π, b = learner.n, learner.γ, learner.approximator, learner.α, learner.π_target, learner.π_behavior
    if length(states) ≥ n
        G = discounted_reward(rewards, γ) + γ^n * Q(nextstates[end], nextactions[end])
        ρ = importance_weight(π, b, @view(states[2:end]), @view(actions[2:end]))
        s, a = states[1], actions[1]
        update!(Q, s, a, α * ρ * (G - Q(s, a)))
        update!(π, s, Q(s, Val(:argmax)))
    end
end

function update!(learner::OffPolicyTDLearner{<:AbstractVApproximator, <:AbstractPolicy,  <:AbstractPolicy, <:Union{Function, Float64}, :SARSA_ImportanceSampling}, states, actions, rewards, nextstates, nextactions, ::Val{false})
    n, γ, V, α, π, b = learner.n, learner.γ, learner.approximator, learner.α, learner.π_target, learner.π_behavior
    if length(states) ≥ n
        G = discounted_reward(rewards, γ) + γ^n * V(nextstates[end])
        ρ = importance_weight(π, b, @view(states[2:end]), @view(actions[2:end]))
        s = states[1]
        update!(V, s, α * ρ * (G - V(s)))
    end
end

"""
    struct DoubleLearner{Tl <: OffPolicyTDLearner, Ts<:AbstractActionSelector} <: AbstractLearner
        Learner1::Tl
        Learner2::Tl
        selector::Ts
    end

See more details at Section (6.7) on Page 126 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct DoubleLearner{Tl <: OffPolicyTDLearner, Ts<:AbstractActionSelector} <: AbstractLearner
    Learner1::Tl
    Learner2::Tl
    selector::Ts
end

(learner::DoubleLearner)(s, ::Val{:dist}) = learner.Learner1(s, Val(:dist)) .+ learner.Learner2(s, Val(:dist))
(learner::DoubleLearner)(s) =  learner.selector(learner(s, Val(:dist)))

function update!(learner::DoubleLearner{<:OffPolicyTDLearner{<:AbstractQApproximator, <:AbstractPolicy, <:AbstractPolicy, <:Union{Function, Float64}, :QLearning}}, buffer::EpisodeSARDBuffer)
    s, a, r, s′ = buffer.state[end-1], buffer.action[end-1], buffer.reward[end], buffer.state[end]

    if rand() < 0.5
        L1, L2 = learner.Learner1, learner.Learner2
    else
        L2, L1 = learner.Learner1, learner.Learner2
    end

    if !buffer.isdone[end]
        update!(L1.Q, s, a, L1.α * (r + L1.γ * L2.Q(s′, L1.Q(s′, Val(:argmax))) - L1.Q(s, a)))
    else
        update!(L1.Q, s, a, L1.α * (r - L1.Q(s, a)))
    end

    update!(L1.π_target, s, L1.Q(s, Val(:argmax)))
end

"""
    DifferentialTDLearner(approximator::Tapp, π::Tp, α::Float64, β::Float64, R̄::Float64=0., n::Int=1, method::Symbol=:SARSA) where {Tapp<:AbstractApproximator, Tp<:PolicyOrSelector}= new{Tapp, Tp, method}(approximator, π, α, β, R̄, n)

See more details at Section (10.3) on Page 251 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct DifferentialTDLearner{Tapp<:AbstractApproximator, Tp<:PolicyOrSelector, method} <: AbstractLearner
    approximator::Tapp
    π::Tp
    α::Float64
    β::Float64
    R̄::Float64
    n::Int
    DifferentialTDLearner(approximator::Tapp, π::Tp, α::Float64, β::Float64, R̄::Float64=0., n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractApproximator, Tp<:PolicyOrSelector}= new{Tapp, Tp, method}(approximator, π, α, β, R̄, n)
end

(learner::DifferentialTDLearner{<:AbstractQApproximator, <:AbstractActionSelector})(s) = learner.approximator(s) |> learner.π
(learner::DifferentialTDLearner{<:AbstractApproximator, <:AbstractPolicy})(s) = learner.π(s)
(learner::DifferentialTDLearner{<:AbstractApproximator, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::DifferentialTDLearner{<:AbstractApproximator, <:PolicyOrSelector, :SARSA}, buffer::Union{EpisodeSARDBuffer, CircularSARDBuffer})
    n = learner.n
    update!(learner,
            @view(buffer.state[max(1, end - n - 1) : end - 1]),
            @view(buffer.action[max(1, end - n - 1) : end - 1]),
            @view(buffer.reward[max(1, end - n) : end]),
            @view(buffer.state[max(1, end - n) : end]),
            @view(buffer.action[max(1, end - n) : end]),
            Val(buffer.isdone[end]))
end

function update!(learner::DifferentialTDLearner{<:AbstractQApproximator, <:PolicyOrSelector, :SARSA}, states, actions, rewards, nextstates, nextactions, ::Val{false})
    n, α, β, Q = learner.n, learner.α, learner.β, learner.approximator
    if length(states) ≥ n
        s, a = states[1], actions[1]
        δ = sum(r -> r - learner.R̄, rewards) + Q(nextstates[end], nextactions[end]) - Q(s, a)
        learner.R̄ += β * δ
        update!(Q, s, a, α * δ)
        update!(learner.π, s, Q(s, Val(:argmax)))
    end
end