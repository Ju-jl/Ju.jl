"""
    struct ReinforceLearner{Tp<:AbstractPolicy}  <: AbstractModelFreeLearner 
        π::Tp
        α::Float64
        γ::Float64
    end

See more details at Section (13.3) on Page 326 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct ReinforceLearner{Tp<:AbstractPolicy}  <: AbstractModelFreeLearner 
    π::Tp
    α::Float64
    γ::Float64
end

(learner::ReinforceLearner)(s) = learner.π(s)
(learner::ReinforceLearner)(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::ReinforceLearner{<:LinearPolicy}, buffer::EpisodeSARDBuffer)
    π, α, γ = learner.π, learner.α, learner.γ
    if isfull(buffer)
        # need to allocate first, because `Reduction` do not support `Iterators.Reverse`
        reversed_G = collect(reverse_discounted_rewards(buffer.reward, γ))
        γᵗ = 1.
        for (G, Sₜ, Aₜ) in zip(Iterators.reverse(reversed_G), @view(buffer.state[1:end-1]), @view(buffer.action[1:end-1]))
            @views error = α * γᵗ * G * (π.features[Sₜ, Aₜ, :] .- sum(b -> π(Sₜ, b) * π.features[Sₜ, b, :],axes(π.features, 2)))
            update!(π, error)
            γᵗ *= γ
        end
    end
end

"""
    mutable struct ReinforceBaselineLearner{Tapp<:AbstractApproximator, Tp<:AbstractPolicy}  <: AbstractModelFreeLearner 
        approximator::Tapp
        π::Tp
        αʷ::Float64
        αᶿ::Float64
        γ::Float64
    end

See more details at Section (13.4) on Page 330 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
mutable struct ReinforceBaselineLearner{Tapp<:AbstractApproximator, Tp<:AbstractPolicy}  <: AbstractModelFreeLearner 
    approximator::Tapp
    π::Tp
    αʷ::Float64
    αᶿ::Float64
    γ::Float64
end

(learner::ReinforceBaselineLearner)(s) = learner.π(s)
(learner::ReinforceBaselineLearner)(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::ReinforceBaselineLearner{<:AbstractVApproximator, <:LinearPolicy}, buffer::EpisodeSARDBuffer)
    V, π, αᶿ, αʷ, γ = learner.approximator, learner.π, learner.αᶿ, learner.αʷ, learner.γ
    if isfull(buffer)
        # need to allocate first, because `Reduction` do not support `Iterators.Reverse`
        reversed_G = collect(reverse_discounted_rewards(buffer.reward, γ))
        γᵗ = 1.
        for (G, Sₜ, Aₜ) in zip(Iterators.reverse(reversed_G), @view(buffer.state[1:end-1]), @view(buffer.action[1:end-1]))
            δ = G - V(Sₜ)
            update!(V, Sₜ, αʷ * δ)
            @views error = αᶿ * γᵗ * δ * (π.features[Sₜ, Aₜ, :] .- sum(b -> π(Sₜ, b) * π.features[Sₜ, b, :],axes(π.features, 2)))
            update!(π, error)
            γᵗ *= γ
        end
    end
end