using Flux

const PolicyOrSelector = Union{AbstractPolicy, AbstractActionSelector}

struct DQN{Tn<:NeuralNetworkQ, Tp<:PolicyOrSelector} <: AbstractModelFreeLearner 
    Q::Tn
    π::Tp
    γ::Float64
    batch_size::Int
    DQN(Q::TQ, π::Tp; γ=0.99, batch_size=32) where {TQ, Tp} = new{TQ, Tp}(Q, π, γ, batch_size)
end

(learner::DQN{<:NeuralNetworkQ, <:AbstractActionSelector})(s) = learner.Q(gpu(s)) |> learner.π
(learner::DQN{<:NeuralNetworkQ, <:AbstractPolicy})(s) = learner.π(s)
(learner::DQN{<:NeuralNetworkQ, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::DQN{<:NeuralNetworkQ, <:AbstractActionSelector}, buffer::CircularSARDBuffer)
    Q, π, γ, batch_size = learner.Q, learner.π, learner.γ, learner.batch_size

    if length(buffer) > batch_size
        (s, a, r, d, s′), _ = sample(buffer, batch_size)
        s, r, d, s′ = gpu(s), gpu(r), gpu(d), gpu(s′)  

        q, q′ = Q(s, a), Q(s′, Val(:max))
        G = @. r + γ * q′ * (1 - d)
        loss = Flux.mse(G, q)
        update!(Q, loss)
        # update!(π, s, Q(s, Val(:argmax)))  # π isa AbstractPolicy
    end
end