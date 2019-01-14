using Flux

const PolicyOrSelector = Union{AbstractPolicy, AbstractActionSelector}

struct DQN{Tn<:NeuralNetworQ, Tp<:PolicyOrSelector} <: AbstractModelFreeLearner 
    Q::Tn
    π::Tp
    γ::Float64
    batch_size::Int
end

(learner::DQN{<:NeuralNetworQ, <:AbstractActionSelector})(s) = learner.Q(s) |> learner.π
(learner::DQN{<:NeuralNetworQ, <:AbstractPolicy})(s) = learner.π(s)
(learner::DQN{<:NeuralNetworQ, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::DQN{<:NeuralNetworQ, <:AbstractActionSelector}, buffer::CircularSARDBuffer)
    Q, π, γ, batch_size = learner.Q, learner.π, learner.γ, learner.batch_size

    if length(buffer) > batch_size
        (s, a, r, d, s′), _ = sample(buffer, batch_size)
        s, r, d, s′ = gpu(s), gpu(r), gpu(d), gpu(s′)  
        a = map(i -> CartesianIndex(a[i], i), eachindex(a))  # nactions * batch_size

        q, q′ = Q(s, a), Q(s′, Val(:max))
        G = @. r + γ * q′ * (1 - d)
        loss = mse(G, q)
        update!(Q, loss)
        # update!(π, s, Q(s, Val(:argmax)))  # π isa AbstractPolicy
    end
end