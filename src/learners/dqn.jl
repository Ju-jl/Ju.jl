using Flux

const PolicyOrSelector = Union{AbstractPolicy, AbstractActionSelector}

mutable struct DQN{Tn<:NeuralNetworkQ, Tp<:PolicyOrSelector, Tf, Tl<:Union{Float32, Float64}} <: AbstractModelFreeLearner 
    Q::Tn
    π::Tp
    γ::Float64
    batch_size::Int
    loss_fun::Tf
    loss::Tl
    function DQN(Q::TQ, π::Tp; γ=0.99, batch_size=32, loss_fun=Flux.mse) where {TQ, Tp}
        init_loss = is_using_gpu ? Float32(0.) : Float64(0.)
        new{TQ, Tp, typeof(loss_fun), typeof(init_loss)}(Q, π, γ, batch_size, loss_fun, init_loss)
    end
end

(learner::DQN{<:NeuralNetworkQ, <:AbstractActionSelector})(s) = learner.Q(gpu(s)) |> learner.π
(learner::DQN{<:NeuralNetworkQ, <:AbstractPolicy})(s) = learner.π(s)
(learner::DQN{<:NeuralNetworkQ, <:AbstractPolicy})(s, ::Val{:dist}) = learner.π(s, Val(:dist))

function update!(learner::DQN{<:NeuralNetworkQ, <:AbstractActionSelector}, buffer::CircularSARDBuffer)
    Q, π, γ, batch_size, loss_fun = learner.Q, learner.π, learner.γ, learner.batch_size, learner.loss_fun

    if length(buffer) > batch_size
        (s, a, r, d, s′), _ = sample(buffer, batch_size)
        s, r, d, s′ = gpu(s), gpu(r), gpu(d), gpu(s′)  

        q, q′ = Q(s, a), Q(s′, Val(:max))
        G = @. r + γ * q′ * (1 - d)
        loss = loss_fun(G, q)
        learner.loss = loss.data
        update!(Q, loss)
        # update!(π, s, Q(s, Val(:argmax)))  # π isa AbstractPolicy
    end
end