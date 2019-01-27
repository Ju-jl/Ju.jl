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

##############################
# Double DQN
##############################

mutable struct DoubleDQN{Ta<:NeuralNetworkQ, Tt<:NeuralNetworkQ, Tp<:AbstractActionSelector, Tf, Tl} <: AbstractModelFreeLearner
    Qₐ::Ta
    Qₜ::Tt
    action_selector::Tp
    update_period::Int
    target_update_period::Int
    training_steps::Int
    batch_size::Int
    loss_fun::Tf
    loss::Tl
    γ::Float64
    function DoubleDQN(Qₐ::Ta, Qₜ::Tt, action_selector::Tp; update_period=4, target_update_period=8000, batch_size=32, loss_fun=Flux.mse, γ=0.99) where {Ta, Tt, Tp}
        init_loss = is_using_gpu ? Float32(0.) : Float64(0.)
        new{Ta, Tt, Tp, typeof(loss_fun), typeof(init_loss)}(Qₐ, Qₜ, action_selector, update_period, target_update_period, 0, batch_size, loss_fun, init_loss, γ)
    end
end

(learner::DoubleDQN)(s) = learner.Qₐ(gpu(s)) |> learner.action_selector

function update!(learner::DoubleDQN, buffer::CircularSARDBuffer)
    Qₐ, Qₜ, γ, batch_size = learner.Qₐ, learner.Qₜ, learner.γ, learner.batch_size
    learner.training_steps += 1

    if length(buffer) > batch_size
        if learner.training_steps % learner.update_period == 0
            (s, a, r, d, s′), _ = sample(buffer, batch_size)
            s, r, d, s′ = gpu(s), gpu(r), gpu(d), gpu(s′)  

            q, q′ = Qₐ(s, a), Qₜ(s′, Qₐ(s′, Val(:argmax)))  # note the difference compared to DQN
            G = @. r + γ * q′ * (1 - d)
            loss = learner.loss_fun(G, q)
            learner.loss = loss.data
            update!(Qₐ, loss)
        end

        if learner.training_steps % learner.target_update_period == 0
            Flux.loadparams!(Qₜ.model, Qₐ.ps)
        end
    end
end