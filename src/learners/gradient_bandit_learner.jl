using NNlib:softmax

"""
    struct GradientBanditLearner{Ts<:AbstractActionSelector, Tb<:Union{Float64, Function}} <: AbstractModelFreeLearner 
        Q::TabularQ 
        selector::Ts
        α::Float64
        baseline::Tb
    end

See more details at Section (2.8) on Page 37 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct GradientBanditLearner{Ts<:AbstractActionSelector, Tb<:Union{Float64, Function}} <: AbstractModelFreeLearner 
    Q::TabularQ 
    selector::Ts
    α::Float64
    baseline::Tb
end

function (learner::GradientBanditLearner)(s::Int) 
    s |> learner.Q |> softmax |> learner.selector
end

function update!(learner::GradientBanditLearner, s, a, R, R̄)
    α = learner.α
    prob = s |> learner.Q |> softmax
    errors = α * (R - R̄) .* (onehot(length(prob), a, Float64) .- prob)
    update!(learner.Q, s, errors)
end

update!(learner::GradientBanditLearner{<:AbstractActionSelector, Float64}, s, a, r) = update!(learner, s, a, r, learner.baseline)
update!(learner::GradientBanditLearner{<:AbstractActionSelector, <:Function}, s, a, r) = update!(learner, s, a, r, learner.baseline(r))

function update!(learner::GradientBanditLearner, buffer::SARDBuffer)
    update!(learner, buffer.state[end-1], buffer.action[end-1], buffer.reward[end])
end

function update!(learner::GradientBanditLearner, buffer::Union{SARDSABuffer, SARDSBuffer})
    update!(learner, buffer.state[end], buffer.action[end], buffer.reward[end])
end