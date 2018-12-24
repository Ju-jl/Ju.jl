struct QLearner{Tapp<:AbstractQApproximator, Ts<:AbstractActionSelector, Tα<:Union{Function, Float64}} <: AbstractModelFreeLearner 
    Q::Tapp
    selector::Ts
    α::Tα
    γ::Float64
end

function (learner::QLearner{<:AbstractQApproximator{Ts}})(s::Ts) where Ts 
    s |> learner.Q |> learner.selector
end

function update!(learner::QLearner{<:AbstractQApproximator{Ts, Ta}}, α::Float64, s::Ts, a::Ta, r::Float64, s′::Ts) where {Ts, Ta}
    γ, Q = learner.γ, learner.Q
    error = α * (r + γ * Q(s′, Val(:max)) - Q(s, a))
    update!(Q, s, a, error)
end

update!(learner::QLearner, buffer::SARDBuffer) =  update!(learner, buffer.state[end-1], buffer.action[end-1], buffer.reward[end], buffer.state[end])
update!(learner::QLearner, buffer::Union{SARDSABuffer,SARDSBuffer}) =  update!(learner, buffer.state[end], buffer.action[end], buffer.reward[end], buffer.nextstate[end])
update!(learner::QLearner{<:AbstractQApproximator{Ts, Ta}, <:AbstractActionSelector, Float64}, s::Ts,  a::Ta,  r::Float64,  s′::Ts) where {Ts, Ta} = update!(learner, learner.α, s,  a,  r,  s′)
update!(learner::QLearner{<:AbstractQApproximator{Ts, Ta}, <:AbstractActionSelector, <:Function}, s::Ts,  a::Ta,  r::Float64,  s′::Ts) where {Ts, Ta} = update!(learner, learner.α((s,  a)), s,  a,  r,  s′)