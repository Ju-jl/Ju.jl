using Flux

struct NeuralNetworkQ{Tm, To, Tp}  <: AbstractQApproximator{Any, Int}
    model::Tm
    opt::To
    ps::Tp
end

function NeuralNetworkQ(model, opt)
    ps = params(model)
    NeuralNetworkQ(model, opt, ps)
end

(Q::NeuralNetworkQ)(s, ::Val{:dist}) = Q.model(s)
(Q::NeuralNetworkQ)(s) = Q(s, Val(:dist))

function (Q::NeuralNetworkQ)(s, ::Val{:max})
    dist = Q(s)
    maximum(dist, dims=ndims(dist))
end

function (Q::NeuralNetworkQ)(s, ::Val{:argmax})
end

function update!(Q::NeuralNetworkQ, loss)
    back!(loss)
    update!(Q.opt, Q.ps)
end