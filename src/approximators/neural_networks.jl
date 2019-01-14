using Flux

struct NeuralNetworkQ{Tm, To, Tp}  <: AbstractQApproximator{Any, Int}
    model::Tm
    opt::To
    ps::Tp
    function NeuralNetworkQ(model, opt)
        m = gpu(model)
        ps = params(m)
        new(m, opt, ps)
    end
end

(Q::NeuralNetworkQ)(s, ::Val{:dist}) = Q.model(s)
(Q::NeuralNetworkQ)(s) = Q(s, Val(:dist))
(Q::NeuralNetworkQ)(s, ::Val{:argmax}) = map(i -> i[1], findmax(Q(s).data)[2])
(Q::NeuralNetworkQ)(s, ::Val{:max}) = maximum(Q(s), dims=1)
(Q::NeuralNetworkQ)(s, a) = Q(s)[a]

function update!(Q::NeuralNetworkQ, loss)
    back!(loss)
    update!(Q.opt, Q.ps)
end