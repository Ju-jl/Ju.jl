using Flux

struct NeuralNetworkQ{Tm, To, Tp}  <: AbstractQApproximator{Any, Int}
    model::Tm
    opt::To
    ps::Tp
    function NeuralNetworkQ(model::Tm, opt::To) where {Tm, To}
        m = gpu(model)
        ps = params(m)
        new{typeof(m), To, typeof(ps)}(m, opt, ps)
    end
end

(Q::NeuralNetworkQ)(s, ::Val{:dist}) = Q.model(s)
(Q::NeuralNetworkQ)(s) = Q(s, Val(:dist))
(Q::NeuralNetworkQ)(s, ::Val{:argmax}) = map(i -> i[1], findmax(Q(s).data, dims=1)[2])
(Q::NeuralNetworkQ)(s, ::Val{:max}) = dropdims(maximum(Q(s), dims=1), dims=1)
(Q::NeuralNetworkQ)(s, a::Int) = Q(s)[a]

function (Q::NeuralNetworkQ)(s, a::AbstractArray{Int, 1})
    dist = Q(s)
    inds = CartesianIndex.(a, axes(dist, 2))
    dist[inds]
end

function update!(Q::NeuralNetworkQ, loss)
    Flux.back!(loss)
    Flux.Optimise.update!(Q.opt, Q.ps)
end