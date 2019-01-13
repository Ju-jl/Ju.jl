struct NeuralNetworkQ{Tm}  <: AbstractQApproximator{Any, Int}
    model::Tm
end

(Q::NeuralNetworkQ)(s, ::Val{:dist}) = Q.model(s)
(Q::NeuralNetworkQ)(s) = Q(s, Val(:dist))
