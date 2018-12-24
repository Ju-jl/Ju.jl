"""
    FourierV <: AbstractVApproximator{Int}

    struct FourierV <: AbstractVApproximator{Int}
        weights::Vector{Float64}
    end

Using Fourier cosine basis to approximate the state value.
`weights` is the featur vector.

See more details at Section (9.5.2) on Page 211 of the book *Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.*
"""
struct FourierV <: AbstractVApproximator{Int}
    weights::Vector{Float64}
end

"""
    FourierV(order::Int)

By specifying the `order`, feature vector will be initialized with 0.
"""
FourierV(order::Int) = FourierV(zeros(Float64, order+1))

function (f::FourierV)(s)
    sum(w * cos((i-1) * π * s) for (i, w) in enumerate(f.weights))
end

function update!(f::FourierV, s, e)
    for i in 1:length(f.weights)
        f.weights[i] += e * cos((i-1) * π * s)
    end
end