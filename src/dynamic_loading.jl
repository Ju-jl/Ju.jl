import Flux:gpu
using CuArrays

gpu(x::SubArray) = CuArray{Float32}(x)

const is_using_gpu = true