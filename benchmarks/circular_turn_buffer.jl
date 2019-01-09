using Ju
using BenchmarkTools

state_size = (80, 80, 1)
actions = 1:10

buffer = CircularSARDBuffer(10^5; state_type=Array{Float64, 3}, state_size=state_size)

push!(buffer, rand(state_size...), rand(actions))

println("\n", repeat('=', 50))
println("\n push! buffer\n")
display(@benchmark push!($buffer, 1.0, false, $(rand(state_size...)), $(rand(actions))))

println("\n", repeat('=', 50))
println("\n batch_sample buffer\n")
display(@benchmark batch_sample($buffer, 32))

println("\n", repeat('=', 50))
println("\n batch_sample buffer\n")
display(@benchmark batch_sample($buffer, 64))

function sample_N_batches(buffer, N, batch_size)
    for _ in 1:N
        batch_sample(buffer, batch_size)
    end
end

N, batch_size = 10, 32
println("\n", repeat('=', 50))
println("\n batch_sample (size=$batch_size) buffer $N times\n")
display(@benchmark sample_N_batches($buffer, $N, $batch_size))

# ==================================================

#  push! buffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     4.833 μs (0.00% GC)
#   median time:      5.250 μs (0.00% GC)
#   mean time:        5.314 μs (0.00% GC)
#   maximum time:     9.633 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     6

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  3.02 KiB
#   allocs estimate:  54
#   --------------
#   minimum time:     2.567 μs (0.00% GC)
#   median time:      2.800 μs (0.00% GC)
#   mean time:        3.121 μs (6.00% GC)
#   maximum time:     165.700 μs (95.98% GC)
#   --------------
#   samples:          10000
#   evals/sample:     9

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  4.14 KiB
#   allocs estimate:  54
#   --------------
#   minimum time:     2.937 μs (0.00% GC)
#   median time:      3.237 μs (0.00% GC)
#   mean time:        3.618 μs (5.56% GC)
#   maximum time:     183.113 μs (95.33% GC)
#   --------------
#   samples:          10000
#   evals/sample:     8

# ==================================================

#  batch_sample (size=32) buffer 10 times

# BenchmarkTools.Trial:
#   memory estimate:  30.16 KiB
#   allocs estimate:  540
#   --------------
#   minimum time:     25.399 μs (0.00% GC)
#   median time:      27.600 μs (0.00% GC)
#   mean time:        30.786 μs (4.66% GC)
#   maximum time:     1.562 ms (94.31% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1