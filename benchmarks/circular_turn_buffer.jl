using Ju
using BenchmarkTools

state_size = (80, 80, 1)
actions = 1:10

buffer = CircularSARDBuffer(10^5; state_type=Array{Float64, 3}, state_size=state_size)

push!(buffer, rand(state_size...), rand(actions))

println("\n", repeat('=', 50))
println("\n push! buffer\n")
display(@benchmark push!($buffer, 1.0, false, $(rand(state_size...)), $(rand(actions))))

batch_indices = BatchIndices(32)
println("\n", repeat('=', 50))
println("\n batch_sample buffer\n")
display(@benchmark batch_sample($buffer, $batch_indices))

batch_indices = BatchIndices(64)
println("\n", repeat('=', 50))
println("\n batch_sample buffer\n")
display(@benchmark batch_sample($buffer, batch_indices))

function sample_N_batches(buffer, N, batch_size)
    for _ in 1:N
        batch_sample(buffer, batch_size)
    end
end

N, batch_indices = 10, BatchIndices(32)
println("\n", repeat('=', 50))
println("\n batch_sample buffer $N times\n")
display(@benchmark sample_N_batches($buffer, $N, $batch_indices))

N, batch_indices = 10, BatchIndices(64)
println("\n", repeat('=', 50))
println("\n batch_sample buffer $N times\n")
display(@benchmark sample_N_batches($buffer, $N, $batch_indices))

# ==================================================

#  push! buffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     3.072 μs (0.00% GC)
#   median time:      3.252 μs (0.00% GC)
#   mean time:        3.405 μs (0.00% GC)
#   maximum time:     20.181 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     8

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  736 bytes
#   allocs estimate:  22
#   --------------
#   minimum time:     1.057 μs (0.00% GC)
#   median time:      1.194 μs (0.00% GC)
#   mean time:        1.343 μs (2.89% GC)
#   maximum time:     132.621 μs (98.34% GC)
#   --------------
#   samples:          10000
#   evals/sample:     10

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  736 bytes
#   allocs estimate:  22
#   --------------
#   minimum time:     1.784 μs (0.00% GC)
#   median time:      1.951 μs (0.00% GC)
#   mean time:        2.129 μs (1.96% GC)
#   maximum time:     145.707 μs (97.78% GC)
#   --------------
#   samples:          10000
#   evals/sample:     10

# ==================================================

#  batch_sample buffer 10 times

# BenchmarkTools.Trial:
#   memory estimate:  7.19 KiB
#   allocs estimate:  220
#   --------------
#   minimum time:     10.459 μs (0.00% GC)
#   median time:      11.846 μs (0.00% GC)
#   mean time:        12.808 μs (3.40% GC)
#   maximum time:     1.528 ms (98.98% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1

# ==================================================

#  batch_sample buffer 10 times

# BenchmarkTools.Trial:
#   memory estimate:  7.19 KiB
#   allocs estimate:  220
#   --------------
#   minimum time:     17.639 μs (0.00% GC)
#   median time:      20.077 μs (0.00% GC)
#   mean time:        21.642 μs (2.07% GC)
#   maximum time:     1.575 ms (98.49% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1