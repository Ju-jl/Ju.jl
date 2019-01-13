using Ju
using BenchmarkTools

state_size = (80, 80, 1)
actions = 1:10

buffer = CircularSARDBuffer(10^5; state_type=Array{Float64, 3}, state_size=state_size)

push!(buffer, rand(state_size...), rand(actions))

println("\n", repeat('=', 50))
println("\n push! buffer\n")
display(@benchmark push!($buffer, 1.0, false, $(rand(state_size...)), $(rand(actions))))

batch_size = 32
println("\n", repeat('=', 50))
println("\n batch_sample buffer (batch_size=$batch_size)\n")
display(@benchmark batch_sample($buffer, $batch_size))

batch_size = 64
println("\n", repeat('=', 50))
println("\n batch_sample buffer (batch_size=$batch_size)\n")
display(@benchmark batch_sample($buffer, $batch_size))

function sample_N_batches(buffer, N, batch_size)
    for _ in 1:N
        batch_sample(buffer, batch_size)
    end
end

N, batch_size = 10, 32
println("\n", repeat('=', 50))
println("\n batch_sample buffer (batch_size=$batch_size) $N times\n")
display(@benchmark sample_N_batches($buffer, $N, $batch_size))

N, batch_size = 10, 64
println("\n", repeat('=', 50))
println("\n batch_sample buffer (batch_size=$batch_size) $N times\n")
display(@benchmark sample_N_batches($buffer, $N, $batch_size))

# ==================================================

#  push! buffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     4.657 μs (0.00% GC)
#   median time:      5.114 μs (0.00% GC)
#   mean time:        5.138 μs (0.00% GC)
#   maximum time:     9.500 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     7

# ==================================================

#  batch_sample buffer (batch_size=32)

# BenchmarkTools.Trial:
#   memory estimate:  3.13 KiB
#   allocs estimate:  34
#   --------------
#   minimum time:     958.163 ns (0.00% GC)
#   median time:      1.081 μs (0.00% GC)
#   mean time:        1.239 μs (11.31% GC)
#   maximum time:     25.488 μs (94.94% GC)
#   --------------
#   samples:          10000
#   evals/sample:     43

# ==================================================

#  batch_sample buffer (batch_size=64)

# BenchmarkTools.Trial:
#   memory estimate:  5.09 KiB
#   allocs estimate:  34
#   --------------
#   minimum time:     1.270 μs (0.00% GC)
#   median time:      1.460 μs (0.00% GC)
#   mean time:        1.709 μs (9.25% GC)
#   maximum time:     80.030 μs (95.69% GC)
#   --------------
#   samples:          10000
#   evals/sample:     10

# ==================================================

#  batch_sample buffer (batch_size=32) 10 times

# BenchmarkTools.Trial:
#   memory estimate:  30.47 KiB
#   allocs estimate:  320
#   --------------
#   minimum time:     8.699 μs (0.00% GC)
#   median time:      10.000 μs (0.00% GC)
#   mean time:        13.832 μs (9.60% GC)
#   maximum time:     1.497 ms (98.13% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1

# ==================================================

#  batch_sample buffer (batch_size=64) 10 times

# BenchmarkTools.Trial:
#   memory estimate:  50.16 KiB
#   allocs estimate:  320
#   --------------
#   minimum time:     12.599 μs (0.00% GC)
#   median time:      14.301 μs (0.00% GC)
#   mean time:        16.655 μs (9.19% GC)
#   maximum time:     741.201 μs (96.80% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1