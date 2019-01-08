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

# ==================================================

#  push! buffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     4.867 μs (0.00% GC)
#   median time:      5.267 μs (0.00% GC)
#   mean time:        5.336 μs (0.00% GC)
#   maximum time:     23.083 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     6

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  2.17 KiB
#   allocs estimate:  34
#   --------------
#   minimum time:     897.324 ns (0.00% GC)
#   median time:      986.486 ns (0.00% GC)
#   mean time:        1.225 μs (8.48% GC)
#   maximum time:     37.987 μs (94.69% GC)
#   --------------
#   samples:          10000
#   evals/sample:     37

# ==================================================

#  batch_sample buffer

# BenchmarkTools.Trial:
#   memory estimate:  3.30 KiB
#   allocs estimate:  34
#   --------------
#   minimum time:     1.230 μs (0.00% GC)
#   median time:      1.400 μs (0.00% GC)
#   mean time:        1.629 μs (7.47% GC)
#   maximum time:     97.510 μs (97.77% GC)
#   --------------
#   samples:          10000
#   evals/sample:     10