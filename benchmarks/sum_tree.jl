using Ju
using BenchmarkTools

t = SumTree(10^5)

println("\n", repeat('=', 50))
println("\n push! priority into sum tree \n")
display(@benchmark push!($t, $(rand())))

batch_size = 32

println("\n", repeat('=', 50))
println("\n sample $batch_size \n")
display(@benchmark sample($t, $batch_size))

batch_size = 64

println("\n", repeat('=', 50))
println("\n sample $batch_size \n")
display(@benchmark sample($t, $batch_size))

# ==================================================

#  push! priority into sum tree

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     29.025 ns (0.00% GC)
#   median time:      29.199 ns (0.00% GC)
#   mean time:        31.569 ns (0.00% GC)
#   maximum time:     86.101 ns (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     995

# ==================================================

#  sample 32

# BenchmarkTools.Trial:
#   memory estimate:  704 bytes
#   allocs estimate:  3
#   --------------
#   minimum time:     4.863 μs (0.00% GC)
#   median time:      5.300 μs (0.00% GC)
#   mean time:        6.415 μs (14.46% GC)
#   maximum time:     7.253 ms (99.87% GC)
#   --------------
#   samples:          10000
#   evals/sample:     7

# ==================================================

#  sample 64

# BenchmarkTools.Trial:
#   memory estimate:  1.25 KiB
#   allocs estimate:  3
#   --------------
#   minimum time:     9.076 μs (0.00% GC)
#   median time:      10.344 μs (0.00% GC)
#   mean time:        10.990 μs (0.00% GC)
#   maximum time:     46.137 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1