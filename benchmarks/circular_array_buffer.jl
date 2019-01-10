using Ju
using BenchmarkTools
using StatsBase:sample

buffer = CircularArrayBuffer{Array{Float64, 3}}(10, (80, 80, 1));

println("\n", repeat('=', 50))
println("\npush! into CircularArrayBuffer\n")
display(@benchmark push!($buffer, $(rand(80, 80, 1))))

println("\n", repeat('=', 50))
println("\ngetindex of CircularArrayBuffer\n")
display(@benchmark $buffer[$(rand(1:length(buffer)))])

println("\n", repeat('=', 50))
println("\nview element of CircularArrayBuffer\n")
display(@benchmark view($buffer, $(rand(1:length(buffer)))))

println("\n", repeat('=', 50))
println("\nview elements of CircularArrayBuffer\n")
display(@benchmark view($buffer, $(sample(1:length(buffer), 5))))

println("\n", repeat('=', 50))
println("\nview elements of CircularArrayBuffer\n")
display(@benchmark view($buffer, $(sample(1:length(buffer), 10))))

# ==================================================

# push! into CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     2.422 μs (0.00% GC)
#   median time:      2.467 μs (0.00% GC)
#   mean time:        2.478 μs (0.00% GC)
#   maximum time:     5.378 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     9

# ==================================================

# getindex of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  50.08 KiB
#   allocs estimate:  2
#   --------------
#   minimum time:     5.300 μs (0.00% GC)
#   median time:      16.560 μs (0.00% GC)
#   mean time:        16.439 μs (21.83% GC)
#   maximum time:     716.460 μs (96.83% GC)
#   --------------
#   samples:          10000
#   evals/sample:     5

# ==================================================

# view element of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  64 bytes
#   allocs estimate:  1
#   --------------
#   minimum time:     13.199 ns (0.00% GC)
#   median time:      15.000 ns (0.00% GC)
#   mean time:        17.064 ns (8.11% GC)
#   maximum time:     530.100 ns (96.81% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1000

# ==================================================

# view elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  320 bytes
#   allocs estimate:  6
#   --------------
#   minimum time:     78.189 ns (0.00% GC)
#   median time:      84.877 ns (0.00% GC)
#   mean time:        104.887 ns (15.44% GC)
#   maximum time:     1.834 μs (94.18% GC)
#   --------------
#   samples:          10000
#   evals/sample:     972

# ==================================================

# view elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  352 bytes
#   allocs estimate:  6
#   --------------
#   minimum time:     86.848 ns (0.00% GC)
#   median time:      93.111 ns (0.00% GC)
#   mean time:        110.418 ns (15.03% GC)
#   maximum time:     1.325 μs (85.43% GC)
#   --------------
#   samples:          10000
#   evals/sample:     958