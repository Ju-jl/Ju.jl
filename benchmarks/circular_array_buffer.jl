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

n = 16
println("\n", repeat('=', 50))
println("\nview $n elements of CircularArrayBuffer\n")
display(@benchmark view($buffer, $(sample(1:length(buffer), n))))

n = 32
println("\n", repeat('=', 50))
println("\nview $n elements of CircularArrayBuffer\n")
display(@benchmark view($buffer, $(sample(1:length(buffer), n))))

# ==================================================

# push! into CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  0 bytes
#   allocs estimate:  0
#   --------------
#   minimum time:     2.411 μs (0.00% GC)
#   median time:      2.456 μs (0.00% GC)
#   mean time:        2.464 μs (0.00% GC)
#   maximum time:     5.133 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     9

# ==================================================

# getindex of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  50.08 KiB
#   allocs estimate:  2
#   --------------
#   minimum time:     5.540 μs (0.00% GC)
#   median time:      15.940 μs (0.00% GC)
#   mean time:        16.852 μs (27.09% GC)
#   maximum time:     8.680 ms (99.80% GC)
#   --------------
#   samples:          10000
#   evals/sample:     5

# ==================================================

# view element of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  64 bytes
#   allocs estimate:  1
#   --------------
#   minimum time:     12.913 ns (0.00% GC)
#   median time:      14.916 ns (0.00% GC)
#   mean time:        23.723 ns (30.27% GC)
#   maximum time:     45.953 μs (99.93% GC)
#   --------------
#   samples:          10000
#   evals/sample:     999

# ==================================================

# view 16 elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  400 bytes
#   allocs estimate:  6
#   --------------
#   minimum time:     96.007 ns (0.00% GC)
#   median time:      101.996 ns (0.00% GC)
#   mean time:        130.636 ns (17.69% GC)
#   maximum time:     45.819 μs (99.68% GC)
#   --------------
#   samples:          10000
#   evals/sample:     952

# ==================================================

# view 32 elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  528 bytes
#   allocs estimate:  6
#   --------------
#   minimum time:     109.849 ns (0.00% GC)
#   median time:      123.340 ns (0.00% GC)
#   mean time:        166.186 ns (18.24% GC)
#   maximum time:     51.037 μs (99.33% GC)
#   --------------
#   samples:          10000
#   evals/sample:     934