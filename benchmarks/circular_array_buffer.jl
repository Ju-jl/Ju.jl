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
#   minimum time:     2.444 μs (0.00% GC)
#   median time:      2.594 μs (0.00% GC)
#   mean time:        3.054 μs (0.00% GC)
#   maximum time:     11.789 μs (0.00% GC)
#   --------------
#   samples:          10000
#   evals/sample:     9

# ==================================================

# getindex of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  50.30 KiB
#   allocs estimate:  8
#   --------------
#   minimum time:     6.200 μs (0.00% GC)
#   median time:      7.300 μs (0.00% GC)
#   mean time:        13.594 μs (25.39% GC)
#   maximum time:     11.704 ms (99.83% GC)
#   --------------
#   samples:          10000
#   evals/sample:     4

# ==================================================

# view element of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  288 bytes
#   allocs estimate:  7
#   --------------
#   minimum time:     601.149 ns (0.00% GC)
#   median time:      629.709 ns (0.00% GC)
#   mean time:        714.948 ns (8.06% GC)
#   maximum time:     305.314 μs (99.72% GC)
#   --------------
#   samples:          10000
#   evals/sample:     175

# ==================================================

# view elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  544 bytes
#   allocs estimate:  12
#   --------------
#   minimum time:     701.379 ns (0.00% GC)
#   median time:      749.655 ns (0.00% GC)
#   mean time:        883.540 ns (10.07% GC)
#   maximum time:     424.337 μs (99.59% GC)
#   --------------
#   samples:          10000
#   evals/sample:     145

# ==================================================

# view elements of CircularArrayBuffer

# BenchmarkTools.Trial:
#   memory estimate:  576 bytes
#   allocs estimate:  12
#   --------------
#   minimum time:     701.429 ns (0.00% GC)
#   median time:      737.857 ns (0.00% GC)
#   mean time:        877.624 ns (8.98% GC)
#   maximum time:     340.946 μs (99.70% GC)
#   --------------
#   samples:          10000
#   evals/sample:     140