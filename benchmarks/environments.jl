using Ju
using BenchmarkTools

function run_env(env, steps=10^3)
    reset!(env)
    while steps > 0
        if observe(env).isdone
            reset!(env)
        end
        env(sample(actionspace(env)))
        steps -= 1
    end
end

function benchmark_env(env, nsteps=10^3)
    println("\n", repeat('=', 50))
    println("\nreset!\n")
    display(@benchmark reset!($env))
    println("\n", repeat('=', 50))
    println("\nobserve\n")
    display(@benchmark observe($env))
    println("\n", repeat('=', 50))
    println("\nrun $nsteps steps\n")
    display(@benchmark run_env($env, $nsteps))
    println("\n", repeat('=', 50))
end

benchmark_env(CartPoleEnv())

# julia --project=benchmarks -e 'include("benchmarks/environments.jl")'

# ==================================================

# reset!

# BenchmarkTools.Trial: 
#   memory estimate:  32 bytes
#   allocs estimate:  1
#   --------------
#   minimum time:     42.047 ns (0.00% GC)
#   median time:      43.184 ns (0.00% GC)
#   mean time:        53.883 ns (13.18% GC)
#   maximum time:     47.807 μs (99.89% GC)
#   --------------
#   samples:          10000
#   evals/sample:     991
# ==================================================

# observe

# BenchmarkTools.Trial: 
#   memory estimate:  32 bytes
#   allocs estimate:  1
#   --------------
#   minimum time:     8.863 ns (0.00% GC)
#   median time:      11.440 ns (0.00% GC)
#   mean time:        19.778 ns (37.83% GC)
#   maximum time:     51.952 μs (99.97% GC)
#   --------------
#   samples:          10000
#   evals/sample:     999
# ==================================================

# run 1000 steps

# BenchmarkTools.Trial: 
#   memory estimate:  31.25 KiB
#   allocs estimate:  1000
#   --------------
#   minimum time:     73.355 μs (0.00% GC)
#   median time:      74.090 μs (0.00% GC)
#   mean time:        86.768 μs (8.98% GC)
#   maximum time:     52.824 ms (99.83% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
# ==================================================
