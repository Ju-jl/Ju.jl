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
    println(repeat('=', 50))
    println("\nreset!\n")
    display(@benchmark reset!($env))
    println(repeat('=', 50))
    println("\nobserve\n")
    display(@benchmark observe($env))
    println(repeat('=', 50))
    println("\nrun $nsteps steps\n")
    display(@benchmark run_env($env))
    println(repeat('=', 50))
end

benchmark_env(CartPoleEnv())