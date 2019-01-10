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