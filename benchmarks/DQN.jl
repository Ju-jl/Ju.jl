# using CuArrays  # Enable GPU
using Ju
using Flux
using Plots
gr()

env = CartPoleEnv()
input_length = length(sample(observationspace(env)))
output_length = length(actionspace(env))

model = Chain(
    Dense(input_length, 128, relu),
    Dense(128, 128, relu),
    Dense(128, output_length)
)

ϵ = exp_decay(init=1.0, λ=1.0, decay_step=500, clip=0.01)
γ, buffer_size, batch_size = 0.99, 10^3, 32

Q = NeuralNetworkQ(model, ADAM())
learner = DQN(Q, EpsilonGreedySelector(ϵ); γ=γ, batch_size=batch_size)
buffer = CircularSARDBuffer(buffer_size; state_type=Array{Float64, 1}, state_size=(input_length,))
agent = Agent(learner, buffer)

callbacks = (stop_at_step(10000), rewards_of_each_episode())
train!(env, agent;callbacks=callbacks)