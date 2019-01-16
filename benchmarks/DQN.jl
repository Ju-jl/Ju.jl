using Ju
using Flux

env = CartPoleEnv()
input_length = length(sample(observationspace(env)))
output_length = length(actionspace(env))

model = Chain(
    Dense(input_length, 128, relu),
    Dense(128, 128, relu),
    Dense(128, output_length)
)

ϵ, γ, buffer_size, batch_size = 0.01, 0.99, 10^4, 32

Q = NeuralNetworkQ(model, ADAM())
learner = DQN(Q, EpsilonGreedySelector(ϵ); γ=γ, batch_size=batch_size)
buffer = CircularSARDBuffer(buffer_size; state_type=Array{Float64, 1}, state_size=(input_length,))
agent = Agent(learner, buffer)