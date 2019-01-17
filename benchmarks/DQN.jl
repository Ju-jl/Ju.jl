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

ϵ, γ, buffer_size, batch_size = 1., 0.99, 10^3, 32

Q = NeuralNetworkQ(model, ADAM())
learner = DQN(Q, EpsilonGreedySelector(ϵ); γ=γ, batch_size=batch_size)
buffer = CircularSARDBuffer(buffer_size; state_type=Array{Float64, 1}, state_size=(input_length,))
agent = Agent(learner, buffer)

function change_epsilon()
    i = 0
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    epsilon_by_frame(i) = epsilon_final + (epsilon_start - epsilon_final) * exp(-1. * i / epsilon_decay)
    function f(env, agent)
        i += 1
        agent.learner.π.ϵ = epsilon_by_frame(i)
    end
end

callbacks = (stop_at_step(10000), rewards_of_each_episode(), change_epsilon())
# train!(env, agent;callbacks=callbacks)