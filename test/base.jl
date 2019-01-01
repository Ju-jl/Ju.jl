using SparseArrays

@testset "base" begin

@testset "helper functions" begin
    @testset "findallmax" begin
        @test findallmax([-Inf, -Inf, -Inf]) == (-Inf, [1, 2, 3])
        @test findallmax([Inf, Inf, Inf]) == (Inf, [1, 2, 3])
        @test findallmax([0,1,2,1,2,1,0]) == (2, [3, 5])
        @test begin
            max_val, inds = findallmax([NaN, NaN, NaN])
            isnan(max_val) && inds == [1, 2, 3]
        end
    end
    @testset "onehot" begin
        @test onehot(5, 3) == [0, 0, 1, 0, 0]
        @test onehot(5, 3, Float64) == [0., 0., 1., 0., 0.]
        @test onehot(5, 3, Float64; isdense=false) == SparseVector([0., 0., 1., 0., 0.])
    end
    @testset "deletefirst!" begin
        @test deletefirst!([1,2,3,2,1], 1) == [2,3,2,1]
        @test deletefirst!([1,2,3,2,1], 2) == [1,3,2,1]
        @test deletefirst!([1,2,3,2,1], 3) == [1,2,2,1]
    end
    @testset "discounted reward" begin
        @test discounted_reward([2., 3., 4.], 0.9) ≈ 2. + 0.9 * 3. + 0.9 * 0.9 * 4
    end
    @testset "reverse_discounted_rewards" begin
        rewards, γ = [2., 3., 4.], 0.9
        @test collect(reverse_discounted_rewards(rewards, γ)) == reverse([discounted_reward(rewards, γ), discounted_reward(rewards[2:end], γ), discounted_reward(rewards[3:end], γ)])
    end
    @testset "importance_weight" begin
        actions = [1, 2]
        π_target = DeterministicPolicy([1,2,1,2], length(actions))
        π_behavior = RandomPolicy([0.5, 0.5])

        @test importance_weight(π_target, π_behavior, [1, 2, 3, 4], [1, 2, 1, 2]) == 16.
        @test importance_weight(π_target, π_behavior, [1, 2, 3, 4], [1, 1, 1, 1]) == 0.
    end
    @testset "reverse_importance_weights" begin
        actions = [1, 2]
        π_target = DeterministicPolicy([1,2,1,2], length(actions))
        π_behavior = RandomPolicy([0.5, 0.5])

        states, actions = [1, 2, 3, 4], [2, 2, 2, 2]
        @test collect(reverse_importance_weights(π_target, π_behavior, states, actions)) == [1., 2., 0., 0., 0.]
    end
end

@testset "iterators" begin
    @testset "reductions" begin
        @test reductions(+, 1:4) == [1, 3, 6, 10]
        @test collect(reductions(+, 1:4; init=10)) == [10, 11, 13, 16, 20]
        @test collect(reductions(+, [1 3; 2 4]; init=10)) == [10, 11, 13, 16, 20]
    end

    @testset "is_first_visit" begin
        @test is_first_visit([1,2,1,3,1,2,3]) ==  [true, true, false, true, false, false, false]
        @test collect(Iterators.reverse(IsFirstVisit([1,2,1,3,1,2,3]))) == [false, false, false, true, false, true, true]
    end
end

@testset "decays" begin
    @testset "incrementer" begin
        x = incrementer()
        for i in 1:10
            @test x() == i
        end
    end
    @testset "multiplier" begin
        x = multiplier(ratio=0.99)
        for i in 1:10
            @test x() ≈ 0.99 ^ (i - 1)
        end
    end
    @testset "inverse_decay" begin
        x = inverse_decay()
        for i in 1:10
            @test x() == 1 / i
        end
    end
    @testset "cached_inverse_decay" begin
        x = cached_inverse_decay()
        for i in 1:3
            for j in 1:3
                @test x(j) == 1 / i
            end
        end
        @test x(3) == 1 / 4
        @test x(4) == 1
    end
end

@testset "callbacks" begin
    @testset "stop_at_step" begin
        env = SimpleRandomWalkEnv()
        agent = Agent(MonteCarloLearner(TabularQ(length(observationspace(env)), length(actionspace(env))), EpsilonGreedySelector(0.01)), EpisodeSARDBuffer())
        cb = stop_at_step(2, false)

        @test cb(env, agent) == false
        @test cb() == 1
        @test cb(env, agent) == true
        @test cb() == 2
    end

    @testset "stop_at_episode" begin
        env = SimpleRandomWalkEnv()
        agent = Agent(MonteCarloLearner(TabularQ(length(observationspace(env)), length(actionspace(env))), EpsilonGreedySelector(0.01)), EpisodeSARDBuffer())
        cb = stop_at_episode(2, false)

        reset!(env)
        @test isend(env) == false
        @test cb(env, agent) == false

        env.state = env.N
        @test isend(env) == true
        @test cb(env, agent) == false  # the first episode

        reset!(env)
        @test isend(env) == false
        @test cb(env, agent) == false

        env.state = env.N
        @test isend(env) == true
        @test cb(env, agent) == true  # the second episode
    end
    
    @testset "stop_when_done" begin
        env = SimpleRandomWalkEnv()
        agent = Agent(MonteCarloLearner(TabularQ(length(observationspace(env)), length(actionspace(env))), EpsilonGreedySelector(0.01)), EpisodeSARDBuffer())
        cb = stop_when_done

        reset!(env)
        @test isend(env) == false
        @test cb(env, agent) == false

        env.state = env.N
        @test isend(env) == true
        @test cb(env, agent) == true
    end

    @testset "steps_per_episode" begin
        env = SimpleRandomWalkEnv()
        agent = Agent(MonteCarloLearner(TabularQ(length(observationspace(env)), length(actionspace(env))), EpsilonGreedySelector(0.01)), EpisodeSARDBuffer())
        cb = steps_per_episode()
        steps = [2,3,4]
        
        for n in steps
            reset!(env)
            for i in 1:(n-1)
                cb(env, agent)
            end
            env.state = env.N
            cb(env, agent)
        end
        @test cb() == steps
    end

    @testset "rewards_of_each_episode" begin
        env = SimpleRandomWalkEnv()
        agent = Agent(MonteCarloLearner(TabularQ(length(observationspace(env)), length(actionspace(env))), EpsilonGreedySelector(0.01)), EpisodeSARDBuffer())
        cb = rewards_of_each_episode()
        rewards = [[1., 2., 3.], [4., 5.], [6.]]

        for reward in rewards
            reset!(env)
            @test isend(env) == false
            for r in reward[1:end-1]
                push!(buffer(agent), 1, 1, r, false, 1, 1)
                cb(env, agent)
            end
            env.state = env.N
            @test isend(env) == true
            push!(buffer(agent), 1, 1, reward[end], true, 1, 1)
            cb(env, agent)
        end
        @test cb() == map(sum, rewards)
    end
end

end