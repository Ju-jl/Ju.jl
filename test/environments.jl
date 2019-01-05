@testset "environments" begin

    function general_environment_test(env, max_steps=100)
        @test reset!(env) == observe(env)

        while observe(env).isdone == false && max_steps > 0
            env(sample(actionspace(env)))
            max_steps -= 1
        end

        @test observe(env).observation ∈ observationspace(env)
    end

    @testset "SimpleRandomWalkEnv" begin
        env = SimpleRandomWalkEnv(N=7, actions=[-1, 1], start=3)
        general_environment_test(env)
    end

    @testset "CartPoleEnv" begin
        env = CartPoleEnv()
        general_environment_test(env)
    end
end