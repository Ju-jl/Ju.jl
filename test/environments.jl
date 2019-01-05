@testset "environments" begin

    function general_environment_test(env, steps=10^2)
        reset!(env)
        @test observe(env).isdone == false
        while steps > 0
            if observe(env).isdone
                reset!(env)
                @test observe(env).isdone == false
            end
            env(sample(actionspace(env)))
            @test observe(env).observation ∈ observationspace(env)
            steps -= 1
        end
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