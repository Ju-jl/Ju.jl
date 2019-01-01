@testset "environments" begin
    N = 7
    start = 3
    actions = [-1, 1]
    env = SimpleRandomWalkEnv(N=N, actions=actions, start=start)

    @test reset!(env) == (observation=start, isdone=false)
    @test observe(env) == (observation=start, isdone=false)
    @test observationspace(env) == DiscreteSpace(N)
    @test actionspace(env) == DiscreteSpace(length(actions))

    while observe(env).isdone == false
        env(rand(1:length(actions)))
    end
    @test observe(env).observation in [1, N]
end