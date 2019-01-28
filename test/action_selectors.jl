@testset "action selectors" begin
    @testset "AlternateSelector" begin
        s = AlternateSelector(3)
        values = zeros(3)
        for _ in 1:3
            @test s(values) == 1
            @test s(values) == 2
            @test s(values) == 3
        end
    end
    
    @testset "EpsilonGreedySelector" begin
        @testset "test distribution" begin
            seed!(123)
            ϵ = 0.1
            s = EpsilonGreedySelector(ϵ)
            values = [1, 2, 0, -1]
            action_counts = countmap(s(values) for _ in 1:1000)
            @test isapprox(action_counts[1] / 1000, ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[2] / 1000, 1 - ϵ + ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[3] / 1000, ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[4] / 1000, ϵ / length(values), atol=0.01)
        end

        # https://github.com/Ju-jl/ReinforcementLearningAnIntroduction.jl/issues/6
        @testset "test generator as input" begin
            seed!(123)
            ϵ = 0.1
            s = EpsilonGreedySelector(ϵ)
            values = (i for i in (1,2,0,-1))
            action_counts = countmap(s(values) for _ in 1:1000)
            @test isapprox(action_counts[1] / 1000, ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[2] / 1000, 1 - ϵ + ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[3] / 1000, ϵ / length(values), atol=0.01)
            @test isapprox(action_counts[4] / 1000, ϵ / length(values), atol=0.01)
        end
    end

    @testset "UpperConfidneceBound" begin
        s = UpperConfidenceBound(2)
        values = [1, 2]
        @test s(values) == 2
        @test s(values) == 1
        @test s(values) == 2
        @test s(values) == 2
        @test s(values) == 1
    end

    @testset "WeightedSample" begin
        seed!(123)
        s = WeightedSample()
        values = [0.2, 0.3, 0.5]
        action_counts = countmap(s(values) for _ in 1:1000)
        for i in 1:length(values)
            @test isapprox(action_counts[i]/1000, values[i], atol=0.1)
        end
    end
end