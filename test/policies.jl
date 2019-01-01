@testset "policies" begin
    @testset "DeterministicPolicy" begin
        p = DeterministicPolicy([1,2,1], 2)
        @test p(1) == 1
        @test p(2) == 2
        @test p(3) == 1

        @test p(1, Val(:dist)) == [1.0, 0.0]
        @test p(2, Val(:dist)) == [0.0, 1.0]
        @test p(3, Val(:dist)) == [1.0, 0.0]

        @test p(3, 1) == 1.0
        @test p(3, 2) == 0.0

        update!(p, 3, 2)

        @test p(3) == 2
        @test p(3, Val(:dist)) == [0.0, 1.0]
        @test p(3, 1) == 0.0
        @test p(3, 2) == 1.0
    end

    @testset "EpsilonGreedyPolicy" begin
        ϵ, nactions, best_actions = 0.1, 2, [1, 2, 1]
        p = EpsilonGreedyPolicy(best_actions, nactions, ϵ)
        seed!(123)
        for (s, a) in enumerate(best_actions)
            @test isapprox(sum((i) -> p(s) == a ? 1 : 0, 1:1000) / 1000, 1 - ϵ + ϵ / nactions, atol=0.05)
        end

        @test p(1, Val(:dist)) ≈ [0.95, 0.05]
        @test p(2, Val(:dist)) ≈ [0.05, 0.95]
        @test p(3, Val(:dist)) ≈ [0.95, 0.05]

        @test p(1, 1) ≈ 0.95
        @test p(1, 2) ≈ 0.05

        update!(p, 1, 2)
        @test p(1, Val(:dist)) ≈ [0.05, 0.95]
        @test p(1, 1) ≈ 0.05
        @test p(1, 2) ≈ 0.95
    end

    @testset "FunctionalPolicy" begin
        p = FunctionalPolicy(s -> 1)
        @test p(nothing) == 1
    end

    @testset "LinearPolicy" begin
        s1 = [1. 0. 1.; 0. 1. 0.]
        s2 = [0. 1. 1.; 1. 1. 1.]
        features = zeros(2, 2, 3)
        features[1, :, :] .= s1
        features[2, :, :] .= s2
        weights = [1., 1., 1.]
        p = LinearPolicy(features, weights)

        seed!(123)
        action_counts = countmap(p(1) for _ in 1:1000)
        prob = [ℯ^2/(ℯ^2+ℯ), ℯ/(ℯ^2+ℯ)]

        @test isapprox([action_counts[1]/1000, action_counts[2]/1000], prob, atol=0.1)
        @test isapprox(p(1, 1), prob[1]) 
        @test isapprox(p(1, Val(:dist)), prob) 

        update!(p, -1.)
        @test isapprox(p(1, Val(:dist)), [0.5, 0.5]) 
    end

    @testset "RandomPolicy" begin
        prob = [0.3, 0.7]
        p = RandomPolicy(prob)

        seed!(123)
        action_counts = countmap(p(1) for _ in 1:1000) 
        @test isapprox([action_counts[1]/1000, action_counts[2]/1000], prob, atol=0.1)
        @test p(1, Val(:dist)) == prob
        @test p(1, 1) == prob[1]
        @test p(1, 2) == prob[2]

        prob = [0.3  0.7; 0.7 0.3]
        p = RandomPolicy(prob)

        seed!(123)
        action_counts = countmap(p(1) for _ in 1:1000) 
        @test isapprox([action_counts[1]/1000, action_counts[2]/1000], prob[1, :], atol=0.1)
        action_counts = countmap(p(2) for _ in 1:1000) 
        @test isapprox([action_counts[1]/1000, action_counts[2]/1000], prob[2, :], atol=0.1)

        @test p(1, Val(:dist)) == prob[1, :]
        @test p(2, Val(:dist)) == prob[2, :]
        @test p(1, 1) == prob[1, 1]
        @test p(1, 2) == prob[1, 2]
        @test p(2, 1) == prob[2, 1]
        @test p(2, 2) == prob[2, 2]
    end
end