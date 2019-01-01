@testset "approximators" begin
    @testset "TabularQ" begin
        A = TabularQ([1 2; 3 4])
        @test A(1) == [1., 2.]
        @test A(2) == [3., 4.]
        @test A(2, 2) == 4
        @test A(2,Val(:dist)) == [3., 4.]
        @test A(2,Val(:max)) == 4.
        @test A(2,Val(:argmax)) == 2

        update!(A, 2, 2, -1.)
        @test A(2) == [3., 3.]
        @test A(2, 2) == 3

        update!(A, 1, [-1., -1.])
        @test A(1) == [0., 1.]
        @test A(1, 1) == 0
        @test A(1, 2) == 1
    end

    @testset "TabularV" begin
        V = TabularV([1., 2., 3., 2., 1.])
        @test V(1) == 1.
        @test V(3) == 3.

        update!(V, 3, -1.)
        @test V(3) == 2.
    end

    @testset "AggregationV" begin
        table = [-1., 1.]
        V = AggregationV(table, s -> s < 0 ? 1 : 2)

        @test V(-5) == -1.
        @test V(5) == 1.

        update!(V, -1, 0.5)
        update!(V, 1, -0.5)

        @test V(-5) == -.5
        @test V(5) == 0.5
    end

    @testset "FourierV" begin
        V = FourierV([1, 1, 1])
        @test V(1) ≈ 1.0
        @test V(2) ≈ 3.0
        
        update!(V, 1, 0.5)
        @test V(1) ≈ 2.5
        @test V(2) ≈ 3.5
    end

    @testset "LinearV" begin
        features = [1 0 1; 0 1 1]
        weights = [1., 2., 3.]
        V = LinearV(features, weights)

        @test V(1) ≈ 1. + 3.
        @test V(2) ≈ 2. + 3.

        update!(V, 1, 0.5)
        @test V(1) ≈ 1.5 + 3.5
        @test V(2) ≈ 2. + 3.5
    end

    @testset "PolynomialV" begin
        V = PolynomialV([1., 2., 3.])

        @test V(2) ≈ 1 + 2. * 2  + 3. * 2^2

        update!(V, 2, 1.)
        @test V(2) ≈ 2 + 4. * 2 + 7. * 2^2
    end

    @testset "tilings" begin
        init_tiling = Tiling((0:5:50, 0:10:100))
        tilings = [init_tiling, init_tiling - [1, 2]]
        @testset "TilingsV" begin
            V = TilingsV(tilings)
            update!(V, [0, 0], 1.)
            @test V([0,0]) == 2.
            @test V([4,8]) == 1.
            @test V([5,10]) == 0.
        end
        @testset "TilingsQ" begin
            Q = TilingsQ(tilings, 2)
            update!(Q, [0, 0], 1, 1.)

            @test Q([0, 0]) ≈ [2., 0.]
            @test Q([0, 0], 1) ≈ 2.
            @test Q([0, 0], Val(:max)) ≈ 2.
            @test Q([0, 0], Val(:argmax)) == 1
        end
    end
end
