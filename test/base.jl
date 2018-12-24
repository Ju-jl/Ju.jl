using SparseArrays

@testset "base" begin

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
end