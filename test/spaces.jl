@testset "Space" begin

@testset "ContinuousSpace" begin
    s = ContinuousSpace(0., 1.)
    @test in(0.5, s) == true
    @test in(0.0, s) == true
    @test in(1.0, s) == true
    @test in(-1.0, s) == false
    @test in(-Inf, s) == false

    @test in(sample(s), s)
    @test eltype(s) == Float64
end

@testset "MultiContinuousSpace" begin
    s = MultiContinuousSpace([-1., -2.], [1., 2.])
    @test in([0, 0], s) == true
    @test in([0, 3], s) == false
    @test in(sample(s), s)
    @test eltype(s) == Array{Float64, 1}
end

@testset "DiscreteSpace" begin
    s = DiscreteSpace(10)

    @test in(0, s) == false
    @test in(1, s) == true
    @test in(5, s) == true
    @test in(10, s) == true

    @test in(sample(s), s)
    @test size(s) == (10,)
    @test length(s) == 10
end

@testset "MultiDiscreteSpace" begin
    s = MultiDiscreteSpace([2,3,2])

    @test in([0,0,0], s) == false
    @test in([1,1,1], s) == true
    @test in([3,3,3], s) == false

    @test size(s) == (2,3,2)
    @test length(s) == 2*3*2
    @test in(sample(s), s)
end

@testset "Space Tuple" begin
    @test in(([0.5], 5, [1, 1]),
             (MultiContinuousSpace(0,1, (1,)), DiscreteSpace(5), MultiDiscreteSpace([2,2]))) == true
    @test in(([0.5], 1, [1, 1]),
              (MultiContinuousSpace(0,1, (1,)), DiscreteSpace(5), MultiDiscreteSpace([2,2]))) == true
    @test in((), 
             (MultiContinuousSpace(0,1, (1,)), DiscreteSpace(5), MultiDiscreteSpace([2,2]))) == false
end

@testset "Space Dict" begin
    @test in(
        Dict(
            "sensors" => Dict(
                "position" => [-10, 0, 10],
                "velocity" => [0.1, 0.2, 0.3],
                "front_cam" => (rand(10, 10, 3), rand(10, 10, 3)),
                "rear_cam" => rand(10,10,3)),
            "ext_controller" => [2, 1, 1],
            "inner_state" => Dict(
                "charge" => 35,
                "job_status" => Dict(
                    "task" => 3,
                    "progress" => [23]))),
        Dict(
            "sensors" =>  Dict(
                "position"=> MultiContinuousSpace(-100, 100, (3,)),
                "velocity"=> MultiContinuousSpace(-1, 1, (3,)),
                "front_cam"=> (MultiContinuousSpace(0, 1, (10, 10, 3)),
                               MultiContinuousSpace(0, 1, (10, 10, 3))),
                "rear_cam" => MultiContinuousSpace(0, 1, (10, 10, 3))),
            "ext_controller" => MultiDiscreteSpace([5, 2, 2]),
            "inner_state" => Dict(
                "charge" => DiscreteSpace(100),
                "job_status" => Dict(
                    "task" => DiscreteSpace(5),
                    "progress" => MultiContinuousSpace(0, 100, (1,)))))) == true
end

end
