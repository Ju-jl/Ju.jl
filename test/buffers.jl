@testset "buffer" begin

@testset "SumTree" begin
    t = SumTree(3)
    @test capacity(t) == 3
    @test length(t) == 0

    push!(t, 1)
    @test capacity(t) == 3
    @test length(t) == 1
    @test size(t) == (1,)
    @test t[1] == 1

    t[1] = 2
    @test capacity(t) == 3
    @test length(t) == 1
    @test t[1] == 2

    push!(t, 3)
    @test capacity(t) == 3
    @test length(t) == 2
    @test t[1:end] == [2, 3]
    @test t.tree == [5., 5., 0., 2, 3, 0]
    @test Ju.get(t, 0) == (1, 2.)
    @test Ju.get(t, 1) == (1, 2.)
    @test Ju.get(t, 2) == (1, 2.)
    @test Ju.get(t, 3) == (2, 3.)
    @test Ju.get(t, 4) == (2, 3.)
    @test Ju.get(t, 5) == (2, 3.)

    t[1] = 1
    t[2] = 2
    @test capacity(t) == 3
    @test length(t) == 2
    @test t[1:end] == [1, 2]
    @test t.tree == [3., 3., 0., 1, 2, 0]
    @test Ju.get(t, 0) == (1, 1.)
    @test Ju.get(t, 1) == (1, 1.)
    @test Ju.get(t, 2) == (2, 2.)
    @test Ju.get(t, 3) == (2, 2.)

    push!(t, 3)
    @test capacity(t) == 3
    @test length(t) == 3
    @test t[1:end] == [1, 2, 3]
    @test t.tree == [6., 3., 3., 1., 2., 3.]
    @test Ju.get(t, 0) == (1, 1.)
    @test Ju.get(t, 1) == (1, 1.)
    @test Ju.get(t, 2) == (2, 2.)
    @test Ju.get(t, 3) == (2, 2.)
    @test Ju.get(t, 4) == (3, 3.)
    @test Ju.get(t, 5) == (3, 3.)
    @test Ju.get(t, 6) == (3, 3.)

    push!(t, 4)
    @test capacity(t) == 3
    @test length(t) == 3
    @test t[1:end] == [2, 3, 4]
    @test t.tree == [9., 6., 3., 4, 2, 3]
    @test Ju.get(t, 0) == (3, 4.)
    @test Ju.get(t, 1) == (3, 4.)
    @test Ju.get(t, 2) == (3, 4.)
    @test Ju.get(t, 3) == (3, 4.)
    @test Ju.get(t, 4) == (3, 4.)
    @test Ju.get(t, 5) == (1, 2.)
    @test Ju.get(t, 6) == (1, 2.)
    @test Ju.get(t, 9) == (2, 3.)
    @test Ju.get(t, 9) == (2, 3.)
    @test Ju.get(t, 9) == (2, 3.)

    empty!(t)
    @test capacity(t) == 3
    @test length(t) == 0

    push!(t, 1)
    push!(t, 2)
    push!(t, 3)
    @test capacity(t) == 3
    @test length(t) == 3
    @test t[1:end] == [1, 2, 3]
    @test t.first == 2
    @test t.tree == [6., 4., 2., 3., 1., 2.]
    @test Ju.get(t, 0) == (3, 3.)
    @test Ju.get(t, 1) == (3, 3.)
    @test Ju.get(t, 2) == (3, 3.)
    @test Ju.get(t, 3) == (3, 3.)
    @test Ju.get(t, 4) == (1, 1.)
    @test Ju.get(t, 5) == (2, 2.)
    @test Ju.get(t, 6) == (2, 2.)
end

@testset "CircularArrayBuffer" begin
    A = ones(2, 2)
    @testset "1D Int" begin
        b = CircularArrayBuffer{Int}(3)

        @test eltype(b) == Int
        @test capacity(b) == 3
        @test isfull(b) == false
        @test isempty(b) == true
        @test length(b) == 0
        @test size(b) == (0,)
        # element must has the exact same length with the element of buffer
        @test_throws DimensionMismatch push!(b, [1, 2])  

        for x in 1:3 push!(b, x) end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 1
        @test b[end] == 3
        @test b[1:end] == [1, 2, 3]

        for x in 4:5 push!(b, x) end

        @test capacity(b) == 3
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[1] == 3
        @test b[end] == 5
        @test b[1:end] == [3, 4, 5]

        empty!(b)
        @test isfull(b) == false
        @test isempty(b) == true
        @test length(b) == 0
        @test size(b) == (0,)

        push!(b, 6)
        @test isfull(b) == false
        @test isempty(b) == false
        @test length(b) == 1
        @test size(b) == (1,)
        @test b[1] == 6

        push!(b, 7)
        push!(b, 8)
        @test isfull(b) == true
        @test isempty(b) == false
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[[1,2,3]] == [6, 7, 8]

        push!(b, 9)
        @test isfull(b) == true
        @test isempty(b) == false
        @test length(b) == 3
        @test size(b) == (3,)
        @test b[[1,2,3]] == [7, 8, 9]
    end

    @testset "2D Float64" begin
        b = CircularArrayBuffer{Array{Float64,2}}(3, (2, 2))

        @test eltype(b) == Array{Float64, 2}
        @test capacity(b) == 3
        @test isfull(b) == false
        @test length(b) == 0
        @test size(b) == (2, 2, 0)
        # element must has the exact same length with the element of buffer
        @test_throws DimensionMismatch push!(b, [1., 2.])  

        for x in 1:3 push!(b, x * A) end

        @test capacity(b) == 3
        @test isfull(b) == true
        @test length(b) == 3
        @test size(b) == (2, 2, 3)
        for i in 1:3 @test b[i] == i * A end
        @test b[end] == 3 * A

        for x in 4:5 push!(b, x * ones(4)) end  # collection is also OK

        @test capacity(b) == 3
        @test length(b) == 3
        @test size(b) == (2, 2, 3)
        @test b[1] == 3 * A
        @test b[end] == 5 * A
        
        @test b == reshape([c for x in 3:5 for c in x*A], 2, 2, 3)
    end
end

@testset "CircularTurnBuffer" begin
    @testset "CircularSARDBuffer" begin
        b = CircularSARDBuffer(3)
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == 3

        push!(b, 1, 1)
        push!(b, 1., false, 2, 2)
        @test length(b) == 1
        @test size(b) == (1,)
        @test isfull(b) == false
        @test isempty(b) == false
        
        push!(b, 2., false, 3, 3)
        push!(b, 3., false, 4, 4)
        @test length(b) == 3
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[1] == (state=1, action=1, reward=1., isdone=false, nextstate=2, nextaction=2)
        @test b[end] == (state=3, action=3, reward=3., isdone=false, nextstate=4, nextaction=4)

        push!(b, 4., false, 5, 5)
        @test length(b) == 3
        @test isfull(b) == true
        @test b[1] == (state=2, action=2, reward=2., isdone=false, nextstate=3, nextaction=3)
        @test b[end] == (state=4, action=4, reward=4., isdone=false, nextstate=5, nextaction=5)
    end
    @testset "CircularSARDBuffer of 2D element type" begin
        b = CircularSARDBuffer(3; state_type=Array{Int,2}, state_size=(2,2))
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == 3

        push!(b, [1 1; 1 1], 1)
        push!(b, 1., false, [2 2; 2 2], 2)
        @test length(b) == 1
        @test isfull(b) == false
        @test isempty(b) == false
        
        push!(b, 2., false, [3 3; 3 3], 3)
        push!(b, 3., false, [4 4; 4 4], 4)
        @test length(b) == 3
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[1] == (state=[1 1; 1 1], action=1, reward=1., isdone=false, nextstate=[2 2; 2 2], nextaction=2)
        @test b[end] == (state=[3 3; 3 3], action=3, reward=3., isdone=false, nextstate=[4 4; 4 4], nextaction=4)

        push!(b, 4., true, [5 5; 5 5], 5)
        @test length(b) == 3
        @test isfull(b) == true
        @test b[1] == (state=[2 2; 2 2], action=2, reward=2., isdone=false, nextstate=[3 3; 3 3], nextaction=3)
        @test b[end] == (state=[4 4; 4 4], action=4, reward=4., isdone=true, nextstate=[5 5; 5 5], nextaction=5)

        # now push a new state-action
        push!(b, [6 6; 6 6], 6)
        @test length(b) == 3
        @test isfull(b) == true
        @test b[1] == (state=[2 2; 2 2], action=2, reward=2., isdone=false, nextstate=[3 3; 3 3], nextaction=3)
        @test b[end] == (state=[4 4; 4 4], action=4, reward=4., isdone=true, nextstate=[6 6; 6 6], nextaction=6)

        batch_inds = [3, 2, 1]
        batch_state = reshape([4 4 4 4 3 3 3 3 2 2 2 2], 2, 2, 3)
        batch_action = [4, 3, 2]
        batch_reward = [4., 3., 2.]
        batch_isdone = [true, false, false]
        batch_nextstate = reshape([6 6 6 6 4 4 4 4 3 3 3 3], 2, 2, 3)
        @test Ju.batch_view(b, batch_inds) == (batch_state, batch_action, batch_reward, batch_isdone, batch_nextstate)
    end
    @testset "CircularSARDSBuffer" begin
        b = CircularSARDSBuffer(3)
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == 3

        push!(b, 1, 1, 1.0, false, 2)
        push!(b, 3, 3, 3.0, false, 4)
        push!(b, 5, 5, 5.0, false, 6)
        @test length(b) == 3
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2)
        @test b[end] == (state=5, action=5, reward=5.0, isdone=false, nextstate=6)

        push!(b, 7, 7, 7.0, false, 8)
        @test b[1] == (state=3, action=3, reward=3.0, isdone=false, nextstate=4)
        @test b[end] == (state=7, action=7, reward=7.0, isdone=false, nextstate=8)
    end
    @testset "CircularSARDSABuffer" begin
        b = CircularSARDSABuffer(3)
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == 3

        push!(b, 1, 1, 1.0, false, 2, 2)
        push!(b, 3, 3, 3.0, false, 4, 4)
        push!(b, 5, 5, 5.0, false, 6, 6)
        @test length(b) == 3
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2, nextaction=2)
        @test b[end] == (state=5, action=5, reward=5.0, isdone=false, nextstate=6, nextaction=6)

        push!(b, 7, 7, 7.0, false, 8, 8)
        @test b[1] == (state=3, action=3, reward=3.0, isdone=false, nextstate=4, nextaction=4)
        @test b[end] == (state=7, action=7, reward=7.0, isdone=false, nextstate=8, nextaction=8)
    end
end

@testset "EpisodeTurnBuffer" begin
    @testset "EpisodeSARDBuffer" begin
        b = EpisodeSARDBuffer()
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true
        @test capacity(b) == typemax(Int)

        push!(b, 1, 1)
        push!(b, 1., false, 2, 2)
        @test length(b) == 1
        @test isfull(b) == false
        @test isempty(b) == false

        # we can also push a turn infor together, 
        # notice that state and action will be iginored
        push!(b, 2., true, 3, 3)  
        @test length(b) == 2
        @test isfull(b) == true
        @test isempty(b) == false
        @test b[end] == (state=2, action=2, reward=2., isdone=true)

        push!(b, 3, 3)
        push!(b, 3., false, 4, 4)
        @test length(b) == 1
        @test isfull(b) == false
        @test isempty(b) == false
        @test b[end] == (state=3, action=3, reward=3., isdone=false)
    end
    @testset "EpisodeSARDSBuffer" begin
        b = EpisodeSARDSBuffer()
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true

        push!(b, 1, 1, 1.0, false, 2)
        push!(b, 3, 3, 3.0, false, 4)
        push!(b, 5, 5, 5.0, false, 6)
        @test length(b) == 3
        @test isfull(b) == false
        @test isempty(b) == false
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2)
        @test b[end] == (state=5, action=5, reward=5.0, isdone=false, nextstate=6)

        push!(b, 7, 7, 7.0, true, 8)
        @test length(b) == 4
        @test isfull(b) == true
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2)
        @test b[end] == (state=7, action=7, reward=7.0, isdone=true, nextstate=8)

        push!(b, 9, 9, 9.0, false, 10)
        @test length(b) == 1
        @test isfull(b) == false
        @test b[1] == b[end] == (state=9, action=9, reward=9.0, isdone=false, nextstate=10)
    end
    @testset "EpisodeSARDSABuffer" begin
        b = EpisodeSARDSABuffer()
        @test length(b) == 0
        @test isfull(b) == false
        @test isempty(b) == true

        push!(b, 1, 1, 1.0, false, 2, 2)
        push!(b, 3, 3, 3.0, false, 4, 4)
        push!(b, 5, 5, 5.0, false, 6, 6)
        @test length(b) == 3
        @test isfull(b) == false
        @test isempty(b) == false
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2, nextaction=2)
        @test b[end] == (state=5, action=5, reward=5.0, isdone=false, nextstate=6, nextaction=6)

        push!(b, 7, 7, 7.0, true, 8, 8)
        @test length(b) == 4
        @test isfull(b) == true
        @test b[1] == (state=1, action=1, reward=1.0, isdone=false, nextstate=2, nextaction=2)
        @test b[end] == (state=7, action=7, reward=7.0, isdone=true, nextstate=8, nextaction=8)

        push!(b, 9, 9, 9.0, false, 10, 10)
        @test length(b) == 1
        @test isfull(b) == false
        @test b[1] == b[end] == (state=9, action=9, reward=9.0, isdone=false, nextstate=10, nextaction=10)
    end
end

end