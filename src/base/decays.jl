export incrementer, multiplier, inverse_decay, cached_inverse_decay, sample_avg, cached_sample_avg, exp_decay

"""
    incrementer(;start=1, step=1)

# Example
```julia
julia> f = incrementer();

julia> [f() for _ in 1:3]
3-element Array{Int64,1}:
 1
 2
 3
```
"""
function incrementer(;start=1, step=1)
    i = start - step
    () -> i += step
end

"""
    multiplier(;start=1.0, ratio=1.0)

# Example
```julia
julia> f = multiplier(ratio=0.9);

julia> [f() for _ in 1:3]
3-element Array{Float64,1}:
 1.0
 0.9
 0.81
```
"""
function multiplier(;start=1.0, ratio=1.0)
    x = start / ratio
    () -> x *= ratio
end

"""
    inverse_decay()

# Example
```julia
julia> f = inverse_decay()
#25 (generic function with 1 method)

julia> [f() for _ in 1:5]
5-element Array{Float64,1}:
 1.0
 0.5
 0.3333333333333333
 0.25
 0.2
```
"""
function inverse_decay()
    x = 0
    (args...) -> begin
        x += 1
        1/x
    end
end

"""
    sample_avg()

# Example
```julia
julia> f = sample_avg();

julia> f(2)
2.0

julia> f(3) # (2+3)/2
2.5

julia> f(5) # (2+3+5)/3
3.3333333333333335
```
"""
function sample_avg()
    t = 0
    avg = 0
    (x) -> begin
        t += 1
        avg += (x-avg)/t
    end
end

"""
    cached_inverse_decay()

# Example
```julia
julia> f = cached_inverse_decay();

julia> f(:a) # cached!
1.0

julia> f(:a) # call again
0.5

julia> f(:a) # call again
0.3333333333333333

julia> f(:b) # a new cache
1.0
```
"""
function cached_inverse_decay()
    cache = Dict()
    function (x)
        if !haskey(cache, x)
            cache[x] = inverse_decay()
        end
        cache[x]()
    end
end

"""
    cached_sample_avg()

# Example
```julia
julia> f = cached_sample_avg();

julia> f(:a, 3)  # cache :a
3.0

julia> f(:a, 5)  # calculate avg value of  :a (3 + 5) / 2
4.0

julia> f(:a, 8)  # calculate avg value of  :a (3 + 5 + 8) /3
5.333333333333333

julia> f(:b, 0)  # cache another value
0.0
```
"""
function cached_sample_avg()
    cache = Dict()
    function (k, x)
        if !haskey(cache, k)
            cache[k] = sample_avg()
        end
        cache[k](x)
    end
end

"""
    exp_decay(init=1.0, λ=0.1, decay_step=1000, clip=1e-4)

See [Exponential Decay](https://en.wikipedia.org/wiki/Exponential_decay)
"""
function exp_decay(;init=1.0, λ=0.1, decay_step=1000, clip=1e-4)
    i = -1
    function f()
        i += 1
        max(init * exp(- λ * i / decay_step), clip)
    end
end