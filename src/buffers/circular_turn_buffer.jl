"""
    CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    CircularTurnBuffer{names, types}(capacities::NTuple{N, Int}, sizes::NTuple{N, NTuple{M, Int} where M}) where {names, types, N}

Using `CircularArrayBuffer` to store each element specified in `names` and `types`.. The memory of the buffer will be pre-allocated.
In RL problems, three of the most common `CircularArrayBuffer` based buffers are: [`CircularSARDBuffer`](@ref), [`CircularSARDSBuffer`](@ref), [`CircularSARDSABuffer`](@ref)
"""
struct CircularTurnBuffer{names, types, Tbs, Tm} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    meta::Tm
    function CircularTurnBuffer{names, types}(
        capacities::NTuple{N, Int},
        sizes::NTuple{N, NTuple{M, Int} where M};
        args...) where {names, types, N}
        buffers = merge(NamedTuple(),
                        (names[i], CircularArrayBuffer{types.parameters[i]}(capacities[i], sizes[i]...)) for i in 1:N)
        meta = args.data
        new{names, types, typeof(buffers), typeof(meta)}(buffers, meta)
    end
end

length(b::CircularTurnBuffer) = length(b.isdone)
capacity(b::CircularTurnBuffer) = capacity(b.isdone)
isfull(b::CircularTurnBuffer) = isfull(b.isdone)


##############################
# CircularSARDBuffer
##############################

"`CircularSARDBuffer` is just an alias for `CircularTurnBuffer{(:state, :action, :reward, :isdone)}`."
const CircularSARDBuffer = CircularTurnBuffer{SARD}

"""
    CircularSARDBuffer(capacity; state_type::Type=Int, action_type::Type=Int, state_size=(), action_size=())

`capacity` specifies how many latest turns the buffer will store at most. 

!!! note
    Note that, the length of state and action is 1 step longer in oder to store the state and action in the next step. This is the supposed behavior of **SARD** buffers.

See also: [`sample`](@ref)
"""
function CircularSARDBuffer(
    capacity;
    state_type::Type=Int,
    action_type::Type=Int,
    state_size=(),
    action_size=())
    CircularSARDBuffer{Tuple{state_type, action_type, Float64, Bool}}(
        (capacity+1, capacity+1, capacity, capacity),
        (state_size, action_size, (), ()))
end

getindex(b::CircularSARDBuffer, i::Int) = (state=b.state[i], action=b.action[i], reward=b.reward[i], isdone=b.isdone[i], nextstate=b.state[i+1], nextaction=b.action[i+1])
function eltype(b::CircularSARDBuffer{names, types}) where {names, types}
    ts, ta, tr, td = types.parameters[2].parameters
    NamedTuple{SARDSA, Tuple{ts, ta, tr, td, ts, ta}}
end

function push!(b::CircularSARDBuffer, s, a)
    if isempty(b)
        push!(b.state, s)
        push!(b.action, a)
    else
        b.state[end] = s
        b.action[end] = a
    end
end

function push!(b::CircularSARDBuffer, r, d, ns, na)
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.state, ns)
    push!(b.action, na)
end

"""
    sample(b::CircularSARDBuffer, batch_size::Int)

Sample a random batch of **S**tates, **A**ctions, **R**ewards, is**D**one,
next**S**tates, next**A**ctions without replacement of `batch_size`.
"""
function sample(b::CircularSARDBuffer, batch_size::Int)
    inds = rand(1:length(b), batch_size)
    batch_view(b, inds), inds
end

function batch_view(b::CircularSARDBuffer, inds)
    view(b.state, inds),
    view(b.action, inds),
    view(b.reward, inds),
    view(b.isdone, inds),
    view(b.state, inds .+ 1)
end

##############################
# CircularSARDSBuffer
##############################
"`CircularSARDSBuffer` is just an alias for `CircularTurnBuffer{(:state, :action, :reward, :isdone, :nextstate)}`."
const CircularSARDSBuffer = CircularTurnBuffer{SARDS}

function CircularSARDSBuffer(
    capacity;
    state_type::Type=Int,
    action_type::Type=Int,
    state_size=(),
    action_size=())
    CircularSARDSBuffer{Tuple{state_type, action_type, Float64, Bool, state_type}}(
        Tuple(capacity for i in 1:5),
        (state_size, action_size, (), (), state_size))
end

function push!(b::CircularSARDSBuffer{Tuple{Ts, Ta, Float64, Bool, Ts}}, s::Ts, a::Ta, r::Float64, d::Bool, ns::Ts) where {Ts, Ta}
    push!(b.state, s)
    push!(b.action, a)
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.nextstate, ns)
end

##############################
# CircularSARDSABuffer
##############################

"`CircularSARDSABuffer` is just an alias for `CircularTurnBuffer{(:state, :action, :reward, :isdone, :nextstate, :nextaction)}`."
const CircularSARDSABuffer = CircularTurnBuffer{SARDSA}

function CircularSARDSABuffer(
    capacity;
    state_type::Type=Int,
    action_type::Type=Int,
    state_size=(),
    action_size=())
    CircularSARDSABuffer{Tuple{state_type, action_type, Float64, Bool, state_type, action_type}}(
        Tuple(capacity for i in 1:6),
        (state_size, action_size, (), (), state_size, action_size))
end

function push!(b::CircularSARDSABuffer{Tuple{Ts, Ta, Float64, Bool, Ts, Ta}}, s::Ts, a::Ta, r::Float64, d::Bool, ns::Ts, na::Ta) where {Ts, Ta}
    push!(b.state, s)
    push!(b.action, a)
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.nextstate, ns)
    push!(b.nextaction, na)
end