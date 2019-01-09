using StatsBase:sample!

"""
    CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    CircularTurnBuffer{names, types}(capacities::NTuple{N, Int}, sizes::NTuple{N, NTuple{M, Int} where M}) where {names, types, N}

Using `CircularArrayBuffer` to store each element specified in `names` and `types`.. The memory of the buffer will be pre-allocated.
In RL problems, three of the most common `CircularArrayBuffer` based buffers are: [`CircularSARDBuffer`](@ref), [`CircularSARDSBuffer`](@ref), [`CircularSARDSABuffer`](@ref)
"""
struct CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    function CircularTurnBuffer{names, types}(
        capacities::NTuple{N, Int},
        sizes::NTuple{N, NTuple{M, Int} where M}) where {names, types, N}
        buffers = merge(NamedTuple(),
                        (names[i], CircularArrayBuffer{types.parameters[i]}(capacities[i], sizes[i]...)) for i in 1:N)
        new{names, types, typeof(buffers)}(buffers)
    end
end

capacity(b::CircularTurnBuffer) = min(capacity(x) for x in b.buffers)

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

function push!(b::CircularSARDBuffer, s, a)
    push!(b.buffers.state, s)
    push!(b.buffers.action, a)
end

function push!(b::CircularSARDBuffer, r, d, ns, na)
    push!(b.buffers.reward, r)
    push!(b.buffers.isdone, d)
    push!(b.buffers.state, ns)
    push!(b.buffers.action, na)
end

getindex(b::CircularSARDBuffer, ::Val{:state}, i) = b.buffers.state[i]
getindex(b::CircularSARDBuffer, ::Val{:action}, i) = b.buffers.action[i]
getindex(b::CircularSARDBuffer, ::Val{:reward}, i) = b.buffers.reward[i]
getindex(b::CircularSARDBuffer, ::Val{:isdone}, i) = b.buffers.isdone[i]
getindex(b::CircularSARDBuffer, ::Val{:nextstate}, i) = b.buffers.state[i+1]
getindex(b::CircularSARDBuffer, ::Val{:nextaction}, i) = b.buffers.action[i+1]

length(b::CircularSARDBuffer) = length(b.buffers.isdone)
capacity(b::CircularSARDBuffer) = capacity(b.buffers.isdone)
isfull(b::CircularSARDBuffer) = isfull(b.buffers.isdone)

"""
    sample!(b::CircularSARDBuffer, batch_indices::BatchIndices)

Sample a random batch of **S**tates, **A**ctions, **R**ewards, is**D**one,
next**S**tates, next**A**ctions without replacement.
"""
function batch_sample(b::CircularSARDBuffer, batch_indices::BatchIndices{SARDSA})
    sample!(1:length(b), batch_indices.indices.state)
    batch_indices.indices.action .= batch_indices.indices.state
    batch_indices.indices.reward .= batch_indices.indices.state
    batch_indices.indices.isdone .= batch_indices.indices.state
    map!(i -> i + 1, batch_indices.indices.nextstate, batch_indices.indices.state)
    batch_indices.indices.nextaction .= batch_indices.indices.nextstate

    view(b.buffers.state, batch_indices.indices.state),
    view(b.buffers.action, batch_indices.indices.action),
    view(b.buffers.reward, batch_indices.indices.reward),
    view(b.buffers.isdone, batch_indices.indices.isdone),
    view(b.buffers.state, batch_indices.indices.nextstate),
    view(b.buffers.action, batch_indices.indices.nextaction)
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

length(b::CircularSARDSBuffer) = length(b.isdone)
capacity(b::CircularSARDSBuffer) = capacity(b.isdone)
isfull(b::CircularSARDSBuffer) = isfull(b.isdone)

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

length(b::CircularSARDSABuffer) = length(b.isdone)
capacity(b::CircularSARDSABuffer) = capacity(b.isdone)
isfull(b::CircularSARDSABuffer) = isfull(b.isdone)