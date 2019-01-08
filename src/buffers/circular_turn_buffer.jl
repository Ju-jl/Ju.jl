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

capacity(b::CircularTurnBuffer) = min(capacity(x) for x in buffers(b))

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

function push!(b::CircularSARDBuffer{Tuple{Ts, Ta, Float64, Bool}}, s::Ts, a::Ta, r::Float64, d::Bool, ns::Ts, na::Ta) where {Ts, Ta}
    if isempty(b)
        push!(b.state, s)
        push!(b.action, a)
    end
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.state, ns)
    push!(b.action, na)
end

function push!(b::CircularSARDBuffer{Tuple{Ts, Ta, Float64, Bool}}, s::Ts, a::Ta) where {Ts, Ta}
    push!(b.state, s)
    push!(b.action, a)
end

function push!(b::CircularSARDBuffer{Tuple{Ts, Ta, Float64, Bool}}, r::Float64, d::Bool, ns::Ts, na::Ta) where {Ts, Ta}
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.state, ns)
    push!(b.action, na)
end

length(b::CircularSARDBuffer) = length(b.isdone)
capacity(b::CircularSARDBuffer) = capacity(b.isdone)
isfull(b::CircularSARDBuffer) = isfull(b.isdone)

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