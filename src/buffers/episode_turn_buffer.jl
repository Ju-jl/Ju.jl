"""
    EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    EpisodeTurnBuffer{names, types}() where {names, types}

Using a Vector to store each element specified by `names` and `types`.

See also: [`EpisodeSARDBuffer`](@ref), [`EpisodeSARDSBuffer`](@ref), [`EpisodeSARDSABuffer`](@ref)
"""
struct EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}
    buffers::Tbs
    function EpisodeTurnBuffer{names, types}() where {names, types}
        length(names) != length(types.parameters) && throw(DimensionMismatch("length of names and types must match!"))
        buffers = merge(NamedTuple(),
                        (names[i], Vector{types.parameters[i]}()) for i in 1:length(names))
        new{names, types, typeof(buffers)}(buffers)
    end
end

capacity(b::EpisodeTurnBuffer) = isfull(b) ? length(b) : typemax(Int)

##############################
# EpisodeSARDBuffer
##############################
"`EpisodeSARDBuffer` is just an alias for `EpisodeTurnBuffer{(:state, :action, :reward, :isdone)}`"
const EpisodeSARDBuffer = EpisodeTurnBuffer{SARD}

EpisodeSARDBuffer(;state_type::Type=Int, action_type::Type=Int) = EpisodeSARDBuffer{Tuple{state_type, action_type, Float64, Bool}}()

"only valid when buffer is empty"
function push!(b::EpisodeSARDBuffer{Tuple{Ts, Ta, Float64, Bool}}, s::Ts, a::Ta) where {Ts, Ta}
    !isempty(b) && empty!(b)
    push!(b.state, s)
    push!(b.action, a)
end

function push!(b::EpisodeSARDBuffer{Tuple{Ts, Ta, Float64, Bool}}, r::Float64, d::Bool, ns::Ts, na::Ta) where {Ts, Ta}
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.state, ns)
    push!(b.action, na)
end

length(b::EpisodeSARDBuffer) = length(b.isdone)
isfull(b::EpisodeSARDBuffer) = length(b.isdone) > 0 && b.isdone[end]

##############################
# EpisodeSARDSBuffer
##############################
"`EpisodeSARDSBuffer` is just an alias for `EpisodeTurnBuffer{(:state, :action, :reward, :isdone, :nextstate)}`"
const EpisodeSARDSBuffer = EpisodeTurnBuffer{SARDS}

EpisodeSARDSBuffer(;state_type::Type=Int, action_type::Type=Int) = EpisodeSARDSBuffer{Tuple{state_type, action_type, Float64, Bool, state_type}}()

function push!(b::EpisodeSARDSBuffer{Tuple{Ts, Ta, Float64, Bool, Ts}}, s::Ts, a::Ta, r::Float64, d::Bool, ns::Ts) where {Ts, Ta}
    isfull(b) && empty!(b)
    push!(b.state, s)
    push!(b.action, a)
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.nextstate, ns)
end

length(b::EpisodeSARDSBuffer) = length(b.isdone)
isfull(b::EpisodeSARDSBuffer) = length(b.isdone) > 0 && b.isdone[end]

##############################
# EpisodeSARDSABuffer
##############################
"`EpisodeSARDSABuffer` is just an alias for `EpisodeTurnBuffer{(:state, :action, :reward, :isdone, :nextstate, :nextaction)}`"
const EpisodeSARDSABuffer = EpisodeTurnBuffer{SARDSA}

EpisodeSARDSABuffer(;state_type::Type=Int, action_type::Type=Int) = EpisodeSARDSABuffer{Tuple{state_type, action_type, Float64, Bool, state_type, action_type}}()

function push!(b::EpisodeSARDSABuffer{Tuple{Ts, Ta, Float64, Bool, Ts, Ta}}, s::Ts, a::Ta, r::Float64, d::Bool, ns::Ts, na::Ta) where {Ts, Ta}
    isfull(b) && empty!(b)
    push!(b.state, s)
    push!(b.action, a)
    push!(b.reward, r)
    push!(b.isdone, d)
    push!(b.nextstate, ns)
    push!(b.nextaction, na)
end

length(b::EpisodeSARDSABuffer) = length(b.isdone)
isfull(b::EpisodeSARDSABuffer) = length(b.isdone) > 0 && b.isdone[end]