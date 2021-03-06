export AbstractTurnBuffer, SARD, SARDSA, SARDBuffer, SARDSABuffer,
       isfull, capacity, buffers, empty!

import Base: size, getindex, setindex!, length, push!, eltype, view,
             empty!, isempty, getproperty, lastindex

import DataStructures:isfull, capacity

"""
    AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1}

`AbstractTurnBuffer` is supertype of a collection of buffers to store the interactions between agents and environments.
It is a subtype of `AbstractArray{NamedTuple{names, types}, 1}` where `names` specifies which fields are to store
and `types` is the coresponding types of the `names`.


| Required Methods| Brief Description |
|:----------------|:------------------|
| `Base.push!(b::AbstractTurnBuffer{names, types}, s[, a, r, d, s′, a′])` | Push a turn info into the buffer. According to different `names` and `types` of the buffer `b`, it may accept different number of arguments |
| `isfull(b)` | Check whether the buffer is full or not |
| `capacity(b)` | The maximum length of buffer |
| `Base.length(b)` | Return the length of buffer |
| `Base.getindex(b::AbstractTurnBuffer{names, types})` | Return a turn of type `NamedTuple{names, types}` |
| `Base.empty!(b)` | Reset the buffer |
| **Optional Methods** | |
| `Base.size(b)` | Return `(length(b),)` by default |
| `Base.isempty(b)` | Check whether the buffer is empty or not. Return `length(b) == 0` by default |
| `Base.lastindex(b)` | Return `length(b)` by default |
"""
abstract type AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1} end

const SARD = (:state, :action, :reward, :isdone)
const SARDS = (:state, :action, :reward, :isdone, :nextstate)
const SARDSA = (:state, :action, :reward, :isdone, :nextstate, :nextaction)
const SARDBuffer = AbstractTurnBuffer{SARD}
const SARDSBuffer = AbstractTurnBuffer{SARDS}
const SARDSABuffer = AbstractTurnBuffer{SARDSA}

function isfull end
function capacity end
buffers(b::AbstractTurnBuffer) = getfield(b, :buffers)

size(b::AbstractTurnBuffer) = (length(b),)
isempty(b::AbstractTurnBuffer) = length(b) == 0
lastindex(b::AbstractTurnBuffer) = length(b)
getindex(b::AbstractTurnBuffer, i::Int) = eltype(b)(x[i] for x in buffers(b))
empty!(b::AbstractTurnBuffer) = for x in buffers(b) empty!(x) end
getproperty(b::AbstractTurnBuffer, f::Symbol) = getfield(buffers(b), f)