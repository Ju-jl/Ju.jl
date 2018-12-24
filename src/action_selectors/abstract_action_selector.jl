import DataStructures:update!
export AbstractActionSelector

"""
    AbstractActionSelector

A subtype of `AbstractActionSelector` is used to generate an action
given the estimated valud of different actions.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `selector(values)` | `selector`, an instance of `AbstractActionSelector`, must be a callable object which takes in an estimation and returns an action |
"""
abstract type AbstractActionSelector end

function update!(as::AbstractActionSelector, s, a) end