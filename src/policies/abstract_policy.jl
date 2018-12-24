export AbstractPolicy,
       update!

"""
    AbstractPolicy

Supertype for policies. A policy takes in a state and returns an action.
"""
abstract type AbstractPolicy end

function update! end