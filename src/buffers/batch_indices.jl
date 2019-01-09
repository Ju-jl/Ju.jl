struct BatchIndices{names, T}
    indices::T
    function BatchIndices(batch_size::Int, names=(:state, :action, :reward, :isdone, :nextstate, :nextaction))
        indices = merge(NamedTuple(),
                        (name, Vector{Int}(undef, batch_size)) for name in names)
        new{names, typeof(indices)}(indices)
    end
end