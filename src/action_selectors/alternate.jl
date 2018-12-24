"""
    AlternateSelector <: AbstractActionSelector

Used to ensure that all actions are selected alternatively.

    AlternateSelector(n::Int)

`n::Int` means the optional actions are `1:n`.
"""
mutable struct AlternateSelector <: AbstractActionSelector
    n::Int
    cur::Int
    AlternateSelector(n::Int) = new(n, 0)
end

function (s::AlternateSelector)()
    s.cur = s.cur + 1 > s.n ? 1 : s.cur + 1
    s.cur
end

"""
    (s::AlternateSelector)(values::Any)

Ignore the action `values`, generate an action alternatively.

## Example

```julia
julia> selector = AlternateSelector(3)
AlternateSelector(3, 0)

julia> any_state = 0 # for AlternateSelector, state can be anything

julia> [selector(any_state) for i in 1:10]  # iterate through all actions
10-element Array{Int64,1}:
 1
 2
 3
 1
 2
 3
 1
 2
 3
 1
```
"""
(s::AlternateSelector)(values::Any) = s()