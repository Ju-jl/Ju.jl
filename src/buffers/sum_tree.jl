"""
    SumTree(capacity::Int)

Efficiently sample and update weights.

Reference:

- https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py#L18-L86

# Example
```julia
```
"""
mutable struct SumTree
    capacity::Int
    first::Int
    tree::Vector{Float64}
    SumTree(capacity) = new(capacity, 1, zeros(capacity*2-1))
end

function push!(t::SumTree, w)
    update!(t::SumTree, t.first, w)
    t.first += 1
    if t.first > t.capacity
        t.first = 1
    end
end

function update!(t::SumTree, i, w)
    tree_ind = t.capacity - 1 + i
    change = w - t.tree[tree_ind]
    t.tree[tree_ind] = w
    while tree_ind != 1
        tree_ind = tree_ind ÷ 2
        t.tree[tree_ind] += change
    end
end

function sample(t::SumTree)
end

function sample(t::SumTree, n)
end