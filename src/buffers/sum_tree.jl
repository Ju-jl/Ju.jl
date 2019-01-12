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
    length::Int
    first::Int
    tree::Vector{Float64}
    SumTree(capacity) = new(capacity, 1, 0, zeros(capacity*2-1))
end

function push!(t::SumTree, p)
    if t.length == t.capacity
        t.first = (t.first == t.capacity ? 1 : t.first + 1)
    else
        t.length += 1
    end
    ind = t.first + t.length - 1
    if ind > t.capacity
        ind -= t.capacity
    end
    tree_ind = t.capacity - 1 + ind
    update!(t::SumTree, tree_ind, p)
end

function update!(t::SumTree, tree_ind, p)
    change = p - t.tree[tree_ind]
    t.tree[tree_ind] = p
    while tree_ind != 1
        tree_ind = tree_ind ÷ 2
        t.tree[tree_ind] += change
    end
end

function indexof(t::SumTree, v)
    parent_ind = 1
    leaf_ind = parent_ind
    while true
        left_child_ind = parent_ind * 2
        right_child_ind = left_child_ind + 1
        if left_child_ind > length(t.tree)
            leaf_ind = parent_ind
            break
        else
            if v ≤ t.tree[left_child_ind]
                parent_ind = left_child_ind
            else
                v -= t.tree[left_child_ind]
                parent_ind = right_child_ind
            end
        end
    end
    ind = leaf_ind - (t.capacity - 1)
    ind >= t.first ? ind - t.first + 1 : ind + t.capacity - t.first + 1
end