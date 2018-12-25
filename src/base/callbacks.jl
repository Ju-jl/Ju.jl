using ProgressMeter:Progress, next!

export stop_at_step, stop_at_episode, stop_when_done, steps_per_episode, rewards_of_each_episode

"""
    stop_at_step(n::Int, is_show_progress::Bool=true)

Return a function, which will return false after been called `n` times.
`is_show_progress` will control whether print the progress meter or not.
"""
function stop_at_step(n::Int, is_show_progress::Bool=true)
    i, p = 0, Progress(n)
    function f(env, agent)
        i += 1
        is_show_progress && next!(p; showvalues = [(:step, i)])
        i ≥ n
    end
end

"""
    stop_at_episode(n::Int, is_show_progress::Bool=true)

Return a function, which will return false after `n` episodes.
`is_show_progress` will control whether print the progress meter or not.
"""
function stop_at_episode(n::Int, is_show_progress::Bool=true)
    i, p = 0, Progress(n)
    function f(env, agent)
        if agent.buffer.isdone[end]
            i += 1
            is_show_progress && next!(p; showvalues = [(:episode, i)])
            i ≥ n
        else
            false
        end
    end
end

"Return false when encountered an end of an episode"
function stop_when_done(env, agent)
    agent.buffer.isdone[end]
end

"A callback(closure) which will record the length of each episode"
function steps_per_episode()
    steps = []
    count = 0
    function acc(env, agent)
        count += 1
        if agent.buffer.isdone[end]
            push!(steps, count)
            count = 0
        end
    end
    function acc()
        steps
    end
end


"A callback(closure) which will record the total reward of each episode"
function rewards_of_each_episode()
    rewards = []
    r = 0.
    function acc(env, agent)
        r += agent.buffer.reward[end]
        if agent.buffer.isdone[end]
            push!(rewards, r)
            r = 0.
        end
    end
    function acc()
        rewards
    end
end