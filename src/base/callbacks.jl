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
    f() = i
end

"""
    stop_at_episode(n::Int, is_show_progress::Bool=true)

Return a function, which will return false after `n` episodes.
`isend(env)` is used to check if it is the end of an episode.
`is_show_progress` will control whether print the progress meter or not.
"""
function stop_at_episode(n::Int, is_show_progress::Bool=true)
    i, p = 0, Progress(n)
    function f(env, agent)
        if isend(env)
            i += 1
            is_show_progress && next!(p; showvalues = [(:episode, i)])
            i ≥ n
        else
            false
        end
    end
    f() = i
end

"Return false when encountered an end of an episode"
stop_when_done(env, agent) = isend(env)

"A callback(closure) which will record the length of each episode"
function steps_per_episode()
    steps = []
    count = 0
    function acc(env, agent)
        count += 1
        if isend(env)
            push!(steps, count)
            count = 0
        end
    end
    function acc()
        steps
    end
end


"""
A callback(closure) which will record the total reward of each episode.
Only support single agent yet.
"""
function rewards_of_each_episode()
    rewards = []
    r = 0.
    function acc(env, agent)
        r += buffer(agent).reward[end]
        if isend(env)
            push!(rewards, r)
            r = 0.
        end
    end
    function acc()
        rewards
    end
end