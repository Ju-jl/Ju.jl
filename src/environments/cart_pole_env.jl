"""
Classic cart-pole system implemented by Rich Sutton et al.
See the original file at [http://incompleteideas.net/sutton/book/code/pole.c](http://incompleteideas.net/sutton/book/code/pole.c).
Or the python version at [https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
"""
struct CartPoleEnv
    gravity::Float64
    masscart::Float64
    masspole::Float64
    total_mass::Float64
    length::Float64
    polemass_length::Float64
    force_mag::Float64
    tau::Float64
    theta_threshold_radians::Float64
    x_threshold::Float64
    max_steps::Int64

    t::Int64
    state::Vector{Float64}
    isdone::Bool

    CartPoleEnv(;
        gravity                 = 9.8,
        masscart                = 1.0,
        masspole                = 0.1,
        total_mass              = masscart + masspole,
        length                  = 0.5,
        polemass_length         = masspole * length,
        force_mag               = 10.0,
        tau                     = 0.02,
        theta_threshold_radians = 12 * 2 * π / 360,
        x_threshold             = 2.4,
        max_steps               = 200
    ) = new(gravity, masscart, masspole, total_mass, length, polemass_length, force_mag, tau, theta_threshold_radians, x_threshold, max_steps,
            0,
            rand(4) ./ 10 .- 0.05)
end

function (env::CartPoleEnv)(a)
    env.t += 1
    x, x_dot, theta, theta_dot = env.state
    force = a == 2 ? env.force_mag : -env.force_mag
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.polemass_length * theta_dot * theta_dot * sintheta) / env.totalmass
    thetaacc = (env.gravity * sintheta - costheta * temp) / (env.length * (4.0 / 3.0 - env.masspole * costheta * costheta / env.total_mass))
    xacc  = temp - env.polemass_length * thetaacc * costheta / env.total_mass

    env.state[1] += env.params.tau * xdot
    env.state[2] += env.params.tau * xacc
    env.state[3] += env.params.tau * thetadot
    env.state[4] += env.params.tau * thetaacc

    env.isdone = abs(env.state[1]) > env.x_threshold ||
                 abs(env.state[3]) > env.theta_threshold ||
                 env.t >= env.params.maxsteps
    
    (observation = env.state,
     reward      = 1.0,
     isdone      = env.isdone)
end

function reset!(env::CartPoleEnv)
    env.t = 0
    env.state .= rand(4) ./ 10 .- 0.05
    env.isdone = false
    (observation = env.state,
     isdone      = env.isdone)
end

observe(env::CartPoleEnv) = (observation=env.state, isdone=env.isdone)
observationspace(env::CartPoleEnv) = MultiContinuousSpace([typemin(Float64) for _ in 1:4], [typemax(Float64) for _ in 1:4])
actionspace(env::CartPoleEnv) = DiscreteSpace(2)