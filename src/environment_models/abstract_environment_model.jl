import StatsBase:sample
export AbstractEnvironmentModel, AbstractSampleModel, AbstractDistributionModel,
       update!, getstates, getactions

"""
    AbstractEnvironmentModel

A supertype of environment models. An environment model is used to simulate the behavior of an environment.
"""
abstract type AbstractEnvironmentModel end

"""
    AbstractSampleModel <: AbstractEnvironmentModel

An `AbstractSampleModel` can be used to generate a random turn info to simulate an environment.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `sample(model)` | Return a turn info (s, a, r, d, s′) |
| `update!(model, buffer, learner)` | Update the `model` given a `buffer` and a `learner` |
"""
abstract type AbstractSampleModel <: AbstractEnvironmentModel end

"""
    AbstractDistributionModel <: AbstractEnvironmentModel

An `AbstractDistributionModel` can return all the possible following up states, rewards, isdone info and corresponding probability.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `sample(model, s, a)` | Return a vector of (s′, r, d, p) |
| `update!(model, buffer, learner)` | Update the `model` given a `buffer` and a `learner` |
| `getstates(model)` | Get the number of states of the model |
| `getactions(model)` | Get the number of actions of the model |
"""
abstract type AbstractDistributionModel <: AbstractEnvironmentModel end

function update! end
function sample end
function getstates end
function getactions end