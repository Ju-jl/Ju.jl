export AbstractLearner, AbstractModelBasedLearner, AbstractModelFreeLearner,
       update!


"""
    AbstractLearner

Supertype for learners. A learner takes in a state and returns an action.
"""
abstract type AbstractLearner end
function update! end

abstract type AbstractModelFreeLearner <: AbstractLearner end
abstract type AbstractMonteCarloLearner <: AbstractModelFreeLearner end

abstract type AbstractModelBasedLearner <: AbstractLearner end
function update! end