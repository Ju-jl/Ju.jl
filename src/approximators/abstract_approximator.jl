export AbstractApproximator, AbstractQApproximator, AbstractVApproximator,
       update!

"""
    AbstractApproximator

`AbstractApproximator` is a supertype of a collection of approximators(including Tabular, Linear, DNN etc.)
"""
abstract type AbstractApproximator end

"""
    AbstractVApproximator{Ts} <: AbstractApproximator

A supertype of state value approximators with the state of type `Ts`.

| Required Methods| Brief Description |
|:----------------|:------------------|
| `V(s)` | `V`, an instance of `AbstractVApproximator`, must be a functional object which takes in state `s` and return the estimated value of that state |
| `update!(V, s, e)` | Update the value approximator of state `s` by a difference of `e` |
"""
abstract type AbstractVApproximator{Ts} <: AbstractApproximator end

"""
    AbstractQApproximator{Ts, Ta} <: AbstractApproximator 

A supertype of action value approximators with the state of type `Ts` and action of type `Ta`.


| Required Methods| Brief Description |
|:----------------|:------------------|
| `Q(s, a)` | `Q`, an instance of `AbstractQApproximator` must be a functional object which takes in state `s` and action `a`. The estimated value of that (state, action) will be returned. |
| `Q(s)` | Return the estimations of all actions at state `s` |
| `update!(Q, s, a, e)` | Update the approximator `Q` of state `s` and action `a` with difference `e` |
| **Optional Methods** | |
| `Q(s, Val(:max))` | Return the maximum value of `Q(s)` |
| `Q(s, Val(:argmax))` | Return the argmax of `Q(s)` (randomly break ties) |
| `update!(Q, s, errors)` | Update the approximator `Q` of state `s` with the difference `errors` of each action |
"""
abstract type AbstractQApproximator{Ts, Ta} <: AbstractApproximator end