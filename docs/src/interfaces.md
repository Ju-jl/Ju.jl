# [Interfaces](@id interfaces_section)

Several interfaces and abstract structures are carefully defined to make this package general enough to support both traditional reinforcement learning and deep reinforcement learning(although most algorithms in this package are targeting traditional RL currently).

```@contents
Pages = ["interfaces.md"]
```

## Environment

```@docs
AbstractEnvironment
AbstractSyncEnvironment
AbstractAsyncEnvironment
```

## Space

```@docs
AbstractSpace
AbstractContinuousSpace
AbstractDiscreteSpace
```

## Agent

```@docs
AbstractAgent
```

## Buffer

```@docs
AbstractTurnBuffer
```

## Learner

```@docs
AbstractLearner
```

## Approximator

```@docs
AbstractApproximator
AbstractVApproximator
AbstractQApproximator
```

## Environment Model

```@docs
AbstractEnvironmentModel
AbstractSampleModel
AbstractDistributionModel
```

## Policy

```@docs
AbstractPolicy
```

## Action Selector

```@docs
AbstractActionSelector
```