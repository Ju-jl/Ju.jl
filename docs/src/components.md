# Components

This package has implemented some basic components which inherit from the abstract types defined at [`Interfaces`](@ref interfaces_section).

```@contents
Pages = ["components.md"]
```

## Environments

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractEnvironment)
```

## Spaces

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractSpace)
```

## Agents

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractAgent)
```

## Buffers

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractTurnBuffer)
```

## Learners

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractLearner)
```

## Approximators

### Value Functions

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractVApproximator)
```

### Action Value Functions (Q functions)

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractQApproximator)
```

## Environment Models

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractEnvironmentModel)
```

## Policies

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractPolicy)
```

## Action Selectors

```@autodocs
Modules = [Ju]
Filter = is_concrete_type_of(AbstractActionSelector)
```