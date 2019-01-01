```@raw html
<div align="center">
<a href="https://en.wikipedia.org/wiki/Xiangqi#Chariot">
<img src="https://tianjun.me/static/site_resources/img/ju.svg" alt="Ju.svg" title="Ju" width="100"/> 
</a>
<blockquote> 
<p> "If I destroy you, what business is it of yours?"</p>
<p>― <a href="https://en.wikipedia.org/wiki/Liu_Cixin">Liu Cixin</a>, <a href="https://en.wikipedia.org/wiki/The_Dark_Forest">The Dark Forest</a></p>
</blockquote>
</div>
```

This package aims to provide several extensible interfaces and reusable components for **Reinforcement Learning**.

## Installation

This package has only been tested on Julia V1.0 and above. To install this package

```julia
(v1.0) pkg> add https://github.com/Ju-jl/Ju.jl
```

Or, if you just want to have a try with [docker](https://docs.docker.com/install/)

```
$ docker run -it --rm tianjun2018/ju
```

Then follow the instructions in the tutorial section.

## Tutorial

```@contents
Pages = ["tutorial.md"]
```

## Interfaces

```@contents
Pages = ["interfaces.md"]
```

## Components

```@contents
Pages = ["components.md"]
```

## Utilities

```@contents
Pages = ["utilities.md"]
```

## Related Packages

- [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl)

    In fact, there already exists an awesome Reinforcement Learning package. And I had cooperated with [Johanni Brea](https://github.com/jbrea) on it for a short time. He is a very talented guy and I'm pretty sure that he has the ability to push that package to a much more powerful new stage given more time. However I suspended refactoring that code due to some family issues. Several months later, I gave a sharing to my friends on how I learned Reinforcement Learning. And then I wrote some scripts to reproduce the figures on the book of [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). One day I thought, "How about making them more general?". And then we have this package (and [ReinforcementLearningAnIntroduction.jl](https://github.com/Ju-jl/ReinforcementLearningAnIntroduction.jl)). You may find that there are a lot of similarities between this package and [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl).

- [Reinforce.jl](https://github.com/JuliaML/Reinforce.jl)

    This package contains some interesting interfaces and it is also the first package I read while learning Reinforcement Learning in Julia.

- [Reinforcement Learning Package Design](https://github.com/Evizero/ReinforcementLearning.jl/blob/master/docs/src/devdocs/design.md)

    [Christof Stocker](https://github.com/Evizero) shared his ideas on how to design a Reinforcement Learning package about a year ago. You may also take a look at it.


## What's Next?

1. Take a look at the [Julia code](https://github.com/Ju-jl/ReinforcementLearningAnIntroduction.jl) for *Reinforcement Learning An Introduction(2nd)*.

    You will find how easy it is to reproduce most of the figures in *Reinforcement Learning An Introduction(2nd)* with this package.
1. Write your own components.

    Trying to extend the existing components in this package will help you to understand both the flexibility and limitation of this package.
1. Consider contributing to this package.

    A person's work time and energy are always limited. I would appreciate it if you would like to get involved in this project. Just open an issue or PR, and we can have a detailed discussion there.