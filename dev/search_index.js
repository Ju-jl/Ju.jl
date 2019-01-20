var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": "<div align=\"center\">\n<a href=\"https://en.wikipedia.org/wiki/Xiangqi#Chariot\">\n<img src=\"https://tianjun.me/static/site_resources/img/ju.svg\" alt=\"Ju.svg\" title=\"Ju\" width=\"100\"/> \n</a>\n<blockquote> \n<p> \"If I destroy you, what business is it of yours?\"</p>\n<p>― <a href=\"https://en.wikipedia.org/wiki/Liu_Cixin\">Liu Cixin</a>, <a href=\"https://en.wikipedia.org/wiki/The_Dark_Forest\">The Dark Forest</a></p>\n</blockquote>\n</div>This package aims to provide several extensible interfaces and reusable components for Reinforcement Learning."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "This package has only been tested on Julia V1.0 and above. To install this package(v1.0) pkg> add https://github.com/Ju-jl/Ju.jlOr, if you just want to have a try with docker$ docker run -it --rm tianjun2018/juThen follow the instructions in the tutorial section."
},

{
    "location": "#Tutorial-1",
    "page": "Home",
    "title": "Tutorial",
    "category": "section",
    "text": "Pages = [\"tutorial.md\"]"
},

{
    "location": "#Interfaces-1",
    "page": "Home",
    "title": "Interfaces",
    "category": "section",
    "text": "Pages = [\"interfaces.md\"]"
},

{
    "location": "#Components-1",
    "page": "Home",
    "title": "Components",
    "category": "section",
    "text": "Pages = [\"components.md\"]"
},

{
    "location": "#Utilities-1",
    "page": "Home",
    "title": "Utilities",
    "category": "section",
    "text": "Pages = [\"utilities.md\"]"
},

{
    "location": "#Related-Packages-1",
    "page": "Home",
    "title": "Related Packages",
    "category": "section",
    "text": "ReinforcementLearning.jl\nIn fact, there already exists an awesome Reinforcement Learning package. And I had cooperated with Johanni Brea on it for a short time. He is a very talented guy and I\'m pretty sure that he has the ability to push that package to a much more powerful new stage given more time. However I suspended refactoring that code due to some family issues. Several months later, I gave a sharing to my friends on how I learned Reinforcement Learning. And then I wrote some scripts to reproduce the figures on the book of Reinforcement Learning: An Introduction. One day I thought, \"How about making them more general?\". And then we have this package (and ReinforcementLearningAnIntroduction.jl). You may find that there are a lot of similarities between this package and ReinforcementLearning.jl.\nReinforce.jl\nThis package contains some interesting interfaces and it is also the first package I read while learning Reinforcement Learning in Julia.\nReinforcement Learning Package Design\nChristof Stocker shared his ideas on how to design a Reinforcement Learning package about a year ago. You may also take a look at it."
},

{
    "location": "#What\'s-Next?-1",
    "page": "Home",
    "title": "What\'s Next?",
    "category": "section",
    "text": "Take a look at the Julia code for Reinforcement Learning An Introduction(2nd).\nYou will find how easy it is to reproduce most of the figures in Reinforcement Learning An Introduction(2nd) with this package.\nWrite your own components.\nTrying to extend the existing components in this package will help you to understand both the flexibility and limitation of this package.\nConsider contributing to this package.\nA person\'s work time and energy are always limited. I would appreciate it if you would like to get involved in this project. Just open an issue or PR, and we can have a detailed discussion there."
},

{
    "location": "tutorial/#",
    "page": "Tutorial",
    "title": "Tutorial",
    "category": "page",
    "text": ""
},

{
    "location": "tutorial/#Introduction-to-Reinforcement-Learning-1",
    "page": "Tutorial",
    "title": "Introduction to Reinforcement Learning",
    "category": "section",
    "text": "Before showing you a concrete example, let\'s review some of the key concepts in RL(Reinforcement Learning).(Image: Environment_Agent_Interaction.png)Generally speaking, RL is to learn how to make actions so as to maximize a numerical reward. Two main characters in RL problems are the agent(s) and the environment. By interacting with the unknown environment, the agent seeks to achieve a goal over time. From the perspective of an agent, it needs to know the observation space and the action space of an environment.Observation Space\nThe observation of an environment could be a scalar, a vector, a matrix or a high order tensor(or a combination of them). For example, in video games we can usually use an `Array{Int8,3} to represent the observation at each time. The Observation Space tells the agent what kind of input the agent may observe.\nAction Space\nThe Action Space defines what actions are valid to an environment. Just like observations, actions can also have different dimensions and can be discrete or continuous.Some of the key concepts of an agent are that:State\nState is the agent’s internal representation of the observation from environment. It may or may not be the same with observation.\nReward\nAfter taking an action in the environment, an agent will get a new state accompanied with a numeric reward. Our goal is to maximize some notation of cumulative rewards.\nPolicy\nA policy is used to map a state to an action. It can be deterministic(a hash table) or stochastic(a parameterized classifier/approximator).\nExperience Buffer (or Trajectories)\nBuffer is used to store the agent’s experiences, including states, actions, rewards and/or some other environment information."
},

{
    "location": "tutorial/#Solving-a-Simple-Random-Walk-Problem-1",
    "page": "Tutorial",
    "title": "Solving a Simple Random Walk Problem",
    "category": "section",
    "text": "(Image: simple_random_walk.png)Suppose our friend Eve starts at the position of 3 in the above figure. It can either move left or right randomly at each time step. If it meets the dog at position 7 then receives a reward of 3 and if it meets the flower at position 1 then receives a reward of 1. In all other cases, the reward is 0. Now consider that Eve choose to move to each direction with the same probability(the policy). And we would like to estimate the value of each position. So that we can further improve the policy.First, let\'s initial the environment and the policy.julia> using Ju\n\njulia> using Random\n\njulia> Random.seed!(123);  # to ensure that you get the same result as it is documented here\n\njulia> policy = RandomPolicy([0.5, 0.5])\nRandomPolicy{1}([0.5, 0.5])\n\njulia> env = SimpleRandomWalkEnv()\nSimpleRandomWalkEnv(7, 3, 3, [-1, 1])\n\njulia> env(sample(actionspace(env)))\n(observation = 2, reward = 0.0, isdone = false)\n\njulia> reset!(env)\n(observation = 3, isdone = false)Since we want to estimate the value of each position, we choose an approximator of type TabularV to store and update our estimations(the value function).julia> approximator = TabularV(length(observationspace(env)));Then we need to decide how to update our estimations. There are many different kinds of algorithms. Here we use one of the simplest algorithms, the MonteCarloLearner. The basic idea behind MonteCarloLearner is that, we apply a policy until the end of an episode and then update the estimation of each state we have encountered by using following up rewards.julia> learner = MonteCarloLearner(approximator, policy, 0.9, 0.1);Here we set the discount rate to 0.9 and the step size of updating to 0.1.Then we use a EpisodeSARDBuffer to store the State, Action, Reward, isDone at each time step.julia> buffer = EpisodeSARDBuffer();Combining all the components above, now we have our agent.julia> agent = Agent(learner, buffer);Finally, we can train our agent!julia> train!(env, agent);Every time we call train!(env, agent), the agent will generate an action according to its policy and feed it into the env. Then the env will consume the action and return a reward, isdone and next state.julia> agent.buffer\n1-element EpisodeTurnBuffer{(:state, :action, :reward, :isdone),Tuple{Int64,Int64,Float64,Bool},NamedTuple{(:state, :action, :reward, :isdone),Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1},Array{Bool,1}}}}:\n (state = 3, action = 2, reward = 0.0, isdone = false)It will be too verbose to train our agent step by step. Fortunately, train! can accept an optional argument named callbacks. We can force the training to stop at some condition, like stop at the end of an episode.julia> callbacks=(stop_at_episode(1),);\n\njulia> train!(env, agent;callbacks=callbacks);\n\njulia> agent.buffer\n8-element EpisodeTurnBuffer{(:state, :action, :reward, :isdone),Tuple{Int64,Int64,Float64,Bool},NamedTuple{(:state, :action, :reward, :isdone),Tuple{Array{Int64,1},Array{Int64,1},Array{Float64,1},Array{Bool,1}}}}:\n (state = 3, action = 2, reward = 0.0, isdone = false)\n (state = 4, action = 2, reward = 0.0, isdone = false)\n (state = 3, action = 1, reward = 0.0, isdone = false)\n (state = 2, action = 2, reward = 0.0, isdone = false)\n (state = 3, action = 2, reward = 0.0, isdone = false)\n (state = 4, action = 1, reward = 0.0, isdone = false)\n (state = 3, action = 1, reward = 0.0, isdone = false)\n (state = 2, action = 1, reward = 1.0, isdone = true)\n\njulia> agent.learner.approximator\nTabularV([0.0, 0.06561, 0.0478297, 0.0531441, 0.0, 0.0, 0.0])As you can see, the agent.learner.approximator has been updated a little. Then we increase the number of training episodes to 1000.julia> callbacks=(stop_at_episode(1000),);\n\njulia> train!(env, agent;callbacks=callbacks);\nProgress: 100%|█████████████████████████████████████████| Time: 0:00:03\n  episode:  1000\n\njulia> agent.learner.approximator\nTabularV([0.0, 0.896335, 0.851624, 1.00933, 1.41548, 2.13538, 0.0])Now we get our estimation of each position under the discount rate of 0.9."
},

{
    "location": "interfaces/#",
    "page": "Interfaces",
    "title": "Interfaces",
    "category": "page",
    "text": ""
},

{
    "location": "interfaces/#interfaces_section-1",
    "page": "Interfaces",
    "title": "Interfaces",
    "category": "section",
    "text": "Several interfaces and abstract structures are carefully defined to make this package general enough to support both traditional reinforcement learning and deep reinforcement learning(although most algorithms in this package are targeting traditional RL currently).Pages = [\"interfaces.md\"]"
},

{
    "location": "interfaces/#Ju.AbstractEnvironment",
    "page": "Interfaces",
    "title": "Ju.AbstractEnvironment",
    "category": "type",
    "text": "AbstractEnvironment{Tos, Tas, N}\n\nSupertype for different kinds of environments with the observation space of type Tos and action space of type Tas. Here the N means the number of agents the environment can interact with.\n\nSee also: AbstractAsyncEnvironment, AbstractSyncEnvironment\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractSyncEnvironment",
    "page": "Interfaces",
    "title": "Ju.AbstractSyncEnvironment",
    "category": "type",
    "text": "AbstractSyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N}\n\nSupertype for synchronized environments with the observation space of type Tos and action space of type Tas. Here the N means the number of agents the environment can interact with. The Sync means that the environment will hang up and wait for the agent\'s input.\n\nRequired Methods Brief Description\nenv(action) Each environment env must be a functional object that receive an action as input and return a NamedTuple{(:observation, :reward, :isdone)}\nobserve(env) Return a NamedTuple{(:observation, :isdone)}\nreset!(env) Reset the environment and return a NamedTuple{(:observation, :isdone)}\nget_next_role(env) Required for multi-agent environments (N > 1). Tell the system which agent to act next\nOptional Methods \nobservationspace(env) Return the observation space of the environment. See also: AbstractSpace\nactionspace(env) Return the action space of the environment.  See also: AbstractSpace\nrender(env) Render the environment\nisend(env) Check whether the env reached an end or not. For single agent environment, observe(env).isdone is returned. For multi-agents environment, get_next_role(env) === nothing is returned\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractAsyncEnvironment",
    "page": "Interfaces",
    "title": "Ju.AbstractAsyncEnvironment",
    "category": "type",
    "text": "AbstractAsyncEnvironment{Tos, Tas, N} <: AbstractEnvironment{Tos, Tas, N}\n\nOpposed to AbstractSyncEnvironment, the Async environments will not wait for the agent\'s input. Instead it will keep running asynchronously.\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Environment-1",
    "page": "Interfaces",
    "title": "Environment",
    "category": "section",
    "text": "AbstractEnvironment\nAbstractSyncEnvironment\nAbstractAsyncEnvironment"
},

{
    "location": "interfaces/#Ju.AbstractSpace",
    "page": "Interfaces",
    "title": "Ju.AbstractSpace",
    "category": "type",
    "text": "AbstractSpace\n\nSupertype of AbstractContinuousSpace and AbstractDiscreteSpace.\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractContinuousSpace",
    "page": "Interfaces",
    "title": "Ju.AbstractContinuousSpace",
    "category": "type",
    "text": "AbstractContinuousSpace <: AbstractSpace\n\nSupertype of different kinds of continuous spaces.\n\nRequired Methods Brief Description\nsample(space) Get a random sample from the space\nBase.in(space, x) Test whether x is in the space\nOptional Methods \nBase.eltype(space) Return the type of the sample in a space\n\nSee also: AbstractDiscreteSpace\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractDiscreteSpace",
    "page": "Interfaces",
    "title": "Ju.AbstractDiscreteSpace",
    "category": "type",
    "text": "AbstractDiscreteSpace <: AbstractSpace\n\nSupertype of different kinds of discrete spaces.\n\nRequired Methods Brief Description\nsample(space) Get a random sample from the space\nBase.in(space, x) Test whether x is in the space\nBase.size(space) Return the size of the space in all dimensions\nOptional Methods \nBase.length(space) Return the number of elements in that space. By default it will be *(Base.size(space)).\nBase.eltype(space) Return the type of the sample in a space\n\nSee also: AbstractContinuousSpace\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Space-1",
    "page": "Interfaces",
    "title": "Space",
    "category": "section",
    "text": "AbstractSpace\nAbstractContinuousSpace\nAbstractDiscreteSpace"
},

{
    "location": "interfaces/#Ju.AbstractAgent",
    "page": "Interfaces",
    "title": "Ju.AbstractAgent",
    "category": "type",
    "text": "AbstractAgent\n\nSupertype of agents. Usually, an agent needs to contain at least an AbstractLearner and an AbstractTurnBuffer.\n\nRequired Methods Brief Description\nagent(obs) agent, an instance of an AbstractAgent, must be a functional object to receive an observation as input and return a pair of state and action (s => a)\nupdate!(agent, s, a, r, d, s′[, a′]) Update the agent after an interaction with environment\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Agent-1",
    "page": "Interfaces",
    "title": "Agent",
    "category": "section",
    "text": "AbstractAgent"
},

{
    "location": "interfaces/#Ju.AbstractTurnBuffer",
    "page": "Interfaces",
    "title": "Ju.AbstractTurnBuffer",
    "category": "type",
    "text": "AbstractTurnBuffer{names, types} <: AbstractArray{NamedTuple{names, types}, 1}\n\nAbstractTurnBuffer is supertype of a collection of buffers to store the interactions between agents and environments. It is a subtype of AbstractArray{NamedTuple{names, types}, 1} where names specifies which fields are to store and types is the coresponding types of the names.\n\nRequired Methods Brief Description\nBase.push!(b::AbstractTurnBuffer{names, types}, s[, a, r, d, s′, a′]) Push a turn info into the buffer. According to different names and types of the buffer b, it may accept different number of arguments\nisfull(b) Check whether the buffer is full or not\ncapacity(b) The maximum length of buffer\nBase.length(b) Return the length of buffer\nBase.getindex(b::AbstractTurnBuffer{names, types}) Return a turn of type NamedTuple{names, types}\nBase.empty!(b) Reset the buffer\nOptional Methods \nBase.size(b) Return (length(b),) by default\nBase.isempty(b) Check whether the buffer is empty or not. Return length(b) == 0 by default\nBase.lastindex(b) Return length(b) by default\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Buffer-1",
    "page": "Interfaces",
    "title": "Buffer",
    "category": "section",
    "text": "AbstractTurnBuffer"
},

{
    "location": "interfaces/#Ju.AbstractLearner",
    "page": "Interfaces",
    "title": "Ju.AbstractLearner",
    "category": "type",
    "text": "AbstractLearner\n\nSupertype for learners. A learner takes in a state and returns an action.\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Learner-1",
    "page": "Interfaces",
    "title": "Learner",
    "category": "section",
    "text": "AbstractLearner"
},

{
    "location": "interfaces/#Ju.AbstractApproximator",
    "page": "Interfaces",
    "title": "Ju.AbstractApproximator",
    "category": "type",
    "text": "AbstractApproximator\n\nAbstractApproximator is a supertype of a collection of approximators(including Tabular, Linear, DNN etc.)\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractVApproximator",
    "page": "Interfaces",
    "title": "Ju.AbstractVApproximator",
    "category": "type",
    "text": "AbstractVApproximator{Ts} <: AbstractApproximator\n\nA supertype of state value approximators with the state of type Ts.\n\nRequired Methods Brief Description\nV(s) V, an instance of AbstractVApproximator, must be a functional object which takes in state s and return the estimated value of that state\nupdate!(V, s, e) Update the value approximator of state s by a difference of e\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractQApproximator",
    "page": "Interfaces",
    "title": "Ju.AbstractQApproximator",
    "category": "type",
    "text": "AbstractQApproximator{Ts, Ta} <: AbstractApproximator\n\nA supertype of action value approximators with the state of type Ts and action of type Ta.\n\nRequired Methods Brief Description\nQ(s, a) Q, an instance of AbstractQApproximator must be a functional object which takes in state s and action a. The estimated value of that (state, action) will be returned.\nQ(s) Return the estimations of all actions at state s\nupdate!(Q, s, a, e) Update the approximator Q of state s and action a with difference e\nOptional Methods \nQ(s, Val(:max)) Return the maximum value of Q(s)\nQ(s, Val(:argmax)) Return the argmax of Q(s) (randomly break ties)\nupdate!(Q, s, errors) Update the approximator Q of state s with the difference errors of each action\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Approximator-1",
    "page": "Interfaces",
    "title": "Approximator",
    "category": "section",
    "text": "AbstractApproximator\nAbstractVApproximator\nAbstractQApproximator"
},

{
    "location": "interfaces/#Ju.AbstractEnvironmentModel",
    "page": "Interfaces",
    "title": "Ju.AbstractEnvironmentModel",
    "category": "type",
    "text": "AbstractEnvironmentModel\n\nA supertype of environment models. An environment model is used to simulate the behavior of an environment.\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractSampleModel",
    "page": "Interfaces",
    "title": "Ju.AbstractSampleModel",
    "category": "type",
    "text": "AbstractSampleModel <: AbstractEnvironmentModel\n\nAn AbstractSampleModel can be used to generate a random turn info to simulate an environment.\n\nRequired Methods Brief Description\nsample(model) Return a turn info (s, a, r, d, s′)\nupdate!(model, buffer, learner) Update the model given a buffer and a learner\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Ju.AbstractDistributionModel",
    "page": "Interfaces",
    "title": "Ju.AbstractDistributionModel",
    "category": "type",
    "text": "AbstractDistributionModel <: AbstractEnvironmentModel\n\nAn AbstractDistributionModel can return all the possible following up states, rewards, isdone info and corresponding probability.\n\nRequired Methods Brief Description\nsample(model, s, a) Return a vector of (s′, r, d, p)\nupdate!(model, buffer, learner) Update the model given a buffer and a learner\ngetstates(model) Get the number of states of the model\ngetactions(model) Get the number of actions of the model\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Environment-Model-1",
    "page": "Interfaces",
    "title": "Environment Model",
    "category": "section",
    "text": "AbstractEnvironmentModel\nAbstractSampleModel\nAbstractDistributionModel"
},

{
    "location": "interfaces/#Ju.AbstractPolicy",
    "page": "Interfaces",
    "title": "Ju.AbstractPolicy",
    "category": "type",
    "text": "AbstractPolicy\n\nSupertype for policies. A policy takes in a state and returns an action.\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Policy-1",
    "page": "Interfaces",
    "title": "Policy",
    "category": "section",
    "text": "AbstractPolicy"
},

{
    "location": "interfaces/#Ju.AbstractActionSelector",
    "page": "Interfaces",
    "title": "Ju.AbstractActionSelector",
    "category": "type",
    "text": "AbstractActionSelector\n\nA subtype of AbstractActionSelector is used to generate an action given the estimated valud of different actions.\n\nRequired Methods Brief Description\nselector(values) selector, an instance of AbstractActionSelector, must be a callable object which takes in an estimation and returns an action\n\n\n\n\n\n"
},

{
    "location": "interfaces/#Action-Selector-1",
    "page": "Interfaces",
    "title": "Action Selector",
    "category": "section",
    "text": "AbstractActionSelector"
},

{
    "location": "components/#",
    "page": "Components",
    "title": "Components",
    "category": "page",
    "text": ""
},

{
    "location": "components/#Components-1",
    "page": "Components",
    "title": "Components",
    "category": "section",
    "text": "This package has implemented some basic components which inherit from the abstract types defined at Interfaces.Pages = [\"components.md\"]"
},

{
    "location": "components/#Ju.CartPoleEnv",
    "page": "Components",
    "title": "Ju.CartPoleEnv",
    "category": "type",
    "text": "Classic cart-pole system implemented by Rich Sutton et al. See the original file at http://incompleteideas.net/sutton/book/code/pole.c. Or the python version at https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.SimpleRandomWalkEnv",
    "page": "Components",
    "title": "Ju.SimpleRandomWalkEnv",
    "category": "type",
    "text": "SimpleRandomWalkEnv\n\nA simple random walk environment for tutorial.\n\nExample\n\njulia> env = SimpleRandomWalkEnv()\nSimpleRandomWalkEnv(7, 3, 3, [-1, 1])\n\njulia> env(1)\n(observation = 2, reward = 0.0, isdone = false)\n\njulia> reset!(env)\n(observation = 3, isdone = false)\n\n\n\n\n\n"
},

{
    "location": "components/#Environments-1",
    "page": "Components",
    "title": "Environments",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractEnvironment)"
},

{
    "location": "components/#Ju.ContinuousSpace",
    "page": "Components",
    "title": "Ju.ContinuousSpace",
    "category": "type",
    "text": "struct ContinuousSpace{T<:Number} <: AbstractContinuousSpace\n    low::T\n    high::T\nend\n\nThe lower bound and upper bound are specifed by low and high.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.DiscreteSpace",
    "page": "Components",
    "title": "Ju.DiscreteSpace",
    "category": "type",
    "text": "struct DiscreteSpace <: AbstractDiscreteSpace\n    n::Int\nend\n\nThe elements in a DiscreteSpace is 1:n\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.MultiContinuousSpace",
    "page": "Components",
    "title": "Ju.MultiContinuousSpace",
    "category": "type",
    "text": "MultiContinuousSpace(low::Number, high::Number, size::Tuple{Vararg{Int}})\nMultiContinuousSpace(low::Array{<:Number}, high::Array{<:Number})\n\nExamples\n\nMultiContinuousSpace(-1, 1, (2,3))\nMultiContinuousSpace([0, 0, 0], [1, 2, 3])\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.MultiDiscreteSpace",
    "page": "Components",
    "title": "Ju.MultiDiscreteSpace",
    "category": "type",
    "text": "struct MultiDiscreteSpace{N} <:AbstractDiscreteSpace\n    counts::Array{Int, N}\nend\n\nThe element in MultiDiscreteSpace{N} is a multi-dimension array.  The number of each dimension is specified by counts.\n\n\n\n\n\n"
},

{
    "location": "components/#Spaces-1",
    "page": "Components",
    "title": "Spaces",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractSpace)"
},

{
    "location": "components/#Ju.Agent",
    "page": "Components",
    "title": "Ju.Agent",
    "category": "type",
    "text": "Agent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tpp<:Function} <: AbstractAgent \nAgent(learner::Tl, buffer::Tb, preprocessor::Tpp=identity, role=:anonymous) where {Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tpp<:Function}\n\nA preprocessor is just a normal function. It transforms the observation from an environment to the internal state, which is then stored in the buffer. role is a Symbol. Usually it is used in multi-agents environment to distinction different agents.\n\nSee also: AbstractLearner, AbstractTurnBuffer\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.Agent-Tuple{Any}",
    "page": "Components",
    "title": "Ju.Agent",
    "category": "method",
    "text": "(agent::Agent)(obs)\n\nTake in an obs from environment and use agent.preprocessor to transform it into an internal state. Then use agent.learner to get the action. Return a pair of state => action.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.DynaAgent",
    "page": "Components",
    "title": "Ju.DynaAgent",
    "category": "type",
    "text": "DynaAgent{Tl<:AbstractLearner, Tb<:AbstractTurnBuffer, Tm<:AbstractEnvironmentModel, Tpp<:Function} <: AbstractAgent\nDynaAgent(learner::Tl, buffer::Tb, model::Tm, nsteps::Int=0, preprocessor::Tpp=identity, role=:anonymous) where {Tl, Tb, Tm, Tpp}\n\nSee more details at Section (8.2) on Page 162 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\nSee also: Agent, AbstractLearner, AbstractTurnBuffer, AbstractEnvironmentModel\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.DynaAgent-Tuple{Any}",
    "page": "Components",
    "title": "Ju.DynaAgent",
    "category": "method",
    "text": "(agent::DynaAgent)(obs)\n\nTake in an obs from environment and use agent.preprocessor to transform it into an internal state. Then use agent.learner to get the action. Return a pair of state => action.\n\n\n\n\n\n"
},

{
    "location": "components/#Agents-1",
    "page": "Components",
    "title": "Agents",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractAgent)"
},

{
    "location": "components/#Ju.CircularSARDBuffer",
    "page": "Components",
    "title": "Ju.CircularSARDBuffer",
    "category": "type",
    "text": "CircularSARDBuffer is just an alias for CircularTurnBuffer{(:state, :action, :reward, :isdone)}.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.CircularSARDBuffer-Tuple{Any}",
    "page": "Components",
    "title": "Ju.CircularSARDBuffer",
    "category": "method",
    "text": "CircularSARDBuffer(capacity; state_type::Type=Int, action_type::Type=Int, state_size=(), action_size=())\n\ncapacity specifies how many latest turns the buffer will store at most. \n\nnote: Note\nNote that, the length of state and action is 1 step longer in oder to store the state and action in the next step. This is the supposed behavior of SARD buffers.\n\nSee also: sample\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.CircularSARDSABuffer",
    "page": "Components",
    "title": "Ju.CircularSARDSABuffer",
    "category": "type",
    "text": "CircularSARDSABuffer is just an alias for CircularTurnBuffer{(:state, :action, :reward, :isdone, :nextstate, :nextaction)}.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.CircularSARDSBuffer",
    "page": "Components",
    "title": "Ju.CircularSARDSBuffer",
    "category": "type",
    "text": "CircularSARDSBuffer is just an alias for CircularTurnBuffer{(:state, :action, :reward, :isdone, :nextstate)}.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.CircularTurnBuffer",
    "page": "Components",
    "title": "Ju.CircularTurnBuffer",
    "category": "type",
    "text": "CircularTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}\nCircularTurnBuffer{names, types}(capacities::NTuple{N, Int}, sizes::NTuple{N, NTuple{M, Int} where M}) where {names, types, N}\n\nUsing CircularArrayBuffer to store each element specified in names and types.. The memory of the buffer will be pre-allocated. In RL problems, three of the most common CircularArrayBuffer based buffers are: CircularSARDBuffer, CircularSARDSBuffer, CircularSARDSABuffer\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpisodeSARDBuffer",
    "page": "Components",
    "title": "Ju.EpisodeSARDBuffer",
    "category": "type",
    "text": "EpisodeSARDBuffer is just an alias for EpisodeTurnBuffer{(:state, :action, :reward, :isdone)}\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpisodeSARDSABuffer",
    "page": "Components",
    "title": "Ju.EpisodeSARDSABuffer",
    "category": "type",
    "text": "EpisodeSARDSABuffer is just an alias for EpisodeTurnBuffer{(:state, :action, :reward, :isdone, :nextstate, :nextaction)}\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpisodeSARDSBuffer",
    "page": "Components",
    "title": "Ju.EpisodeSARDSBuffer",
    "category": "type",
    "text": "EpisodeSARDSBuffer is just an alias for EpisodeTurnBuffer{(:state, :action, :reward, :isdone, :nextstate)}\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpisodeTurnBuffer",
    "page": "Components",
    "title": "Ju.EpisodeTurnBuffer",
    "category": "type",
    "text": "EpisodeTurnBuffer{names, types, Tbs} <: AbstractTurnBuffer{names, types}\nEpisodeTurnBuffer{names, types}() where {names, types}\n\nUsing a Vector to store each element specified by names and types.\n\nSee also: EpisodeSARDBuffer, EpisodeSARDSBuffer, EpisodeSARDSABuffer\n\n\n\n\n\n"
},

{
    "location": "components/#Buffers-1",
    "page": "Components",
    "title": "Buffers",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractTurnBuffer)"
},

{
    "location": "components/#Ju.DifferentialTDLearner",
    "page": "Components",
    "title": "Ju.DifferentialTDLearner",
    "category": "type",
    "text": "DifferentialTDLearner(approximator::Tapp, π::Tp, α::Float64, β::Float64, R̄::Float64=0., n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractApproximator, Tp<:PolicyOrSelector}= new{Tapp, Tp, method}(approximator, π, α, β, R̄, n)\n\nSee more details at Section (10.3) on Page 251 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.DoubleLearner",
    "page": "Components",
    "title": "Ju.DoubleLearner",
    "category": "type",
    "text": "struct DoubleLearner{Tl <: OffPolicyTDLearner, Ts<:AbstractActionSelector} <: AbstractLearner\n    Learner1::Tl\n    Learner2::Tl\n    selector::Ts\nend\n\nSee more details at Section (6.7) on Page 126 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.GradientBanditLearner",
    "page": "Components",
    "title": "Ju.GradientBanditLearner",
    "category": "type",
    "text": "struct GradientBanditLearner{Ts<:AbstractActionSelector, Tb<:Union{Float64, Function}} <: AbstractModelFreeLearner \n    Q::TabularQ \n    selector::Ts\n    α::Float64\n    baseline::Tb\nend\n\nSee more details at Section (2.8) on Page 37 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.MonteCarloExploringStartLearner",
    "page": "Components",
    "title": "Ju.MonteCarloExploringStartLearner",
    "category": "type",
    "text": "MonteCarloExploringStartLearner(approximator::Tapp, π::Tp, π_start::RandomPolicy, γ::Float64, α::Float64 = 1.0; is_first_visit::Bool = true) where {Tapp,Tp}\n\nSee more details at Section (5.3) on Page 99 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.MonteCarloLearner",
    "page": "Components",
    "title": "Ju.MonteCarloLearner",
    "category": "type",
    "text": "MonteCarloLearner(approximator::Tapp, π::Tp, γ::Float64=1., α::Float64 = 1.0, first_visit::Bool = true) where {Tapp,Tp}\n\nSee more details at Section (5.1) on Page 92 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.OffPolicyMonteCarloLearner",
    "page": "Components",
    "title": "Ju.OffPolicyMonteCarloLearner",
    "category": "type",
    "text": "OffPolicyMonteCarloLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Float64 = 1.0; isfirstvisit::Bool = true, sampling::Symbol = :OrdinaryImportanceSampling,) where {Tapp,Tpb,Tpt}\n\nSee more details at Section (5.7) on Page 111 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.OffPolicyTDLearner",
    "page": "Components",
    "title": "Ju.OffPolicyTDLearner",
    "category": "type",
    "text": "OffPolicyTDLearner(approximator::Tapp, π_behavior::Tpb, π_target::Tpt, γ::Float64, α::Float64, n::Int=0, method::Symbol=:SARSA_ImportanceSampling) where {Tapp<:AbstractApproximator, Tpb<:AbstractPolicy, Tpt<:AbstractPolicy}\n\nSee more details at Section (7.3) on Page 148 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.ReinforceBaselineLearner",
    "page": "Components",
    "title": "Ju.ReinforceBaselineLearner",
    "category": "type",
    "text": "mutable struct ReinforceBaselineLearner{Tapp<:AbstractApproximator, Tp<:AbstractPolicy}  <: AbstractModelFreeLearner \n    approximator::Tapp\n    π::Tp\n    αʷ::Float64\n    αᶿ::Float64\n    γ::Float64\nend\n\nSee more details at Section (13.4) on Page 330 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.ReinforceLearner",
    "page": "Components",
    "title": "Ju.ReinforceLearner",
    "category": "type",
    "text": "struct ReinforceLearner{Tp<:AbstractPolicy}  <: AbstractModelFreeLearner \n    π::Tp\n    α::Float64\n    γ::Float64\nend\n\nSee more details at Section (13.3) on Page 326 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TDLearner",
    "page": "Components",
    "title": "Ju.TDLearner",
    "category": "type",
    "text": "TDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0) where {Tapp<:AbstractVApproximator, Tp<:PolicyOrSelector} = new{Tapp, Tp, :SRS}(approximator, π, γ, α, n)\nTDLearner(approximator::Tapp, π::Tp, γ::Float64, α::Float64, n::Int=0, method::Symbol=:SARSA) where {Tapp<:AbstractQApproximator, Tp<:PolicyOrSelector}\n\nSee more details at Section (7.1) on Page 142 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TDλReturnLearner",
    "page": "Components",
    "title": "Ju.TDλReturnLearner",
    "category": "type",
    "text": "struct TDλReturnLearner{Tapp <: AbstractApproximator, Tp <: PolicyOrSelector} <: AbstractModelFreeLearner \n    approximator::Tapp\n    π::Tp\n    γ::Float64\n    α::Float64\n    λ::Float64\nend\n\nSee more details at Section (12.2) on Page 292 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Learners-1",
    "page": "Components",
    "title": "Learners",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractLearner)"
},

{
    "location": "components/#Approximators-1",
    "page": "Components",
    "title": "Approximators",
    "category": "section",
    "text": ""
},

{
    "location": "components/#Ju.AggregationV",
    "page": "Components",
    "title": "Ju.AggregationV",
    "category": "type",
    "text": "struct AggregationV{Tf<:Function} <: AbstractVApproximator{Int}\n    table::Vector{Float64}\n    f::Tf\nend\n\nUsing a.f to map a state s into an Int, then use a.table to check the corresponding state value.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.FourierV",
    "page": "Components",
    "title": "Ju.FourierV",
    "category": "type",
    "text": "FourierV <: AbstractVApproximator{Int}\n\nstruct FourierV <: AbstractVApproximator{Int}\n    weights::Vector{Float64}\nend\n\nUsing Fourier cosine basis to approximate the state value. weights is the featur vector.\n\nSee more details at Section (9.5.2) on Page 211 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.FourierV-Tuple{Int64}",
    "page": "Components",
    "title": "Ju.FourierV",
    "category": "method",
    "text": "FourierV(order::Int)\n\nBy specifying the order, feature vector will be initialized with 0.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.LinearV",
    "page": "Components",
    "title": "Ju.LinearV",
    "category": "type",
    "text": "struct LinearV <: AbstractVApproximator{Int}\n    features::Array{Float64, 2}\n    weights::Vector{Float64}\nend\n\nUsing a matrix features to represent each state along with a vector of weights.\n\nSee more details at Section (9.4) on Page 205 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.PolynomialV",
    "page": "Components",
    "title": "Ju.PolynomialV",
    "category": "type",
    "text": "struct PolynomialV <: AbstractVApproximator{Int}\n    weights::Vector{Float64}\nend\n\nSee more details at Section (9.5.1) on Page 210 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TabularV",
    "page": "Components",
    "title": "Ju.TabularV",
    "category": "type",
    "text": "struct TabularV <: AbstractVApproximator{Int}\n    table::Vector{Float64}\nend\n\nUsing a table of type Vector{Float64} to record the state values.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TilingsV",
    "page": "Components",
    "title": "Ju.TilingsV",
    "category": "type",
    "text": "TilingsV{Tt<:Tiling} <: AbstractVApproximator{Vector{Float64}}\nTilingsV(tilings::Vector{Tt}) where Tt<:Tiling\n\nUsing a vector of tilings to encode state. Each tiling has an independent weight.\n\nSee more details at Section (9.5.4) on Page 217 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\nSee also: Tiling, TilingsQ\n\n\n\n\n\n"
},

{
    "location": "components/#Value-Functions-1",
    "page": "Components",
    "title": "Value Functions",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractVApproximator)"
},

{
    "location": "components/#Ju.TabularQ",
    "page": "Components",
    "title": "Ju.TabularQ",
    "category": "type",
    "text": "struct TabularQ <: AbstractQApproximator{Int, Int}\n    table::Array{Float64, 2}\nend\n\nUsing a table of type Array{Float64,2} to record the action value of each state.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TabularQ",
    "page": "Components",
    "title": "Ju.TabularQ",
    "category": "type",
    "text": "TabularQ(ns::Int, na::Int=1, init::Float64=0.)\n\nInitial a table of size (ns, na) filled with value of init.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.TilingsQ",
    "page": "Components",
    "title": "Ju.TilingsQ",
    "category": "type",
    "text": "TilingsQ{Tt<:Tiling} <: AbstractQApproximator{Vector{Float64}, Int}\nTilingsQ(tilings::Vector{Tt}, nactions) where Tt<:Tiling\n\nThe only difference compared to TilingsV is that now the weight of each tiling is a matrix.\n\n\n\n\n\n"
},

{
    "location": "components/#Action-Value-Functions-(Q-functions)-1",
    "page": "Components",
    "title": "Action Value Functions (Q functions)",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractQApproximator)"
},

{
    "location": "components/#Ju.DeterministicDistributionModel",
    "page": "Components",
    "title": "Ju.DeterministicDistributionModel",
    "category": "type",
    "text": "struct DeterministicDistributionModel <: AbstractDistributionModel       table::Array{Vector{NamedTuple{(:nextstate, :reward, :prob), Tuple{Int, Float64, Float64}}}, 2}    end\n\nStore all the transformations in the table.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.DynamicDistributionModel",
    "page": "Components",
    "title": "Ju.DynamicDistributionModel",
    "category": "type",
    "text": "struct DynamicDistributionModel{Tf<:Function} <: AbstractDistributionModel\n    f::Tf\n    ns::Int\n    na::Int\nend\n\nUsing a general function f to store the transformations.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.ExperienceSampleModel",
    "page": "Components",
    "title": "Ju.ExperienceSampleModel",
    "category": "type",
    "text": "  ExperienceSampleModel <: AbstractSampleModel\n\nGenerate a turn sample based on previous experiences.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.PrioritizedSweepingSampleModel",
    "page": "Components",
    "title": "Ju.PrioritizedSweepingSampleModel",
    "category": "type",
    "text": "PrioritizedSweepingSampleModel <: AbstractSampleModel\nPrioritizedSweepingSampleModel(θ::Float64=1e-4)\n\nSee more details at Section (8.4) on Page 168 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Environment-Models-1",
    "page": "Components",
    "title": "Environment Models",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractEnvironmentModel)"
},

{
    "location": "components/#Ju.DeterministicPolicy",
    "page": "Components",
    "title": "Ju.DeterministicPolicy",
    "category": "type",
    "text": "struct DeterministicPolicy <: AbstractPolicy\n    table::Vector{Int}\n    nactions::Int\nend\n\nThe action to be adopted is stored in table.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpsilonGreedyPolicy",
    "page": "Components",
    "title": "Ju.EpsilonGreedyPolicy",
    "category": "type",
    "text": "struct EpsilonGreedyPolicy <: AbstractPolicy\n    table::Vector{Int}\n    nactions::Int\n    ϵ::Float64\nend\n\nJust like the DeterministicPolicy, the best actions are stored in the table. However the best action will only be taken at a portion of 1 - ϵ.\n\nSee also: EpsilonGreedySelector\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.FunctionalPolicy",
    "page": "Components",
    "title": "Ju.FunctionalPolicy",
    "category": "type",
    "text": "This is just a wrapper\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.LinearPolicy",
    "page": "Components",
    "title": "Ju.LinearPolicy",
    "category": "type",
    "text": "struct LinearPolicy <: AbstractPolicy\n    features::Array{Float64, 3}\n    weights::Vector{Float64}\nend\n\nThe probability of each action is calculate by features and weights and then normalized by softmax.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.RandomPolicy",
    "page": "Components",
    "title": "Ju.RandomPolicy",
    "category": "type",
    "text": "RandomPolicy(prob::Array{Float64, 2})\nRandomPolicy(prob::Vector{Float64})\n\nThe probability of each action is predefined by prob. If prob is a vector, then all states share the same prob.\n\n\n\n\n\n"
},

{
    "location": "components/#Policies-1",
    "page": "Components",
    "title": "Policies",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractPolicy)"
},

{
    "location": "components/#Ju.AlternateSelector",
    "page": "Components",
    "title": "Ju.AlternateSelector",
    "category": "type",
    "text": "AlternateSelector <: AbstractActionSelector\n\nUsed to ensure that all actions are selected alternatively.\n\nAlternateSelector(n::Int)\n\nn::Int means the optional actions are 1:n.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.AlternateSelector-Tuple{Any}",
    "page": "Components",
    "title": "Ju.AlternateSelector",
    "category": "method",
    "text": "(s::AlternateSelector)(values::Any)\n\nIgnore the action values, generate an action alternatively.\n\nExample\n\njulia> selector = AlternateSelector(3)\nAlternateSelector(3, 0)\n\njulia> any_state = 0 # for AlternateSelector, state can be anything\n\njulia> [selector(any_state) for i in 1:10]  # iterate through all actions\n10-element Array{Int64,1}:\n 1\n 2\n 3\n 1\n 2\n 3\n 1\n 2\n 3\n 1\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpsilonGreedySelector",
    "page": "Components",
    "title": "Ju.EpsilonGreedySelector",
    "category": "type",
    "text": "EpsilonGreedySelector <: AbstractActionSelector\nEpsilonGreedySelector(ϵ)\n\nThe best action is selected for a proportion 1 - epsilon and a random action (with uniform probability) is selected for a proportion epsilon. ϵ can also be a decay. See the following examples.\n\nExample\n\njulia> selector = EpsilonGreedySelector(0.1)\nEpsilonGreedySelector(0.1)\n\njulia> countmap(selector([1,2,1,1]) for _ in 1:1000)\nDict{Any,Int64} with 4 entries:\n  4 => 37\n  2 => 915\n  3 => 22\n  1 => 26\n\njulia> ϵ = expdecay(init=1.0, λ=1.0, decaystep=500, clip=0.1) (::getfield(Ju, Symbol(\"#f#34\")){Float64,Float64,Int64,Float64}) (generic function with 1 method)\n\njulia> selector = EpsilonGreedySelector(ϵ) EpsilonGreedySelector{getfield(Ju, Symbol(\"#f#34\")){Float64,Float64,Int64,Float64}}(getfield(Ju, Symbol(\"#f#34\")){Float64,Float64,Int64,Float64}(1.0, 1.0, 500, 0.1, Core.Box(-1)))\n\njulia> countmap(selector([1,2,1,1]) for _ in 1:1000) Dict{Any,Int64} with 4 entries:   4 => 101   2 => 677   3 => 106   1 => 116\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.EpsilonGreedySelector-Tuple{AbstractArray{#s15,1} where #s15<:Number}",
    "page": "Components",
    "title": "Ju.EpsilonGreedySelector",
    "category": "method",
    "text": "(p::EpsilonGreedySelector)(values::AbstractArray{T, 1}) where T\n\nnote: Note\nIf multiple values with the same maximum value are found. Then a random one will be returned!NaN will be filtered unless all the values are NaN. In that case, a random one will be returned.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.UpperConfidenceBound",
    "page": "Components",
    "title": "Ju.UpperConfidenceBound",
    "category": "type",
    "text": "UpperConfidenceBound <: AbstractActionSelector\nUpperConfidenceBound(na, c=2.0, t=0)\n\nArguments\n\nna is the number of actions used to create a internal counter.\nt is used to store current time step.\nc is used to control the degree of exploration.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.UpperConfidenceBound-Tuple{AbstractArray}",
    "page": "Components",
    "title": "Ju.UpperConfidenceBound",
    "category": "method",
    "text": "(ucb::UpperConfidenceBound)(values::AbstractArray)\n\nUnlike EpsilonGreedySelector, uncertaintyies are considered in UCB.\n\nnote: Note\nIf multiple values with the same maximum value are found. Then a random one will be returned!\n\nA_t = undersetaarg max left Q_t(a) + c sqrtfracln tN_t(a) right\n\nSee more details at Section (2.7) on Page 35 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.WeightedSample",
    "page": "Components",
    "title": "Ju.WeightedSample",
    "category": "type",
    "text": "WeightedSample <: AbstractActionSelector\nWeightedSample()\n\n\n\n\n\n"
},

{
    "location": "components/#Ju.WeightedSample-Tuple{AbstractArray}",
    "page": "Components",
    "title": "Ju.WeightedSample",
    "category": "method",
    "text": "(p::WeightedSample)(values::AbstractArray)\n\nnote: Note\nAction values are normalized to have a sum of 1.0 and then used as the probability to sample a random action.\n\n\n\n\n\n"
},

{
    "location": "components/#Action-Selectors-1",
    "page": "Components",
    "title": "Action Selectors",
    "category": "section",
    "text": "Modules = [Ju]\nFilter = is_concrete_type_of(AbstractActionSelector)"
},

{
    "location": "utilities/#",
    "page": "Utilities",
    "title": "Utilities",
    "category": "page",
    "text": ""
},

{
    "location": "utilities/#Utilities-1",
    "page": "Utilities",
    "title": "Utilities",
    "category": "section",
    "text": "Pages = [\"utilities.md\"]"
},

{
    "location": "utilities/#Ju.policy_evaluation!-Tuple{AbstractVApproximator,AbstractPolicy,AbstractDistributionModel}",
    "page": "Utilities",
    "title": "Ju.policy_evaluation!",
    "category": "method",
    "text": "policy_evaluation!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4)\n\nSee more details at Section (4.1) on Page 75 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.policy_improvement!-Tuple{AbstractVApproximator,AbstractPolicy,AbstractDistributionModel}",
    "page": "Utilities",
    "title": "Ju.policy_improvement!",
    "category": "method",
    "text": "policy_improvement!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9)\n\nSee more details at Section (4.2) on Page 76 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.policy_iteration!-Tuple{AbstractVApproximator,AbstractPolicy,AbstractDistributionModel}",
    "page": "Utilities",
    "title": "Ju.policy_iteration!",
    "category": "method",
    "text": "policy_iteration!(V::AbstractVApproximator, π::AbstractPolicy, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))\n\nSee more details at Section (4.3) on Page 80 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.train!-Tuple{Any,Any}",
    "page": "Utilities",
    "title": "Ju.train!",
    "category": "method",
    "text": "train!(env, agent; callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),))\n\nTrain an agent in env. By default callbacks is (stop_at_step(1),). So it will only train one step forward.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.train!-Union{Tuple{N}, Tuple{AbstractSyncEnvironment{Tss,Tas,N} where Tas where Tss,Tuple{Vararg{Agent{#s66,#s67,Tpp} where Tpp<:Function where #s67<:(AbstractTurnBuffer{(:state, :action, :reward, :isdone),types} where types) where #s66<:AbstractLearner,N}}}} where N",
    "page": "Utilities",
    "title": "Ju.train!",
    "category": "method",
    "text": "train!(env::AbstractSyncEnvironment{Tss, Tas, N} where {Tss, Tas},\n       agents::Tuple{Vararg{<:Agent{<:AbstractLearner, <:SARDSBuffer}, N}};\n       callbacks::Tuple{Vararg{<:Function}}=(stop_at_step(1),)) where N\n\nTODO: Add an AgentManager struct to better organize agents.\n\nFor sync environments of mulit-agents, it becomes much more complicated compared to the single agent environments. Here is an implementation for one of the most common cases. Each agent take an action alternately. In every step, all agents may observe partial/complete information of the environment from their own perspective.\n\nYou may consider to overwrite this function according to the problem you want to solve.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.value_iteration!-Tuple{AbstractVApproximator,AbstractDistributionModel}",
    "page": "Utilities",
    "title": "Ju.value_iteration!",
    "category": "method",
    "text": "value_iteration!(V::AbstractVApproximator, Model::AbstractDistributionModel; γ::Float64=0.9, θ::Float64=1e-4, max_iter=typemax(Int))\n\nSee more details at Section (4.4) on Page 83 of the book Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Core-1",
    "page": "Utilities",
    "title": "Core",
    "category": "section",
    "text": "The most widely used functions while solving RL problems are listed below:Modules = [Ju]\nPages = [\"core.jl\"]"
},

{
    "location": "utilities/#Ju.rewards_of_each_episode-Tuple{}",
    "page": "Utilities",
    "title": "Ju.rewards_of_each_episode",
    "category": "method",
    "text": "A callback(closure) which will record the total reward of each episode. Only support single agent yet.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.steps_per_episode-Tuple{}",
    "page": "Utilities",
    "title": "Ju.steps_per_episode",
    "category": "method",
    "text": "A callback(closure) which will record the length of each episode\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.stop_at_episode",
    "page": "Utilities",
    "title": "Ju.stop_at_episode",
    "category": "function",
    "text": "stop_at_episode(n::Int, is_show_progress::Bool=true)\n\nReturn a function, which will return false after n episodes. isend(env) is used to check if it is the end of an episode. is_show_progress will control whether print the progress meter or not.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.stop_at_step",
    "page": "Utilities",
    "title": "Ju.stop_at_step",
    "category": "function",
    "text": "stop_at_step(n::Int, is_show_progress::Bool=true)\n\nReturn a function, which will return false after been called n times. is_show_progress will control whether print the progress meter or not.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.stop_when_done-Tuple{Any,Any}",
    "page": "Utilities",
    "title": "Ju.stop_when_done",
    "category": "method",
    "text": "Return false when encountered an end of an episode\n\n\n\n\n\n"
},

{
    "location": "utilities/#Callbasks-1",
    "page": "Utilities",
    "title": "Callbasks",
    "category": "section",
    "text": "The callbacks are used in train!. Here are some of the predefined callbacks:Modules = [Ju]\nPages = [\"callbacks.jl\"]"
},

{
    "location": "utilities/#Ju.cached_inverse_decay-Tuple{}",
    "page": "Utilities",
    "title": "Ju.cached_inverse_decay",
    "category": "method",
    "text": "cached_inverse_decay()\n\nExample\n\njulia> f = cached_inverse_decay();\n\njulia> f(:a) # cached!\n1.0\n\njulia> f(:a) # call again\n0.5\n\njulia> f(:a) # call again\n0.3333333333333333\n\njulia> f(:b) # a new cache\n1.0\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.cached_sample_avg-Tuple{}",
    "page": "Utilities",
    "title": "Ju.cached_sample_avg",
    "category": "method",
    "text": "cached_sample_avg()\n\nExample\n\njulia> f = cached_sample_avg();\n\njulia> f(:a, 3)  # cache :a\n3.0\n\njulia> f(:a, 5)  # calculate avg value of  :a (3 + 5) / 2\n4.0\n\njulia> f(:a, 8)  # calculate avg value of  :a (3 + 5 + 8) /3\n5.333333333333333\n\njulia> f(:b, 0)  # cache another value\n0.0\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.exp_decay-Tuple{}",
    "page": "Utilities",
    "title": "Ju.exp_decay",
    "category": "method",
    "text": "exp_decay(init=1.0, λ=0.1, decay_step=1000, clip=1e-4)\n\nSee Exponential Decay\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.incrementer-Tuple{}",
    "page": "Utilities",
    "title": "Ju.incrementer",
    "category": "method",
    "text": "incrementer(;start=1, step=1)\n\nExample\n\njulia> f = incrementer();\n\njulia> [f() for _ in 1:3]\n3-element Array{Int64,1}:\n 1\n 2\n 3\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.inverse_decay-Tuple{}",
    "page": "Utilities",
    "title": "Ju.inverse_decay",
    "category": "method",
    "text": "inverse_decay()\n\nExample\n\njulia> f = inverse_decay()\n#25 (generic function with 1 method)\n\njulia> [f() for _ in 1:5]\n5-element Array{Float64,1}:\n 1.0\n 0.5\n 0.3333333333333333\n 0.25\n 0.2\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.multiplier-Tuple{}",
    "page": "Utilities",
    "title": "Ju.multiplier",
    "category": "method",
    "text": "multiplier(;start=1.0, ratio=1.0)\n\nExample\n\njulia> f = multiplier(ratio=0.9);\n\njulia> [f() for _ in 1:3]\n3-element Array{Float64,1}:\n 1.0\n 0.9\n 0.81\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.sample_avg-Tuple{}",
    "page": "Utilities",
    "title": "Ju.sample_avg",
    "category": "method",
    "text": "sample_avg()\n\nExample\n\njulia> f = sample_avg();\n\njulia> f(2)\n2.0\n\njulia> f(3) # (2+3)/2\n2.5\n\njulia> f(5) # (2+3+5)/3\n3.3333333333333335\n\n\n\n\n\n"
},

{
    "location": "utilities/#Decays-1",
    "page": "Utilities",
    "title": "Decays",
    "category": "section",
    "text": "Decays are used to efficiently calculate some variables which change based on time step.Modules = [Ju]\nPages = [\"decays.jl\"]"
},

{
    "location": "utilities/#Ju.IsFirstVisit",
    "page": "Utilities",
    "title": "Ju.IsFirstVisit",
    "category": "type",
    "text": "IsFirstVisit(itr)\n\nReturn an iterator which signifies whether each element in itr occurs for the first time.\n\nExample\n\njulia> s = [1,2,3,1,4,2,5];\n\njulia> is_first_visit(s)\n7-element Array{Bool,1}:\n  true\n  true\n  true\n false\n  true\n false\n  true\n\njulia> Iterators.reverse(IsFirstVisit(s))\nBase.Iterators.Reverse{IsFirstVisit{Array{Int64,1}}}(IsFirstVisit{Array{Int64,1}}([1, 2, 3, 1, 4, 2, 5]))\n\njulia> collect(Iterators.reverse(IsFirstVisit(s)))\n7-element Array{Any,1}:\n  true\n false\n  true\n false\n  true\n  true\n  true\n\nwarning: Warning\nAlthough IsFirstVisit supports Iterators.reverse, you should still take care of the memory usage. Internally we will walk through the itr first and calculate the count of each unique element.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.Reductions",
    "page": "Utilities",
    "title": "Ju.Reductions",
    "category": "type",
    "text": "struct Reductions{T<:Union{NamedTuple{(:init,)}, NamedTuple{()}}, F, I}\n    op::F\n    itr::I\n    kw::T\nend\n\nConstruct an iterator to get all the intermediate values while calling reduce(op, itr;kw...)\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.reductions-Tuple{Any,Any}",
    "page": "Utilities",
    "title": "Ju.reductions",
    "category": "method",
    "text": "reductions(op, iter; init)\n\nReturn an Iterator of the intermediate values of the reduction (as per reduce) of iter by op.\n\nnote: Note\nYou can not apply Iterators.reverse to Reductions (due to time complexity). If you really want to get the reversed Reductions, consider collect first and then call reverse!\n\nExample\n\njulia> reductions(+, [2,3,4])\n3-element Array{Int64,1}:\n 2\n 5\n 9\n\njulia> reductions(+, [2,3,4], init=3)\nReductions{NamedTuple{(:init,),Tuple{Int64}},typeof(+),Array{Int64,1}}(+, [2, 3, 4], (init = 3,))\n\njulia> collect(reductions(+, [2,3,4], init=3))\n4-element Array{Any,1}:\n  3\n  5\n  8\n 12\n\n\n\n\n\n"
},

{
    "location": "utilities/#StatsBase.countmap-Tuple{Any}",
    "page": "Utilities",
    "title": "StatsBase.countmap",
    "category": "method",
    "text": "extend the countmap in StatsBase to support general iterator\n\n\n\n\n\n"
},

{
    "location": "utilities/#Iterators-1",
    "page": "Utilities",
    "title": "Iterators",
    "category": "section",
    "text": "Some iterators are very helpful while implementing traditional RL algorithms.Modules = [Ju]\nPages = [\"iterators.jl\"]"
},

{
    "location": "utilities/#Ju.deletefirst!-Tuple{Array{T,1} where T,Any}",
    "page": "Utilities",
    "title": "Ju.deletefirst!",
    "category": "method",
    "text": "deletefirst!(A::Vector, element)\n\nFind the first element in A and delete it. == is used to compare equality.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.findallmax-Tuple{Any}",
    "page": "Utilities",
    "title": "Ju.findallmax",
    "category": "method",
    "text": "findallmax(A::AbstractArray)\n\nLike findmax, but all the indices of the maximum value are returned.\n\nwarning: Warning\nAll elements of value NaN in A will be ignored, unless all elements are NaN. In that case, the returned maximum value will be NaN and the returned indices will be collect(1:length(A))\n\n#Examples\n\njulia> findallmax([-Inf, -Inf, -Inf])\n(-Inf, [1, 2, 3])\n\njulia> findallmax([Inf, Inf, Inf])\n(Inf, [1, 2, 3])\n\njulia> findallmax([Inf, 0, Inf])\n(Inf, [1, 3])\n\njulia> findallmax([0,1,2,1,2,1,0])\n(2, [3, 5])\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.importance_weight-NTuple{4,Any}",
    "page": "Utilities",
    "title": "Ju.importance_weight",
    "category": "method",
    "text": "importance_weight(π, b, states, actions)\n\nCalculate the importance weight between the target policy π and behavior policy b given states and actions.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.onehot",
    "page": "Utilities",
    "title": "Ju.onehot",
    "category": "function",
    "text": "onehot(n::Int, x::Int, t::Type=Int; isdense::Bool=true)\n\nIf isdense is false, a SparseArray is returned.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.reverse_discounted_rewards-Tuple{Any,Any}",
    "page": "Utilities",
    "title": "Ju.reverse_discounted_rewards",
    "category": "method",
    "text": "reverse_discounted_rewards(rewards, γ)\n\nGiven the rewards and discount ratio γ, the discounted reward at each time step in the reversed order is returned. The returned object is of type Reductions.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.reverse_importance_weights-NTuple{4,Any}",
    "page": "Utilities",
    "title": "Ju.reverse_importance_weights",
    "category": "method",
    "text": "reverse_importance_weights(π, b, states, actions)\n\nCalculate the importance weight at each time step in the reversed order between the target policy π and behavior policy b given states and actions.\n\nThe returned object is of type Reductions\n\n\n\n\n\n"
},

{
    "location": "utilities/#Helper-Functions-1",
    "page": "Utilities",
    "title": "Helper Functions",
    "category": "section",
    "text": "Following are some commonly used functions.Modules = [Ju]\nPages = [\"helper_functions.jl\"]"
},

{
    "location": "utilities/#StatsBase.sample",
    "page": "Utilities",
    "title": "StatsBase.sample",
    "category": "function",
    "text": "sample(b::CircularSARDBuffer, batch_size::Int)\n\nSample a random batch of States, Actions, Rewards, isDone, nextStates, nextActions without replacement of batch_size.\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.CircularArrayBuffer",
    "page": "Utilities",
    "title": "Ju.CircularArrayBuffer",
    "category": "type",
    "text": "CircularArrayBuffer{E, T, N}\n\nUsing a N dimension Array to simulate a circular buffer of N-1 dimensional elements. Here E is the type of element and T is same with the eltype of E. Call eltype(b::CircularArrayBuffer{E,T,N}) will return E.\n\nExamples\n\njulia> b = CircularArrayBuffer{Float64}(2)\n0-element CircularArrayBuffer{Float64,Float64,1}\n\njulia> push!(b, rand())\n1-element CircularArrayBuffer{Float64,Float64,1}:\n 0.9709012378596478\n\njulia> push!(b, rand())\n2-element CircularArrayBuffer{Float64,Float64,1}:\n 0.9709012378596478\n 0.4510778027035365\n\njulia> push!(b, rand())\n2-element CircularArrayBuffer{Float64,Float64,1}:\n 0.4510778027035365\n 0.6774251288208646\n\njulia> b = CircularArrayBuffer{Array{Float64,2}}(3, (3,3))\n3×3×0 CircularArrayBuffer{Array{Float64,2},Float64,3}\n\njulia> push!(b, randn(3,3))\n3×3×1 CircularArrayBuffer{Array{Float64,2},Float64,3}:\n[:, :, 1] =\n -0.548592   0.926179  -1.40998\n -0.0888621  0.177208   0.342665\n  0.0925987  1.18531    0.962738\n\n\n\n\n\n"
},

{
    "location": "utilities/#Ju.Tiling",
    "page": "Utilities",
    "title": "Ju.Tiling",
    "category": "type",
    "text": "Tiling(ranges::NTuple{N, Tr}) where {N, Tr}\n\nUsing a tuple of ranges to simulate a tiling.\n\nExample\n\njulia> t = Tiling((1:2:5, 10:5:20))\nTiling{2,StepRange{Int64,Int64}}((1:2:5, 10:5:20), [1 3; 2 4])\n\njulia> Ju.encode(t, (2, 12))  # encode into an Int\n1\n\njulia> Ju.encode(t, (2, 18))\n\njulia> t2 = t - (1, 3)  # shift a little to get a new Tiling\nTiling{2,StepRange{Int64,Int64}}((0:2:4, 7:5:17), [1 3; 2 4])3\n\nSee also: TilingsV, TilingsQ\n\n\n\n\n\n"
},

{
    "location": "utilities/#Others-1",
    "page": "Utilities",
    "title": "Others",
    "category": "section",
    "text": "sample\nCircularArrayBuffer\nTiling"
},

]}
