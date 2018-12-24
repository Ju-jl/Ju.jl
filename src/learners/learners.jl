export QLearner,
       GradientBanditLearner,
       MonteCarloLearner, MonteCarloExploringStartLearner, OffPolicyMonteCarloLearner,
       TDLearner, OffPolicyTDLearner, DoubleLearner, QLearner, DifferentialTDLearner, TDλReturnLearner,
       ReinforceLearner,  ReinforceBaselineLearner

# include("Q_learner.jl")
include("gradient_bandit_learner.jl")
include("monte_carlo_learner.jl")
include("temporal_difference_learner.jl")
include("reinforce_learner.jl")