struct FunctionalPolicy{Ta <: Function} <: AbstractPolicy
    action_generator::Ta
end

(p::FunctionalPolicy)(s) = p.action_generator(s)