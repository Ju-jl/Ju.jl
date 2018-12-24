"""
    struct DynamicDistributionModel{Tf<:Function} <: AbstractDistributionModel
        f::Tf
        ns::Int
        na::Int
    end

Using a general function `f` to store the transformations.
"""
struct DynamicDistributionModel{Tf<:Function} <: AbstractDistributionModel
    f::Tf
    ns::Int
    na::Int
end

function update!(::DynamicDistributionModel, ::AbstractTurnBuffer) end

getstates(m::DynamicDistributionModel) = m.ns
getactions(m::DynamicDistributionModel) = m.na
(m::DynamicDistributionModel)(s, a) = m.f(s, a)