using Documenter
using Ju

is_concrete_type_of(t) = x -> x isa Type && x <: t && !isabstracttype(x)

makedocs(
    sitename = "Ju.jl",
    format = Documenter.HTML(),
    modules = [Ju],
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Interfaces" => "interfaces.md",
        "Components" => "components.md",
        "Utilities" => "utilities.md"
    ]
)

deploydocs(
    repo = "github.com/Ju-jl/Ju.jl.git",
    target = "build",
    julia = "1.0",
    osname = "linux",
    deps = nothing,
    make = nothing,
)