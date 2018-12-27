using Documenter
using Ju

is_concrete_type_of(t) = x -> x isa Type && x <: t && !isabstracttype(x)

makedocs(
    sitename = "Ju.jl",
    format = Documenter.HTML(),
    assets = ["assets/favicon.ico"],
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
)