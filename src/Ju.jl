module Ju

using Requires

include("interfaces.jl")
include("components.jl")
include("core.jl")

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("dynamic_loading.jl")
end

end # module