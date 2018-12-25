FROM julia:1.0

ADD . /Ju.jl
WORKDIR /Ju.jl
RUN ["julia", "-e", "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); pkg\"precompile\""]
CMD ["julia", "--project"]