using Test

# QuadTS is for numerical integration in [-1, 1]
@testset "QuadTS" begin
    include("quadts.jl")
end

# QuadES is for numerical integration in [0, ∞]
@testset "QuadES" begin
    include("quades.jl")
end

# QuadSS is for numerical integration in [-∞, ∞]
@testset "QuadSS" begin
    include("quadss.jl")
end

# QuadDE is the general purpose numerical integrator
# If one wants to know how to use, the only file they should check is this.
@testset "QuadDE" begin
    include("quadde.jl")
end
