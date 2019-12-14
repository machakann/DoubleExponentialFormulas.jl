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
# NOTE: If one wants to know how to use, the only files they should check are
#       the below items.
@testset "QuadDE" begin
    @testset "Varius intervals" begin
        include("quadde/intervals.jl")
    end

    @testset "Basic principles" begin
        include("quadde/principles.jl")
    end

    @testset "Kahaner's problems" begin
        include("quadde/Kahaner.jl")
    end
end
