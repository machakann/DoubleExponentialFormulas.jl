using DoubleExponentialFormulas
using Test

function examine(q, f; expect, atol)
    # Error tolerance
    N = 10

    # Test
    I, E = q(f, atol=atol)
    @test isapprox(I, expect, atol=N*atol)
    @test E ≤ atol
end

@testset "DoubleExponentialFormulas.jl" begin
    # Test integrals are cited from:
    # Kahaner, D.K.: Comparison of numerical quadrature formulas, Mathematical
    # software, Rice,  J.R. (Ed.), Academic Press. pp.229-259 (1971)

    # Test problem 4
    let
        f(x::T) where {T<:AbstractFloat} = T(0.92)*cosh(x) - cos(x)
        f(x::BigFloat) = BigFloat("0.92")*cosh(x) - cos(x)
        expect = BigFloat("4.7942822668880166735857796183531e-1")
        examine(QuadTS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadTS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadTS(BigFloat), f, expect=expect, atol=1e-17)
    end


    # Test problem 5
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^4 + x^2 + T(0.9))
        f(x::BigFloat) = 1/(x^4 + x^2 + BigFloat("0.9"))
        expect = BigFloat("1.5822329637296729331174689490262e0")
        examine(QuadTS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadTS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadTS(BigFloat), f, expect=expect, atol=1e-17)
    end


    # Test problem 20
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^2 + T(1.005))
        f(x::BigFloat) = 1/(x^2 + BigFloat("1.005"))
        expect = BigFloat("1.5643964440690497730914930158085e0")
        examine(QuadTS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadTS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadTS(BigFloat), f, expect=expect, atol=1e-17)
    end


    # Test [0, ∞) interval (with Exp-Sinh quadrature)
    let
        f(x::AbstractFloat) = exp(-x)
        expect = 1
        examine(QuadES(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadES(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadES(BigFloat), f, expect=expect, atol=1e-17)


        f(x::AbstractFloat) = -exp(-x)*log(x)
        expect = BigFloat("5.77215664901532860606512090082e-1")  # γ constant
        examine(QuadES(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadES(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadES(BigFloat), f, expect=expect, atol=1e-17)


        f(x::AbstractFloat) = 2/(1 + x^2)
        expect = π
        examine(QuadES(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadES(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadES(BigFloat), f, expect=expect, atol=1e-17)
    end


    # Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
    let
        f(x::AbstractFloat) = exp(-x^2)
        expect = sqrt(BigFloat(π))
        examine(QuadSS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadSS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadSS(BigFloat), f, expect=expect, atol=1e-17)


        f(x::AbstractFloat) = x^2*exp(-3*x^2)
        expect = sqrt(BigFloat(π)/3)/(2*3)
        examine(QuadSS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadSS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadSS(BigFloat), f, expect=expect, atol=1e-17)


        f(x::AbstractFloat) = 1/(1 + x^2)
        expect = π
        examine(QuadSS(Float32),  f, expect=expect, atol=1e-6)
        examine(QuadSS(Float64),  f, expect=expect, atol=1e-14)
        examine(QuadSS(BigFloat), f, expect=expect, atol=1e-17)
    end
end
