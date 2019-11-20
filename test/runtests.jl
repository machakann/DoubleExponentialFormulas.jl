using DoubleExponentialFormulas
using Test

@testset "DoubleExponentialFormulas.jl" begin
    # Test integrals are cited from:
    # Kahaner, D.K.: Comparison of numerical quadrature formulas, Mathematical
    # software, Rice,  J.R. (Ed.), Academic Press. pp.229-259 (1971)

    # Test problem 4
    let
        f(x::T) where {T<:AbstractFloat} = T(0.92)*cosh(x) - cos(x)
        f(x::BigFloat) = BigFloat("0.92")*cosh(x) - cos(x)
        expect = BigFloat("4.7942822668880166735857796183531e-1")

        atol = 1e-6
        I, E = quadts(f, Float32, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        I, E = quadts(f, Float64, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        I, E = quadts(f, BigFloat, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol


        atol = 1e-6
        qts = QuadTS(Float32)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        qts = QuadTS(Float64)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        qts = QuadTS(BigFloat)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol
    end


    # Test problem 5
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^4 + x^2 + T(0.9))
        f(x::BigFloat) = 1/(x^4 + x^2 + BigFloat("0.9"))
        expect = BigFloat("1.5822329637296729331174689490262e0")

        atol = 1e-6
        I, E = quadts(f, Float32, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        I, E = quadts(f, Float64, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        I, E = quadts(f, BigFloat, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol


        atol = 1e-6
        qts = QuadTS(Float32)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        qts = QuadTS(Float64)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        qts = QuadTS(BigFloat)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol
    end


    # Test problem 20
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^2 + T(1.005))
        f(x::BigFloat) = 1/(x^2 + BigFloat("1.005"))
        expect = BigFloat("1.5643964440690497730914930158085e0")

        atol = 1e-6
        I, E = quadts(f, Float32, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        I, E = quadts(f, Float64, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        I, E = quadts(f, BigFloat, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol


        atol = 1e-6
        qts = QuadTS(Float32)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-15
        qts = QuadTS(Float64)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol

        atol = 1e-17
        qts = QuadTS(BigFloat)
        I, E = qts(f, rtol=0.0, atol=atol)
        @test isapprox(I, expect, atol=atol)
        @test E ≤ atol
    end
end
