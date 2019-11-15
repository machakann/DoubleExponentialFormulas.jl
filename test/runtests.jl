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
        @test isapprox(quadts(f, Float32, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-15
        @test isapprox(quadts(f, Float64, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-17
        @test isapprox(quadts(f, BigFloat, rtol=0.0, atol=atol), expect, atol=atol)
    end


    # Test problem 5
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^4 + x^2 + T(0.9))
        f(x::BigFloat) = 1/(x^4 + x^2 + BigFloat("0.9"))
        expect = BigFloat("1.5822329637296729331174689490262e0")

        atol = 1e-6
        @test isapprox(quadts(f, Float32, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-15
        @test isapprox(quadts(f, Float64, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-17
        @test isapprox(quadts(f, BigFloat, rtol=0.0, atol=atol), expect, atol=atol)
    end


    # Test problem 20
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^2 + T(1.005))
        f(x::BigFloat) = 1/(x^2 + BigFloat("1.005"))
        expect = BigFloat("1.5643964440690497730914930158085e0")

        atol = 1e-6
        @test isapprox(quadts(f, Float32, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-15
        @test isapprox(quadts(f, Float64, rtol=0.0, atol=atol), expect, atol=atol)

        atol = 1e-17
        @test isapprox(quadts(f, BigFloat, rtol=0.0, atol=atol), expect, atol=atol)
    end
end
