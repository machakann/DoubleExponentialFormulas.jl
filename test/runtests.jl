using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

@testset "DoubleExponentialFormulas.jl" begin
    # Test integrals are cited from:
    # Kahaner, D.K.: Comparison of numerical quadrature formulas, Mathematical
    # software, Rice, J.R. (Ed.), Academic Press. pp.229-259 (1971)

    # Test problem 1
    let
        f(x::AbstractFloat) = exp(x)
        expect = BigFloat("1.71828182845904523536028747135")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 2
    # non-smooth function
    let
        f(x::AbstractFloat) = floor(min(x/3*10, one(x)))
        expect = BigFloat("7.00000000000000000000000000000e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)
    end


    # Test problem 3
    let
        f(x::AbstractFloat) = sqrt(x)
        expect = BigFloat("6.66666666666666666666666666666e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 4
    let
        f(x::T) where {T<:AbstractFloat} = cosh(x)*92/100 - cos(x)
        expect = BigFloat("4.7942822668880166735857796183531e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadTS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 5
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^4 + x^2 + T(0.9))
        f(x::BigFloat) = 1/(x^4 + x^2 + BigFloat("0.9"))
        expect = BigFloat("1.5822329637296729331174689490262e0")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadTS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 6
    let
        f(x::AbstractFloat) = x*sqrt(x)
        expect = BigFloat("4.00000000000000000000000000000e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 7
    let
        f(x::AbstractFloat) = 1/sqrt(x)
        expect = 2

        rtol = 1e-3
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-8
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-12
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 8
    let
        f(x::AbstractFloat) = 1/(x^4 + 1)
        expect = BigFloat("8.66972987339911037573995163882e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 9
    let
        f(x::AbstractFloat) = 2/(2 + sin(x*314159/10000))
        expect = BigFloat("1.15470066904371304340692220986e0")

        rtol = 1e-5
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)
    end


    # Test problem 10
    let
        f(x::AbstractFloat) = 1/(1 + x)
        expect = BigFloat("6.93147180559945309417232121458e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 11
    let
        f(x::AbstractFloat) = 1/(exp(x) + 1)
        expect = BigFloat("3.7988549304172247536823662649e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 12
    let
        f(x::AbstractFloat) = x/(exp(x) - 1)
        expect = BigFloat("7.77504634112248276417586545425e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 13
    # Attenuating oscillation
    let
        f(x::AbstractFloat) = sin(x*314159/1000)/(x*314159/100000)
        expect = BigFloat("9.0986452565692970698e-3")

        # FIXME: Works well if smaller h0 is given
        rtol = 1e-5
        I, E = QuadTS(Float32, h0=one(Float32)/32)(f, 0.1, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64, h0=one(Float64)/32)(f, 0.1, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-18
        I, E = QuadTS(BigFloat, h0=one(BigFloat)/32)(f, 0.1, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 14
    # Super fast decay around x ~ 0
    # FIXME: split the integral range into parts
    let
        f(x::T) where {T<:AbstractFloat} = 5*sqrt(T(2))*exp(x^2*(-50)*314_159/100_000)
        expect = BigFloat("5.0000021117e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadTS(BigFloat)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)
    end


    # Test problem 15
    # Super fast decay around x ~ 0
    # FIXME: split the integral range into parts
    let
        f(x::AbstractFloat) = 25*exp(-x*25)
        expect = 1

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-11
        I, E = QuadTS(Float64)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-11
        I, E = QuadTS(BigFloat)(f, 0, 10, rtol=rtol)
        @test_broken isapprox(I, expect, atol=0.01)
        @test_skip E ≤ rtol*norm(I)
    end


    # Test problem 16
    # Super fast decay around x ~ 0
    let
        f(x::AbstractFloat) = 1/(2500*x^2 + 1)*50/314_159*100_000
        expect = BigFloat("4.99363802871016550828171090341e-1")

        rtol = 1e-5
        I, E = QuadTS(Float32)(f, 0, 10, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0, 10, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 10, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 17
    let
        f(x::AbstractFloat) = (sin(x*50*314_159/100_000)/(x*50*314_159/100_000))^2*50
        expect = BigFloat("1.12139569626709460839885e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0.01, 1, rtol=rtol)
        # FIXME: necessarily to give atol?
        @test isapprox(I, expect, atol=1e-5)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, 0.01, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        # NOTE: the accuracy is not improved so much?
        rtol = 1e-15
        I, E = QuadTS(BigFloat)(f, 0.01, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 18
    # Attenuating oscillation
    # NOTE: expect value is doubtful...
    let
        f(x::AbstractFloat) = cos(cos(x) + 3*sin(x) + 2*cos(2x) + 3*sin(2x) + 3*cos(3x))
        expect = BigFloat("8.386763233809718250439e-1")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, π, rtol=rtol)
        @test isapprox(I, expect, atol=1e-5)
        @test E ≤ rtol*norm(I)

        rtol = 1e-11
        I, E = QuadTS(Float64)(f, 0, π, rtol=rtol)
        @test isapprox(I, expect, atol=1e-7)
        @test E ≤ rtol*norm(I)

        rtol = 1e-11
        I, E = QuadTS(BigFloat)(f, 0, π, rtol=rtol)
        @test isapprox(I, expect, atol=1e-7)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 19
    let
        f(x::AbstractFloat) = log(x)
        expect = -1

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-11
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-30
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 20
    let
        f(x::T) where {T<:AbstractFloat} = 1/(x^2 + T(1.005))
        f(x::BigFloat) = 1/(x^2 + BigFloat("1.005"))
        expect = BigFloat("1.5643964440690497730914930158085e0")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadTS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadTS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test problem 21
    # including spikes
    # FIXME: split the integral range into parts
    let
        f(x::AbstractFloat) = 1/cosh((10x-2))^2+1/cosh((100x-40))^4+1/cosh((1000x-600))^6
        expect = BigFloat("0.2108027355005492773756")

        rtol = 1e-6
        I, E = QuadTS(Float32)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-13
        I, E = QuadTS(Float64)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadTS(BigFloat)(f, 0, 1, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)
    end


    # Test [0, ∞) interval (with Exp-Sinh quadrature)
    let
        f(x::AbstractFloat) = exp(-x)
        expect = 1

        rtol = 1e-6
        I, E = QuadES(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadES(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadES(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end

    # Test [0, ∞) interval (with Exp-Sinh quadrature)
    let
        f(x::AbstractFloat) = -exp(-x)*log(x)
        expect = BigFloat("5.77215664901532860606512090082e-1")  # γ constant

        rtol = 1e-6
        I, E = QuadES(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadES(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadES(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test [0, ∞) interval (with Exp-Sinh quadrature)
    let
        f(x::AbstractFloat) = 2/(1 + x^2)
        expect = π

        rtol = 1e-6
        I, E = QuadES(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadES(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadES(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test [0, ∞) interval (with Exp-Sinh quadrature)
    # Dirichlet integral
    let
        f(x::AbstractFloat) = 2*sin(x)/x
        expect = π

        rtol = 1e-6
        I, E = QuadES(Float32)(f, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadES(Float64)(f, rtol=rtol)
        @test_broken isapprox(I, expect, rtol=10rtol)
        @test_skip E ≤ rtol*norm(I)

        # FIXME: Something wrong. Infinite loop?
        # rtol = 1e-17
        # I, E = QuadES(BigFloat)(f, rtol=rtol)
        # @test_broken isapprox(I, expect, rtol=10rtol)
        # @test_skip E ≤ rtol*norm(I)
    end


    # Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
    # Gauss integral
    let
        f(x::AbstractFloat) = exp(-x^2)
        expect = sqrt(BigFloat(π))

        rtol = 1e-6
        I, E = QuadSS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadSS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadSS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
    # Gauss integral
    let
        f(x::AbstractFloat) = x^2*exp(-3*x^2)
        expect = sqrt(BigFloat(π)/3)/(2*3)

        rtol = 1e-6
        I, E = QuadSS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadSS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadSS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end


    # Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
    let
        f(x::AbstractFloat) = 1/(1 + x^2)
        expect = π

        rtol = 1e-6
        I, E = QuadSS(Float32)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-14
        I, E = QuadSS(Float64)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)

        rtol = 1e-17
        I, E = QuadSS(BigFloat)(f, rtol=rtol)
        @test isapprox(I, expect, rtol=10rtol)
        @test E ≤ rtol*norm(I)
    end
end
