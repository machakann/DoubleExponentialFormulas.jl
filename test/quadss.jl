using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quadss32 = QuadSS(Float32)
quadss64 = QuadSS(Float64)
quadssBF = QuadSS(BigFloat)


# Test Base.show works without error
let
    io = IOBuffer()
    @test (Base.show(io, MIME("text/plain"), quadss32); true)
    @test (Base.show(io, MIME("text/plain"), quadss64); true)
    @test (Base.show(io, MIME("text/plain"), quadssBF); true)
end


# Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
# Gauss integral
let
    f(x::AbstractFloat) = exp(-x^2)
    expect = sqrt(BigFloat(π))

    I, E = quadss32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadss64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quadssBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
# Gauss integral
let
    f(x::AbstractFloat) = x^2*exp(-3*x^2)
    expect = sqrt(BigFloat(π)/3)/(2*3)

    I, E = quadss32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadss64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quadssBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test (-∞, ∞) interval (with Sinh-Sinh quadrature)
let
    f(x::AbstractFloat) = 1/(1 + x^2)
    expect = π

    I, E = quadss32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadss64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quadssBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test non-scalar output with QuadSS
let
    f(x::AbstractFloat) = [exp(-x^2), 2*exp(-x^2)]
    expect = [sqrt(BigFloat(π)), 2*sqrt(BigFloat(π))]

    I, E = quadss32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadss64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quadssBF(f, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end
