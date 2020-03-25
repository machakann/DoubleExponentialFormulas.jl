using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quades32 = QuadES(Float32)
quades64 = QuadES(Float64)
quadesBF = QuadES(BigFloat)


# Test Base.show works without error
let
    io = IOBuffer()
    @test (Base.show(io, MIME("text/plain"), quades32); true)
    @test (Base.show(io, MIME("text/plain"), quades64); true)
    @test (Base.show(io, MIME("text/plain"), quadesBF); true)
end


# Test [0, ∞] interval (with Exp-Sinh quadrature)
let
    f(x::AbstractFloat) = exp(-x)
    expect = 1

    I, E = quades32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quades64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    rtol = 1e-30
    I, E = quadesBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test [0, ∞] interval (with Exp-Sinh quadrature)
let
    f(x::AbstractFloat) = -exp(-x)*log(x)
    expect = BigFloat("5.77215664901532860606512090082e-1")  # γ constant

    I, E = quades32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quades64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    rtol = 1e-30
    I, E = quadesBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test [0, ∞] interval (with Exp-Sinh quadrature)
let
    f(x::AbstractFloat) = 2/(1 + x^2)
    expect = π

    I, E = quades32(f)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quades64(f)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    rtol = 1e-30
    I, E = quadesBF(f, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end


# Test [0, ∞] interval (with Exp-Sinh quadrature)
# Dirichlet integral
let
    f(x::AbstractFloat) = 2*sin(x)/x
    expect = π

    I, E = quades32(f)
    @test I isa Float32
    @test_broken I ≈ expect
    @test_skip E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quades64(f)
    @test I isa Float64
    @test_broken I ≈ expect
    @test_skip E ≤ sqrt(eps(typeof(I)))*norm(I)

    # FIXME: Something wrong. Infinite loop?
    # rtol = 1e-17
    # I, E = quadesBF(f, rtol=rtol)
    # @test I isa BigFloat
    # @test_broken isapprox(I, expect, rtol=10rtol)
    # @test_skip E ≤ rtol*norm(I)
end


# Test non-scalar output with QuadES
let
    f(x::AbstractFloat) = [exp(-x), 2exp(-x)]
    expect = [1, 2]

    I, E = quades32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quades64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quadesBF(f, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end
