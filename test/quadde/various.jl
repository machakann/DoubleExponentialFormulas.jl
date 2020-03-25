using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quadde32 = QuadDE(Float32)
quadde64 = QuadDE(Float64)
quaddeBF = QuadDE(BigFloat)


# Test Base.show works without error
let
    io = IOBuffer()
    @test (Base.show(io, MIME("text/plain"), quadde32); true)
    @test (Base.show(io, MIME("text/plain"), quadde64); true)
    @test (Base.show(io, MIME("text/plain"), quaddeBF); true)
end


# Abel transform
let
    f(r::AbstractFloat) = exp(-(r^2 - 1))
    expect = sqrt(BigFloat(π))

    y = 1.0
    g = r -> 2*f(r)*r/sqrt(r^2 - y^2)

    I, E = quadde32(g, 1, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quadde64(g, 1, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(g, 1, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end
