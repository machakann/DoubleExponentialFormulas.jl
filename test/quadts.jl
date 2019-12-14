using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quadts32 = QuadTS(Float32)
quadts64 = QuadTS(Float64)
quadtsBF = QuadTS(BigFloat)


let
    f(x) = 1
    expect = 2

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end


let
    f(x) = x
    expect = 0

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end


let
    f(x) = 3*x^2
    expect = 2

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end


let
    f(x) = 2/(1 + x^2)
    expect = π

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end


# Singular points at x = ±1
let
    f(x) = (2 - x)/sqrt(1 - x^2)
    expect = 2π

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test isapprox(I, expect, rtol=1e-3)
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test isapprox(I, expect, rtol=1e-7)
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=1e-16)
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end


# Test non-scalar output with QuadTS
let
    f(x::AbstractFloat) = [1/(1 + x^2), 2/(1 + x^2)]
    expect = [BigFloat(π)/2, BigFloat(π)]

    I, E = quadts32(f)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadts64(f)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadtsBF(f)
    @test eltype(I) == BigFloat
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)
end
