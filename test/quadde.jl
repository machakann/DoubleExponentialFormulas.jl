using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quadde32 = QuadDE(Float32)
quadde64 = QuadDE(Float64)
quaddeBF = QuadDE(BigFloat)

# Test integral interval [-1, 1] with QuadDE
let
    f(x::AbstractFloat) = 2/(1 + x^2)
    expect = π

    I, E = quadde32(f, -1, 1)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, -1, 1)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -1, 1, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, -1, 0, 1)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, -1, 0, 1)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, -1, 0, 1, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [a, b] with QuadDE
# a, b are arbitrary finite float numbers
let
    f(x::AbstractFloat) = 1/(1 + (x/2)^2)
    expect = π

    I, E = quadde32(f, -2, 2)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, -2, 2)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -2, 2, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, -2, 0, 2)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, -2, 0, 2)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, -2, 0, 2, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [0, ∞] with QuadDE
let
    f(x::AbstractFloat) = exp(-x)
    expect = 1

    I, E = quadde32(f, 0, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, 0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, 0, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, 0, 1, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, 0, 1, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, 0, 1, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [a, ∞] with QuadDE
let
    f(x::AbstractFloat) = exp(-(x - 1))
    expect = 1

    I, E = quadde32(f, 1, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, 1, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, 1, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, 1, 2, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, 1, 2, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, 1, 2, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [-∞, 0] with QuadDE
let
    f(x::AbstractFloat) = exp(x)
    expect = 1

    I, E = quadde32(f, -Inf, 0)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, -Inf, 0)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, 0, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, -Inf, 1, 0)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, -Inf, 1, 0)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, 1, 0, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [-∞, b] with QuadDE
let
    f(x::AbstractFloat) = exp(x + 1)
    expect = 1

    I, E = quadde32(f, -Inf, -1)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, -Inf, -1)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, -1, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, -Inf, -2, -1)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, -Inf, -2, -1)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, -2, -1, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test integral interval [-∞, ∞] with QuadDE
let
    f(x::AbstractFloat) = exp(-x^2)
    expect = sqrt(BigFloat(π))

    I, E = quadde32(f, -Inf, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    I, E = quadde64(f, -Inf, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)

    # Split into multiple integral intevals
    I, E = quadde32(f, -Inf, 0, Inf)
    @test I isa Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    I, E = quadde64(f, -Inf, 0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(I))*norm(I)*2

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, 0, Inf, rtol=rtol)
    @test I isa BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)*2
end


# Test anti-symmetric equivalence when switching inteval limits with QuadDE
# ∫f(x)dx in [a, b] = -∫f(x)dx in [b, a]
let
    f(x::AbstractFloat) = exp(-x^2)

    # [-1, 1]
    I1, _ = quadde32(f, -1, 1)
    I2, _ = quadde32(f, 1, -1)
    @test I1 ≈ -I2

    I1, E = quadde64(f, -1, 1)
    I2, E = quadde64(f, 1, -1)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, -1, 1)
    I2, E = quaddeBF(f, 1, -1)
    @test I1 ≈ -I2


    # [a, b] (a and b are finite numbers)
    I1, _ = quadde32(f, -2, 2)
    I2, _ = quadde32(f, 2, -2)
    @test I1 ≈ -I2

    I1, E = quadde64(f, -2, 2)
    I2, E = quadde64(f, 2, -2)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, -2, 2)
    I2, E = quaddeBF(f, 2, -2)
    @test I1 ≈ -I2


    # [0, ∞]
    I1, _ = quadde32(f, 0, Inf)
    I2, _ = quadde32(f, Inf, 0)
    @test I1 ≈ -I2

    I1, E = quadde64(f, 0, Inf)
    I2, E = quadde64(f, Inf, 0)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, 0, Inf)
    I2, E = quaddeBF(f, Inf, 0)
    @test I1 ≈ -I2


    # [a, ∞]
    I1, _ = quadde32(f, 2, Inf)
    I2, _ = quadde32(f, Inf, 2)
    @test I1 ≈ -I2

    I1, E = quadde64(f, 2, Inf)
    I2, E = quadde64(f, Inf, 2)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, 2, Inf)
    I2, E = quaddeBF(f, Inf, 2)
    @test I1 ≈ -I2


    # [-∞, 0]
    I1, _ = quadde32(f, -Inf, 0)
    I2, _ = quadde32(f, 0, -Inf)
    @test I1 ≈ -I2

    I1, E = quadde64(f, -Inf, 0)
    I2, E = quadde64(f, 0, -Inf)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, -Inf, 0)
    I2, E = quaddeBF(f, 0, -Inf)
    @test I1 ≈ -I2


    # [-∞, b]
    I1, _ = quadde32(f, -Inf, 2)
    I2, _ = quadde32(f, 2, -Inf)
    @test I1 ≈ -I2

    I1, E = quadde64(f, -Inf, 2)
    I2, E = quadde64(f, 2, -Inf)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, -Inf, 2)
    I2, E = quaddeBF(f, 2, -Inf)
    @test I1 ≈ -I2


    # [-∞, ∞]
    I1, _ = quadde32(f, -Inf, Inf)
    I2, _ = quadde32(f, Inf, -Inf)
    @test I1 ≈ -I2

    I1, E = quadde64(f, -Inf, Inf)
    I2, E = quadde64(f, Inf, -Inf)
    @test I1 ≈ -I2

    I1, E = quaddeBF(f, -Inf, Inf)
    I2, E = quaddeBF(f, Inf, -Inf)
    @test I1 ≈ -I2
end


# Test non-scalar output with QuadDE
let
    f(x::AbstractFloat) = [1/(1 + x^2), 2/(1 + x^2)]
    expect = [BigFloat(π)/2, BigFloat(π)]

    I, E = quadde32(f, -1, 1)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadde64(f, -1, 1)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -1, 1, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)


    I, E = quadde32(f, -1, 0, 1)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadde64(f, -1, 0, 1)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -1, 0, 1, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)


    f(x::AbstractFloat) = [exp(-x), 2exp(-x)]
    expect = [1, 2]

    I, E = quadde32(f, 0, Inf)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadde64(f, 0, Inf)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, 0, Inf, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)


    f(x::AbstractFloat) = [exp(-x^2), 2*exp(-x^2)]
    expect = [sqrt(BigFloat(π)), 2*sqrt(BigFloat(π))]

    I, E = quadde32(f, -Inf, Inf)
    @test eltype(I) == Float32
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    I, E = quadde64(f, -Inf, Inf)
    @test eltype(I) == Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(eltype(I)))*norm(I)

    rtol = 1e-30
    I, E = quaddeBF(f, -Inf, Inf, rtol=rtol)
    @test eltype(I) == BigFloat
    @test isapprox(I, expect, rtol=10rtol)
    @test E ≤ rtol*norm(I)
end

