using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

quadde32 = QuadDE(Float32)
quadde64 = QuadDE(Float64)
quaddeBF = QuadDE(BigFloat)

# Test ∫f(x)dx = 0 in [a, b] if a == b
let
    a = rand(Int)

    # Odd function
    I, E = quadde32(x -> x, a, a)
    @test I isa Float32
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quadde64(x -> x, a, a)
    @test I isa Float64
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quaddeBF(x -> x, a, a)
    @test I isa BigFloat
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)


    # Even function
    I, E = quadde32(x -> x^2, a, a)
    @test I isa Float32
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quadde64(x -> x^2, a, a)
    @test I isa Float64
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)

    I, E = quaddeBF(x -> x^2, a, a)
    @test I isa BigFloat
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I)))*norm(I)
end


# Test the anti-symmetric equivalence when switching inteval limits with QuadDE
# ∫f(x)dx in [a, b] = -∫f(x)dx in [b, a]
let
    f(x::AbstractFloat) = exp(-x^2)

    # [-1, 1]
    I1, E1 = quadde32(f, -1, 1)
    I2, E2 = quadde32(f, 1, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, -1, 1)
    I2, E2 = quadde64(f, 1, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, -1, 1)
    I2, E2 = quaddeBF(f, 1, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [a, b] (a and b are finite numbers)
    I1, E1 = quadde32(f, -2, 2)
    I2, E2 = quadde32(f, 2, -2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, -2, 2)
    I2, E2 = quadde64(f, 2, -2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, -2, 2)
    I2, E2 = quaddeBF(f, 2, -2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [0, ∞]
    I1, E1 = quadde32(f, 0, Inf)
    I2, E2 = quadde32(f, Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, 0, Inf)
    I2, E2 = quadde64(f, Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, 0, Inf)
    I2, E2 = quaddeBF(f, Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [a, ∞]
    I1, E1 = quadde32(f, 2, Inf)
    I2, E2 = quadde32(f, Inf, 2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, 2, Inf)
    I2, E2 = quadde64(f, Inf, 2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, 2, Inf)
    I2, E2 = quaddeBF(f, Inf, 2)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [-∞, 0]
    I1, E1 = quadde32(f, -Inf, 0)
    I2, E2 = quadde32(f, 0, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, -Inf, 0)
    I2, E2 = quadde64(f, 0, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, -Inf, 0)
    I2, E2 = quaddeBF(f, 0, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [-∞, b]
    I1, E1 = quadde32(f, -Inf, 2)
    I2, E2 = quadde32(f, 2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, -Inf, 2)
    I2, E2 = quadde64(f, 2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, -Inf, 2)
    I2, E2 = quaddeBF(f, 2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # [-∞, ∞]
    I1, E1 = quadde32(f, -Inf, Inf)
    I2, E2 = quadde32(f, Inf, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f, -Inf, Inf)
    I2, E2 = quadde64(f, Inf, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f, -Inf, Inf)
    I2, E2 = quaddeBF(f, Inf, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)
end


# Parity check for even function
let
    # Even function
    f(x::AbstractFloat) = exp(-x^2)

    # ∫f(x)dx in [a, b] == ∫f(x)dx in [-b, -a] if f(x) is an even function
    I1, E1 = quadde32(f,  0, 1)
    I2, E2 = quadde32(f, -1, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  0, 1)
    I2, E2 = quadde64(f, -1, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  0, 1)
    I2, E2 = quaddeBF(f, -1, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    I1, E1 = quadde32(f,  1,  2)
    I2, E2 = quadde32(f, -2, -1)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  1,  2)
    I2, E2 = quadde64(f, -2, -1)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  1,  2)
    I2, E2 = quaddeBF(f, -2, -1)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # ∫f(x)dx in [a, ∞] == ∫f(x)dx in [-∞, -a] if f(x) is an even function
    I1, E1 = quadde32(f,  0, Inf)
    I2, E2 = quadde32(f, -Inf, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  0, Inf)
    I2, E2 = quadde64(f, -Inf, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  0, Inf)
    I2, E2 = quaddeBF(f, -Inf, 0)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    I1, E1 = quadde32(f,  Inf,  2)
    I2, E2 = quadde32(f, -2, -Inf)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  Inf,  2)
    I2, E2 = quadde64(f, -2, -Inf)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  Inf,  2)
    I2, E2 = quaddeBF(f, -2, -Inf)
    @test I1 ≈ I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)
end


# Parity check for odd function
let
    # Odd function
    f(x::AbstractFloat) = x*exp(-x^2)

    # ∫f(x)dx in [a, b] == -∫f(x)dx in [-b, -a] if f(x) is an odd function
    I1, E1 = quadde32(f,  0, 1)
    I2, E2 = quadde32(f, -1, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  0, 1)
    I2, E2 = quadde64(f, -1, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  0, 1)
    I2, E2 = quaddeBF(f, -1, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    I1, E1 = quadde32(f,  1,  2)
    I2, E2 = quadde32(f, -2, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  1,  2)
    I2, E2 = quadde64(f, -2, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  1,  2)
    I2, E2 = quaddeBF(f, -2, -1)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    # ∫f(x)dx in [a, ∞] == ∫f(x)dx in [-∞, -a] if f(x) is an odd function
    I1, E1 = quadde32(f,  0, Inf)
    I2, E2 = quadde32(f, -Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  0, Inf)
    I2, E2 = quadde64(f, -Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  0, Inf)
    I2, E2 = quaddeBF(f, -Inf, 0)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    I1, E1 = quadde32(f,  Inf,  2)
    I2, E2 = quadde32(f, -2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quadde64(f,  Inf,  2)
    I2, E2 = quadde64(f, -2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I1, E1 = quaddeBF(f,  Inf,  2)
    I2, E2 = quaddeBF(f, -2, -Inf)
    @test I1 ≈ -I2
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)
end


# Test the equivalence w/o splitting integral interval
# ∫f(x)dx in [a, b] = ∫f(x)dx in [a, c] + ∫f(x)dx in [c, b]  where a ≤ c ≤ b
let
    f(x::AbstractFloat) = exp(-x^2)

    I,  E  = quadde32(f, -1, 1)
    I1, E1 = quadde32(f, -1, 0)
    I2, E2 = quadde32(f, 0, 1)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I,  E  = quadde64(f, -1, 1)
    I1, E1 = quadde64(f, -1, 0)
    I2, E2 = quadde64(f, 0, 1)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I,  E  = quaddeBF(f, -1, 1)
    I1, E1 = quaddeBF(f, -1, 0)
    I2, E2 = quaddeBF(f, 0, 1)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)


    I,  E  = quadde32(f, -Inf, Inf)
    I1, E1 = quadde32(f, -Inf, 0)
    I2, E2 = quadde32(f, 0, Inf)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I,  E  = quadde64(f, -Inf, Inf)
    I1, E1 = quadde64(f, -Inf, 0)
    I2, E2 = quadde64(f, 0, Inf)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)

    I,  E  = quaddeBF(f, -Inf, Inf)
    I1, E1 = quaddeBF(f, -Inf, 0)
    I2, E2 = quaddeBF(f, 0, Inf)
    @test I ≈ I1 + I2
    @test E  ≤ sqrt(eps(typeof(I)))*norm(I)
    @test E1 ≤ sqrt(eps(typeof(I1)))*norm(I1)
    @test E2 ≤ sqrt(eps(typeof(I2)))*norm(I2)
end
