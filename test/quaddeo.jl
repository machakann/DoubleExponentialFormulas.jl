using DoubleExponentialFormulas: quaddeo
using LinearAlgebra
using Test

# Test functions in
# Ooura, T., Mori, M., 1991.
# The double exponential formula for oscillatory functions over the half infinite interval.
# Journal of Computational and Applied Mathematics 38, 353–360.
# https://doi.org/10.1016/0377-0427(91)90181-I

# 1
let f(x) = exp(-x)*cos(x), expect = 1/2
    I, E = quaddeo(f, 1.0, π/2, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 2
let f(x) = x*sin(x)/(1 + x^2), expect = π/(2*exp(1))
    I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 3
let f(x) = cos(x)/(1 + x^2), expect = π/(2*exp(1))
    I, E = quaddeo(f, 1.0, π/2, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 4
let f(x) = log((x^2 + 4)/(x^2 + 1))*cos(x), expect = (exp(-1) - exp(-2))*π
    I, E = quaddeo(f, 1.0, π/2, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 5 (Dirichlet integral)
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 6
let f(x) = 1/sqrt(x)*sin(x), expect = sqrt(π/2)
    I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 7
let f(x) = cos(x)/sqrt(x), expect = sqrt(π/2)
    I, E = quaddeo(f, 1.0, π/2, 0.0, Inf)
    @test I isa Float64
    @test_broken I ≈ expect
    @test abs(I - expect) < 1e-6
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

# 8
let f(x) = sin(x)*log(x), expect = -Base.MathConstants.eulergamma
    I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end
