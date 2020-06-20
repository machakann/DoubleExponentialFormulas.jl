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


# Various interval (-∞, ∞)
let f(x) = sin(x)/x, expect = π
    I, E = quaddeo(f, 1.0, 0.0, -Inf, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Various interval (0, ∞)
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Various interval (a, ∞)
let f(x) = sin(x)/x, expect = π/2 - BigFloat("1.851937051982466170361053370157991363345809728981154909804")
    I, E = quaddeo(f, 1.0, 0.0, π, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Various interval (-∞, 0)
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, -Inf, 0.0)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Various interval (-∞, b)
let f(x) = sin(x)/x, expect = π/2 - BigFloat("1.851937051982466170361053370157991363345809728981154909804")
    I, E = quaddeo(f, 1.0, 0.0, -Inf, -π)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Various interval (a, b)
let f(x) = sin(x)/x, expect = BigFloat("-0.43378547584983772011527320785824193420335637968613863796")
    I, E = quaddeo(f, 1.0, 0.0, π, 2π)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end
let f(x) = sin(x)/x, expect = BigFloat("0.43378547584983772011527320785824193420335637968613863796")
    I, E = quaddeo(f, 1.0, 0.0, -π, -2π)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end
let f(x) = sin(x)/x, expect = BigFloat("3.703874103964932340722106740315982726691619457962309819609")
    I, E = quaddeo(f, 1.0, 0.0, -π, π)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Corner case (a == b)
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 0.0, 0.0)
    @test I isa Float64
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 1.0, 1.0)
    @test I isa Float64
    @test I ≈ 0
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Corner case (a > b)
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, Inf, 0.0)
    @test I isa Float64
    @test I ≈ -π/2
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end


# Multiple integral intervals
let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 0.0, 1.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end

let f(x) = sin(x)/x, expect = π/2
    I, E = quaddeo(f, 1.0, 0.0, 0.0, 1.0, 2.0, Inf)
    @test I isa Float64
    @test I ≈ expect
    @test E ≤ sqrt(eps(typeof(I))*norm(I))
end
