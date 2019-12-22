using DoubleExponentialFormulas
using LinearAlgebra: norm
using Test

# Cited from J. Comput. Appl. Math. 127 (2001) 287-296
let
    f(x) = 1/((2 - x)*(1 - x)^(1/4)*(1 + x)^(3/4))
    expect = 1.9490

    I, E = quadde(f, -1, 0, 1, atol=1e-4)
    @test eltype(I) == Float64
    @test abs(I - expect) ≤ 1e-4
    @test E ≤ 1e-4
end

