# DoubleExponentialFormulas

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://machakann.github.io/DoubleExponentialFormulas.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://machakann.github.io/DoubleExponentialFormulas.jl/dev)
[![Build Status](https://travis-ci.com/machakann/DoubleExponentialFormulas.jl.svg?branch=master)](https://travis-ci.com/machakann/DoubleExponentialFormulas.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/machakann/DoubleExponentialFormulas.jl?svg=true)](https://ci.appveyor.com/project/machakann/DoubleExponentialFormulas-jl)
[![Codecov](https://codecov.io/gh/machakann/DoubleExponentialFormulas.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/machakann/DoubleExponentialFormulas.jl)
[![Coveralls](https://coveralls.io/repos/github/machakann/DoubleExponentialFormulas.jl/badge.svg?branch=master)](https://coveralls.io/github/machakann/DoubleExponentialFormulas.jl?branch=master)



## Introduction

This package provides functions for one-dimensional numerical integration using the double exponential formula [1,2] also known as the [tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) and its variants.



## Instllation

Press `]` on a Julia REPL to enter the Pkg mode and run the following command.

```
add https://github.com/machakann/DoubleExponentialFormulas.jl.git
```



## Usage

### Handy interface in Float64 precision

```
    I, E = quadde(f::Function, a::Real, b::Real, c::Real...;
                  atol::Real=zero(Float64),
                  rtol::Real=atol>0 ? zero(Float64) : sqrt(eps(Float64)))
```

The `quadde` function provides a handy way to integrate a function `f(x)` over an arbitrary interval.

```julia
using DoubleExponentialFormulas
using LinearAlgebra: norm

f(x) = 1/(1 + x^2)
I, E = quadde(f, -1, 1)

I ≈ π/2    # true
E ≤ sqrt(eps(Float64))*norm(I)  # true
```

The above example computes `∫ 1/(1+x^2) dx in [-1, 1]`. The `I` is the obtained integral value and the `E` is an estimated numerical error. The `E` is not exactly equal to the difference from the true value. However, one can expect that the integral value `I` is converged if `E <= max(atol, rtol*norm(I))` is true. Otherwise, the obtained `I` would be unreliable; the number of repetitions exceeds the `maxlevel` before converged.

Half-infinite intervals and the infinite interval are also valid, as far as the integral is convergent.

```julia
# Computes ∫ 1/(1+x^2) dx in [0, ∞)
I, E = quadde(x -> 1/(1 + x^2), 0, Inf)
I ≈ π/2    # true

# Computes ∫ 1/(1+x^2) dx in (-∞, ∞)
I, E = quadde(x -> 1/(1 + x^2), -Inf, Inf)
I ≈ π      # true
```

Optionally, one can divide the integral interval [a, b, c, ...], which returns `∫f(x)dx in [a, b] + ∫f(x)dx in [b, c] + ⋯`.  It is worth noting that discontinuity or singularity is allowed at the endpoints.

```julia
# Computes ∫ 1/sqrt(|x|) dx in (-∞, ∞)
# The integrand has a singular point at x = 0
I, E = quadde(x -> 1/sqrt(abs(x)), -1, 0, 1)
I ≈ 4    # true
```


### Optimized numerical integrators

User can get an optimized integrators, for example, for better accuracy; `QuadDE` will provides the functionality.

```
qde = QuadDE(BigFloat; h0=one(BigFloat)/8, maxlevel=10)
qde(x -> 2/(1 + x^2), -1,  1)
```

User can specify the required precision as a type (`T<:AbstractFloat`), the starting step size `h0` and the maximum number of repetition `maxlevel`. The `h0` and `maxlevel` shown above are the default values, so it can be omitted. `QuadDE` instance is an callable object which has the same interface of `quadde`, actually `quadde` is an alias to `QuadDE(Float64)(...)` with a precalculated instance.

`QuadDE` tries to calculate integral values `maxlevel` times at a maximum; the step size of a trapezoid is started from `h0` and is halved in each following repetition for finer accuracy. The repetition is terminated when the difference from the previous estimation gets smaller than a certain threshold.  The threshold is determined by the runtime parameters, `atol` or `rtol`.

Using smaller `h0` may help if the integrand `f(x)` includes fine structure, such as spikes, in the integral interval. However, it seems that the subdivision of the interval would be more effective in many cases. Try subdivision first, and then think of an optimized integrator.

See [documentation](https://machakann.github.io/DoubleExponentialFormulas.jl/stable) for more details.


### Numerical integrator for decaying oscillatory integrands

```
    quaddeo(f::Function, ω::Real, θ::Real, a::Real, b::Real;
            h0::Real=one(ω)/5, maxlevel::Integer=12,
            atol::Real=zero(ω),
            rtol::Real=atol>0 ? zero(atol) : sqrt(eps(typeof(atol))))
```

The `quaddeo` function is specialized for the decaying oscillatory integrands, [3-5]

    f(x) = g(x)sin(ωx + θ),

where `g(x)` is a decaying algebraic function. `ω` and `θ` are the frequency and the phase of the oscillatory part of the integrand. If the oscillatory part is `sin(ωx)`, then `θ = 0.0`; if it is `cos(ωx)` instead, then `θ = π/(2ω)`.

```julia
using DoubleExponentialFormulas
using LinearAlgebra: norm

f(x) = sin(x)/x;
I, E = quaddeo(f, 1.0, 0.0, 0.0, Inf);
I ≈ π/2  # true
```


## References


1. Takahasi, H.; Mori, M. Double Exponential Formulas for Numerical Integration. *Publ. Res. Inst. Math. Sci.* **1973,** *9 (3),* 721–741. [10.2977/prims/1195192451](https://doi.org/10.2977/prims/1195192451).

1. Mori, M.; Sugihara, M. The Double-Exponential Transformation in Numerical Analysis. *J. Comput. Appl. Math.* **2001,** *127 (1–2),* 287–296. [10.1016/S0377-0427(00)00501-X](https://doi.org/10.1016/S0377-0427(00)00501-X).

1. Ooura, T.; Mori, M. The double exponential formula for oscillatory functions over the half infinite interval. *J. Comput. Appl. Math.* **1991,** *38,* 353–360. [10.1016/0377-0427(91)90181-I](https://doi.org/10.1016/0377-0427(91)90181-I)

1. [http://www.kurims.kyoto-u.ac.jp/~ooura/intde-j.html](http://www.kurims.kyoto-u.ac.jp/~ooura/intde-j.html)

1. [http://www.kurims.kyoto-u.ac.jp/~ooura/intdefaq-j.html](http://www.kurims.kyoto-u.ac.jp/~ooura/intdefaq-j.html)
