# DoubleExponentialFormulas.jl

This package provides functions for one-dimensional numerical integration using the double exponential formula [^1][^2] also known as the [tanh-sinh quadrature](https://en.wikipedia.org/wiki/Tanh-sinh_quadrature) and its variants.



## Theory

The double exponential formulas transform an integrand into another form well-suited to the trapezoidal rule. For example, the tanh-sinh quadrature converts an integration of ``f(x)`` on ``x`` over ``[-1, 1]`` into an equivalent integration on ``t`` over ``[-∞, ∞]``,

```math
\int_{-1}^{1} f(x) dx = \int_{-\infty}^{\infty} f\left(x(t)\right) \frac{dx}{dt} dt = \int_{-\infty}^{\infty} f\left(x(t)\right)w(t) dt
```

where

```math
x(t) = \tanh \left( \frac{π}{2} \sinh t \right) \\
w(t) = \frac{\frac{π}{2} \cosh t}{\cosh^{2}\left(\frac{π}{2} \sinh t \right)}.
```

Since ``f\left(x(t)\right)w(t)`` rapidly decays with ``|t|`` increasing, the transformed integral is highly insensitive to both edges of the integral interval. Therefore, the infinite interval can be cut off at a certain (finite) point, and the numerical integration just works even if the integrand ``f(x)`` has singularity or discontinuity at the endpoints. For instance, generally, it is difficult to estimate the following integral because of the singularity at the endpoints.

```math
\int_{-1}^{1} f(x) dx = \int_{-1}^{1} \frac{dx}{(2-x)(1-x)^{1/4}(1+x)^{3/4}}
```

However, the transformed integrand ``f\left(x(t)\right)w(t)`` quickly drops and almost vanishes at ``|t| \approx 4``. The endpoint singularity has gone, and the range ``|t| > 4`` doesn't have any significant contribution.

![Change of variable](https://imgur.com/0hJKg50.png)

This transformation accelerates the convergence and enhances the accuracy in a numerical integration; typically, the trapezoidal rule is used. The last integral is approximated to a weighted sum with a step size ``h``,

```math
\int_{-\infty}^{\infty} f\left(x(t)\right)w(t) dt \approx h\sum_{k = -\infty}^{\infty} f(x_k)w_k
```

where ``x_k = x(kh)`` and ``w_k = w(kh)``. The infinite summation is terminated if ``f(x_k)w_k`` gets small enough.

Another point is that the ``x_k`` and ``w_k`` is independent of integrands. Therefore, the table of these numbers can be pre-calculated and is re-usable. This mechanism significantly cut down the computational time.


### Tanh-sinh quadrature

The tanh-sinh quadrature covers the numerical integrations over the range ``[-1, 1]`` using the following abscissas and weights.

```math
x(t) = \tanh \left( \frac{π}{2} \sinh t \right) \\
w(t) = \frac{\frac{π}{2} \cosh t}{\cosh^{2}\left(\frac{π}{2} \sinh t \right)}
```

Furthermore, the additional change of variables extends to the integration with arbitrary finite intervals ``[a, b]``.

```math
x(u) = \frac{b + a}{2} + \frac{b - a}{2}u
```

The ``x(u)`` changes from ``a`` to ``b`` when ``u`` changes from ``-1`` to ``1``. Hence,

```math
\int_{a}^{b} f(x) dx = \frac{b - a}{2} \int_{-1}^{1} f\left(x(u)\right) du = \frac{b - a}{2} \int_{-\infty}^{\infty} f\left(x\left(u(t)\right)\right)w(t) dt.
```


### Exp-sinh quadrature

The exp-sinh quadrature is available for the numerical integration over the half-infinite interval ``[0, \infty)``.

```math
x(t) = \exp \left( \frac{π}{2} \sinh t \right) \\
w(t) = \frac{π}{2} \cosh t \exp\left(\frac{π}{2} \sinh t \right)
```

Finite shift of the integral interval ``x(u) = u + a`` immediately enables integrating over ``[a, \infty)``.

```math
\int_{a}^{\infty} f(x) dx = \int_{0}^{\infty} f\left(x(u)\right) du = \int_{-\infty}^{\infty} f\left(x\left(u(t)\right)\right)w(t) dt
```

Similarly, reversing the abscissas around the origin ``x(u) = -u`` converts the integration on ``x`` over ``(-\infty, b]`` into an equivalent integral on ``u`` over ``[-b, \infty)``.

```math
\int_{-\infty}^{b} f(x) dx = \int_{-b}^{\infty} f\left(x(u)\right) du
```

Note that ``\int_{a}^{b} f(x) dx = -\int_{b}^{a} f(x) dx``. In conclusion, the exp-sinh quadrature covers arbitrary half-infinite integral intervals, ``[a, \infty)`` and ``(-\infty, b]``.



### Sinh-sinh quadrature

The sinh-sinh quadrature covers the infinite integral interval ``(-\infty, \infty)``.

```math
x(t) = \sinh \left( \frac{π}{2} \sinh t \right) \\
w(t) = \frac{π}{2} \cosh t \cosh\left(\frac{π}{2} \sinh t \right)
```

The transformation doesn't change the interval. However, ``f\left(x(t)\right)w(t)`` decays quickly in the ``t`` space, as far as the integral isn't divergent, to make the numerical integration quite efficient.

```math
\int_{-\infty}^{\infty} f(x) dx = \int_{-\infty}^{\infty} f\left(x(t)\right)w(t) dt
```



## Instllation

Press `]` on a Julia REPL to enter the Pkg mode, and run the following command.

```
add https://github.com/machakann/DoubleExponentialFormulas.jl.git
```



## Usage

### Handy interface in `Float64` precision

```
    I, E = quadde(f::Function, a::Real, b::Real, c::Real...;
                  atol::Real=zero(Float64),
                  rtol::Real=atol>0 ? zero(Float64) : sqrt(eps(Float64)))
```

The `quadde` function provides a handy way to integrate a function ``f(x)`` in an arbitrary interval.

```julia
using DoubleExponentialFormulas
using LinearAlgebra: norm

f(x) = 1/(1 + x^2)
I, E = quadde(f, -1, 1)

I ≈ π/2                   # true
E ≤ sqrt(eps(I))*norm(I)  # true
```

The above example computes ``\int_{-1}^{1} \frac{1}{1 + x^2} dx``. The `I` is the obtained integral value and the `E` is an estimated numerical error. The `E` is not exactly equal to the difference from the true value. However, one can expect that the integral value `I` is converged if `E <= max(atol, rtol*norm(I))` is true. Otherwise, the obtained `I` would be unreliable; the number of repetitions exceeds the `maxlevel` before converged.

Half-infinite intervals, ``[a, \infty)`` and ``(-\infty, b]``, and the infinite interval, ``(\infty, -\infty)``, are also valid as far as the integral is convergent.

```julia
# Computes ∫ 1/(1+x^2) dx in [0, ∞)
I, E = quadde(x -> 1/(1 + x^2), 0, Inf)
I ≈ π/2    # true

# Computes ∫ 1/(1+x^2) dx in (-∞, ∞)
I, E = quadde(x -> 1/(1 + x^2), -Inf, Inf)
I ≈ π      # true
```

![Fig.1-3](https://imgur.com/id5rPIP.png)

Optionally, one can subdivide the integral interval [a, b, c, ...], which returns ``\int_a^b f(x) dx + \int_b^c f(x) dx + \cdots``.  It is worth noting that discontinuity or singularity is allowed at the endpoints.

```julia
# Computes ∫ 1/sqrt(|x|) dx in (-∞, ∞)
# The integrand has a singular point at x = 0
I, E = quadde(x -> 1/sqrt(abs(x)), -1, 0, 1)
I ≈ 4    # true
```

![Fig.4](https://imgur.com/ckPlHsi.png)


### Optimized numerical integrators

The type `QuadDE` represents the pre-calculated table of ``x_k`` and ``w_k``. If one needs an optimized table, for example with a smaller step size ``h`` or with a better precision using `BigFloat`, `QuadDE` will provides the functionality.

```
qde = QuadDE(BigFloat; h0=one(BigFloat), maxlevel=12)
qde(x -> 2/(1 + x^2), -1,  1)
```

User can specify the required precision as a type (`T<:AbstractFloat`), the starting step size `h0` and the maximum number of repetition `maxlevel`. The `h0` and `maxlevel` shown above are the default values, so it can be omitted. `QuadDE` instance is an callable object which has the same interface of `quadde`, actually `quadde` is an alias to `QuadDE(Float64)(...)` with a pre-calculated instance.

`QuadTS` tries to calculate integral values `maxlevel` times at a maximum; the step size of a trapezoid is started from `h0` and is halved in each following repetition for finer accuracy. The repetition is terminated when the difference from the previous estimation gets smaller than a certain threshold.  The threshold is determined by the runtime parameters, `atol` or `rtol`.

Using smaller `h0` may help if the integrand `f(x)` includes fine structure, such as spikes, in the integral interval. However, the subdivision of the interval would be more effective in many cases. Try subdivision first, and then think of an optimized integrator.



[^1]: Takahasi, H.; Mori, M. Double Exponential Formulas for Numerical Integration. Publ. Res. Inst. Math. Sci. 1973, 9 (3), 721–741.  https://doi.org/10.2977/prims/1195192451.

[^2]: Mori, M.; Sugihara, M. The Double-Exponential Transformation in Numerical Analysis. J. Comput. Appl. Math. 2001, 127 (1–2), 287–296.  https://doi.org/10.1016/S0377-0427(00)00501-X.
