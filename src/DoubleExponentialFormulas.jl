module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    quadde,
    quadts,
    quades,
    quadss

const h0inv = 8
const nmax = 1_000_000

function quadde end

function quadts(f::Function, T::Type{<:AbstractFloat}=Float64;
                rtol::Real=sqrt(eps(float(T))), atol::Real=0,
                maxeval::Integer=10_000_000)
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    deformula(f, ϕ, ϕ′, T, rtol, atol, maxeval)
end

function quades end

function quadss end


# Compute double exponential formula iteratively until satisfying the given tolerance
function deformula(f::Function, ϕ::Function, ϕ′::Function, T::Type{<:AbstractFloat},
                   rtol::Real, atol::Real, maxeval::Integer)
    I0 = f(ϕ(zero(T)))*ϕ′(zero(T))
    h0 = one(T)/h0inv
    I, n0 = trapez(f, ϕ, ϕ′, I0, h0, 1, 1, nmax, rtol, atol)
    h = h0
    Ih = I*h
    eval = 0
    p = 1
    while eval ≤ maxeval
        prevIh = Ih
        h = h0/2^p
        n = n0*2^p
        I, k, Ih, tol = trapez(f, ϕ, ϕ′, I, h, 1, 2, n, rtol, atol)
        eval += k + 1
        E = norm(prevIh - Ih)
        !(E > tol) && break
        p += 1
    end
    h*I
end


# Compute the trapezoidal integration with translated integrals and weights
# Note: ϕ(-t) == -ϕ(t), w(-t) == w(t)
function trapez(f::Function, ϕ::Function, ϕ′::Function, I, h::Real,
                kstart::Int, kinc::Int, kend::Int, rtol::Real, atol::Real)
    tol = zero(float(eltype(I)))
    Ih = zero(I)
    lastk = 0
    for k in kstart:kinc:kend
        lastk = k
        x = ϕ(k*h)
        w = ϕ′(k*h)
        dI1 = f(x)*w
        dI2 = f(-x)*w
        I += dI1
        I += dI2
        Ih = I*h
        tol = max(norm(Ih)*rtol, atol)
        !(norm(dI1*h) + norm(dI2*h) > tol) && break
    end
    I, lastk, Ih, tol
end


end # module
