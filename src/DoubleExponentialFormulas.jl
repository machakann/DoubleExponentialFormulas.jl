module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    quadde,
    quadts,
    quades,
    quadss,
    QuadTS

const h0inv = 8


function quadde end


struct QuadTS{T<:AbstractFloat,N}
    h0::T
    x0::T
    w0::T
    table::NTuple{N,Vector{Tuple{T,T}}}
end
function QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    t0 = zero(T)
    table = generate_table(ϕ, ϕ′, maxlevel, T(h0))
    QuadTS{T,maxlevel+1}(T(h0), ϕ(t0), ϕ′(t0), Tuple(table))
end

function (qts::QuadTS{T,N})(f::Function; rtol::Real=sqrt(eps(T)),
                            atol::Real=zero(T)) where {T<:AbstractFloat,N}
    x0 = qts.x0
    w0 = qts.w0
    h0 = qts.h0
    I0 = f(x0)*w0
    I, Ih = trapez(f, qts.table[1], I0, h0, rtol, atol)
    E = zero(eltype(Ih))
    for level in 1:(N-1)
        prevIh = Ih
        h = h0/2^level
        I, Ih, tol = trapez(f, qts.table[level+1], I, h, rtol, atol)
        E = norm(prevIh - Ih)
        !(E > tol) && break
    end
    Ih, E
end

function quadts(f::Function, T::Type{<:AbstractFloat}=Float64;
                rtol::Real=sqrt(eps(float(T))), atol::Real=0, maxlevel::Integer=10)
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    deformula(f, ϕ, ϕ′, T, rtol, atol, maxlevel)
end

function quades end

function quadss end


# Compute double exponential formula iteratively until satisfying the given tolerance
function deformula(f::Function, ϕ::Function, ϕ′::Function, T::Type{<:AbstractFloat},
                   rtol::Real, atol::Real, maxlevel::Integer)
    I0 = f(ϕ(zero(T)))*ϕ′(zero(T))
    h0 = one(T)/h0inv
    nmax = 10^6
    I, n0, Ih = trapez(f, ϕ, ϕ′, I0, h0, 1, 1, nmax, rtol, atol)
    E = zero(eltype(Ih))
    eval = 0
    p = 1
    while p ≤ maxlevel
        prevIh = Ih
        h = h0/2^p
        n = n0*2^p
        I, _, Ih, tol = trapez(f, ϕ, ϕ′, I, h, 1, 2, n, rtol, atol)
        E = norm(prevIh - Ih)
        !(E > tol) && break
        p += 1
    end
    Ih, E
end


# Compute the trapezoidal integration with translated integrals and weights
# Note: ϕ(-t) == -ϕ(t), w(-t) == w(t)
function trapez(f::Function, ϕ::Function, ϕ′::Function, I, h::Real,
                kstart::Int, kinc::Int, kend::Int, rtol::Real, atol::Real)
    T = float(eltype(I))
    tol = zero(T)
    Ih = zero(I)
    lastk = 0
    for k in kstart:kinc:kend
        lastk = k
        x = ϕ(k*h)
        1 - x ≤ eps(T) && break
        w = ϕ′(k*h)
        w ≤ floatmin(T) && break
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

function trapez(f::Function, table::Vector{Tuple{T,T}}, I, h::T,
                rtol::Real, atol::Real) where {T<:AbstractFloat}
    tol = zero(float(eltype(I)))
    Ih = zero(I)
    for (x, w) in table
        dI1 = f(x)*w
        dI2 = f(-x)*w
        I += dI1
        I += dI2
        Ih = I*h
        tol = max(norm(Ih)*rtol, atol)
        !(norm(dI1*h) + norm(dI2*h) > tol) && break
    end
    I, Ih, tol
end


function generate_table(ϕ::Function, ϕ′::Function, maxlevel::Integer,
                        h0::T) where {T<:AbstractFloat}
    table = Vector{Vector{Tuple{T,T}}}(undef, maxlevel+1)
    for level in 0:maxlevel
        table[level+1] = weights(ϕ, ϕ′, level, h0)
    end
    table
end


function weights(ϕ::Function, ϕ′::Function, level::Integer,
                 h0::T) where {T<:AbstractFloat}
    @assert level ≥ 0
    table = Tuple{T,T}[]
    h = h0/2^level
    k = 1
    step = level == 0 ? 1 : 2
    while true
        xk = ϕ(k*h)
        1 - xk ≤ eps(T) && break
        wk = ϕ′(k*h)
        wk ≤ floatmin(T) && break
        push!(table, (xk, wk))
        k += step
    end
    table
end


end # module
