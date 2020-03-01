module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    QuadTS,
    QuadES,
    QuadSS,
    QuadDE,
    quadde


"""
    startindex(f::Function, weights, istart::Integer)

Returns the first index i which `f(weights[i][1])` is not NaN,
where `i >= istart`.

This function is employed to avoid sampling `f(x)` with too large `abs(x)` in
trapezoidal rule. For example, the limit of `x*exp(-x^2)` may result in NaN
when `x` approaches infinity since the expression reduced to be `Inf*0` in
float point number computations.
"""
function startindex(f::Function, weights, istart::Integer)
    iend = length(weights)
    for i in istart:iend
        x, _ = @inbounds weights[i]
        y = try
            f(x)
        catch
            continue
        end
        all(.!isnan.(y)) && return i
    end
    return iend
end


@noinline function mapsum(f, A::AbstractVector, thr::Real,
                          ifirst::Integer, ilast::Integer, blksize::Int)
    if ifirst == ilast
        @inbounds a1 = A[ifirst]
        return f(a1)
    elseif ifirst + blksize > ilast
        # sequential portion
        if thr == 0
            return _mapsum_sequential(f, A, ifirst, ilast)
        else
            return _mapsum_sequential_edge(f, A, thr, ifirst, ilast)
        end
    else
        # pairwise portion
        imid = (ifirst + ilast) >> 1
        v1 = mapsum(f, A, zero(thr), ifirst, imid, blksize)
        v2 = mapsum(f, A, thr, imid+1, ilast, blksize)
        return v1 + v2
    end
end

"""
    mapsum(f, A::AbstractVector, thr::Real,
           ifirst::Integer=1, ilast::Integer=length(A))

Return the total summation of items in `A` with applying a function `f`.

NOTE: This function doesn't check `istart` and `iend`. Be careful to use.
"""
mapsum(f, A::AbstractVector, thr::Real, ifirst::Integer=1, ilast::Integer=length(A)) =
    mapsum(f, A, thr, ifirst, ilast, 512)
mapsum(f, A::AbstractVector) = mapsum(f, A, 0.0)

function _mapsum_sequential(f, A::AbstractVector, ifirst::Integer, ilast::Integer)
    @inbounds a1 = A[ifirst]
    @inbounds a2 = A[ifirst+1]
    v = f(a1) + f(a2)
    @simd for i in ifirst + 2 : ilast
        @inbounds ai = A[i]
        v = v + f(ai)
    end
    return v
end

function _mapsum_sequential_edge(f, A::AbstractVector, thr::Real,
                                 ifirst::Integer, ilast::Integer)
    @inbounds a1 = A[ilast]
    @inbounds a2 = A[ilast-1]
    dv1 = f(a1)
    dv2 = f(a2)
    normdv1 = norm(dv1)
    normdv2 = norm(dv2)
    v = dv1 + dv2
    initialcount = 3
    quitcount = initialcount
    quitcount -=
    if normdv1 < thr && normdv2 < thr
        2
    elseif normdv2 < thr
        1
    else
        0
    end
    for i in ilast - 2 : -1 : ifirst
        @inbounds ai = A[i]
        dv = f(ai)
        normdv = norm(dv)
        v = v + dv
        if norm(dv) < thr
            quitcount -= 1
        else
            quitcount = 3
        end
        quitcount < 1 && break
    end
    return v
end


"""
    estimate_error(T::Type{<:AbstractFloat}, prevI, I)

Estimate an error from the true integral(, though it may not be very accurate).

With double exponential formula, the error from the true value `I` decays
exponentially.
    ΔI(h) = I - I(h) ≈ exp(-C/h)
`I(h)` is the numerical integral with a step size `h`, `C` is a
constant depending on the integrand. If the step size is halved,
    ΔI(h/2) = I - I(h/2) ≈ exp(-2C/h) ≈ ( ΔI(h) )²,
the significant digits gets almost twice. Therefore, `I(h/2)` should be
considerably closer to the true value than `I(h)`.
    ΔI(h/2) ≈ ( ΔI(h) )² ≈ ( I - I(h) )² ≈ ( I(h/2) - I(h) )²
Introduce a magnification coefficient `M` for safety.
    ΔI(h/2) ≈ M*( I(h/2) - I(h) )²
Furthermore, the floating point numbers generally have a finite
significant digits; `ΔI(h/2)` would asymptotically get closer to `I*ε`
at minimum, rather than 0.
    ΔI(h/2) ≈ M*( I(h/2) - I(h) )² + I*ε
            ≈ M*( I(h/2) - I(h) )² + I(h/2)*ε

FIXME: The logical ground of the safety factor is very weak. Any better way?
"""
function estimate_error(T::Type{<:AbstractFloat}, prevI, I)
    ε = eps(T)
    M = 20
    return M*norm(I - prevI)^2 + norm(I)*ε
end


include("quadts.jl")
include("quades.jl")
include("quadss.jl")
include("quadde.jl")


end # module
