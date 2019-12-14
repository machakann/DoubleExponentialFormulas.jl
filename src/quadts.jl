using LinearAlgebra: norm


struct QuadTSWeightTable{T<:AbstractFloat} <: AbstractVector{Tuple{T,T}}
    table::Vector{Tuple{T,T}}
end
Base.size(wt::QuadTSWeightTable) = size(wt.table)
Base.getindex(wt::QuadTSWeightTable, i::Int) = getindex(wt.table, i)


struct QuadTS{T<:AbstractFloat,N}
    h0::T
    origin::Tuple{T,T}
    tables::NTuple{N,QuadTSWeightTable{T}}
end
function QuadTS(T::Type{<:AbstractFloat}; maxlevel::Integer=10, h0::Real=one(T)/h0inv)
    @assert maxlevel > 0
    t0 = zero(T)
    tables, origin = generate_tables(QuadTSWeightTable, maxlevel, T(h0))
    QuadTS{T,maxlevel+1}(T(h0), origin, tables)
end

function (q::QuadTS{T,N})(f::Function; atol::Real=zero(T),
                          rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    x0, w0 = q.origin
    I = f(x0)*w0
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    sample(t) = f(t[1])*t[2] + f(-t[1])*t[2]
    for level in 0:(N-1)
        I += sum_pairwise(sample, q.tables[level+1])
        h = h0/2^level
        prevIh = Ih
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 0 && break
    end
    Ih, E
end
function (q::QuadTS{T,N})(f::Function, intervals::Tuple{Real,Real,Vararg{Real}};
                          atol::Real=zero(T), rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    I = init(f, q, intervals)
    h0 = q.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    for level in 0:(N-1)
        # FIXME: This anonymous function is not necessarily in principle, but
        #        memory usage increases and execution time gets longer if it is
        #        omitted. I don't know why. It should be equal to
        #        `increment(f, x0, w0, intervals)`
        I += increment(x -> f(x), q.tables[level+1], intervals)
        h = h0/2^level
        prevIh = Ih
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 0 && break
    end
    Ih, E
end
(q::QuadTS{T,N})(f::Function, a::Real, b::Real; kwargs...) where {T<:AbstractFloat,N} = q(f, (a, b); kwargs...)
(q::QuadTS{T,N})(f::Function, a::Real, b::Real, c::Real...; kwargs...) where {T<:AbstractFloat,N} = q(f, (a, b, c...); kwargs...)


function init(f::Function, q::QuadTS{T,N}, a::Real, b::Real) where {T<:AbstractFloat,N}
    x0, w0 = q.origin
    if a > b
        return -init(f, q, b, a)
    end

    if a == b
        zero(f(a)), zero(T)
    end

    if a == -1 && b == 1
        return f(x0)*w0
    end

    _a = T(a)
    _b = T(b)
    s = _b + _a
    t = _b - _a
    f′(u) = f((s + t*u)/2)
    return f′(x0)*w0*t/2
end
function init(f::Function, q::QuadTS{T,N}, intervals) where {T<:AbstractFloat,N}
    n = length(intervals)
    a = intervals[1]
    b = intervals[2]
    I = init(f, q, a, b)
    for i in 3:n
        a = intervals[i-1]
        b = intervals[i]
        I += init(f, q, a, b)
    end
    I
end


function increment(f::Function, table::QuadTSWeightTable{T}, a::Real, b::Real) where {T<:AbstractFloat}
    if a > b
        -increment(f, table, b, a)
    end

    if a == b
        zero(f(a))
    end

    sampling_function(f::Function) = xw -> begin
        x, w = xw
        f(x)*w + f(-x)*w
    end

    if a == -1 && b == 1
        sum_pairwise(sampling_function(f), table)
    else
        _a = T(a)
        _b = T(b)
        s = _b + _a
        t = _b - _a
        f′(u) = f((s + t*u)/2)
        I = sum_pairwise(sampling_function(f′), table)
        I*t/2
    end
end
function increment(f::Function, table::QuadTSWeightTable{T}, intervals) where {T<:AbstractFloat}
    n = length(intervals)
    a = intervals[1]
    b = intervals[2]
    I = increment(f, table, a, b)
    for i in 3:n
        a = intervals[i-1]
        b = intervals[i]
        I += increment(f, table, a, b)
    end
    I
end


function generate_tables(::Type{QuadTSWeightTable}, maxlevel::Integer, h0::T) where {T<:AbstractFloat}
    ϕ(t) = tanh(sinh(t)*π/2)
    ϕ′(t) = (cosh(t)*π/2)/cosh(sinh(t)*π/2)^2
    tables = Vector{QuadTSWeightTable}(undef, maxlevel+1)
    for level in 0:maxlevel
        table = Tuple{T,T}[]
        h = h0/2^level
        k = 1
        step = level == 0 ? 1 : 2
        while true
            t = k*h
            xk = ϕ(t)
            1 - xk ≤ eps(T) && break
            wk = ϕ′(t)
            wk ≤ floatmin(T) && break
            push!(table, (xk, wk))
            k += step
        end
        reverse!(table)
        tables[level+1] = QuadTSWeightTable{T}(table)
    end

    x0 = ϕ(zero(T))
    w0 = ϕ′(zero(T))
    origin = (x0, w0)
    Tuple(tables), origin
end
