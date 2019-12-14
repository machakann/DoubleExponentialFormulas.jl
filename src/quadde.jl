using LinearAlgebra: norm


struct QuadDE{T<:AbstractFloat,N}
    qts::QuadTS{T,N}
    qes::QuadES{T,N}
    qss::QuadSS{T,N}
end
function QuadDE(T::Type{<:AbstractFloat}; maxlevel::Integer=10, kwargs...)
    @assert maxlevel > 0
    qts = QuadTS(T; maxlevel=maxlevel, kwargs...)
    qes = QuadES(T; maxlevel=maxlevel, kwargs...)
    qss = QuadSS(T; maxlevel=maxlevel, kwargs...)
    QuadDE{T,maxlevel+1}(qts, qes, qss)
end

function (q::QuadDE{T,N})(f::Function, a::Real, b::Real;
                          kwargs...) where {T<:AbstractFloat,N}
    if a > b
        I, E = q(f, b, a; kwargs...)
        return -I, E
    end

    if a == b
        return zero(f(a)), zero(T)
    end

    if a == -Inf && b == Inf
        # integrate over [-∞, ∞]
        return q.qss(f; kwargs...)
    elseif b == Inf
        if a == 0
            # integrate over [0, ∞]
            return q.qes(f; kwargs...)
        else
            # integrate over [a, ∞]
            return q.qes(u -> f(u + a); kwargs...)
        end
    elseif a == -Inf
        # integrate over [-∞, b]
        if b == 0
            return q.qes(u -> f(-u); kwargs...)
        else
            return q.qes(u -> f(-u + b); kwargs...)
        end
    else
        # integrate over [a, b]
        return q.qts(f, a, b; kwargs...)
    end
end
function (q::QuadDE{T,N})(f::Function, intervals::Tuple{Real,Real,Vararg{Real}};
                          atol::Real=zero(T), rtol::Real=atol>0 ? zero(T) : sqrt(eps(T))) where {T<:AbstractFloat,N}
    f⁺ = f
    f⁻ = u -> f(-u)
    I = init(x -> f(x), q, intervals)
    h0 = q.qts.h0
    Ih = I*h0
    E = zero(eltype(Ih))
    es_istart⁺ = 1
    ss_istart⁺ = 1
    ss_istart⁻ = 1
    for level in 0:(N-1)
        ts_table  = q.qts.tables[level+1]
        es_table⁺ = q.qes.tables⁺[level+1]
        es_table⁻ = q.qes.tables⁻[level+1]
        ss_table  = q.qss.tables[level+1]
        es_istart⁺ = startindex(f⁺, es_table⁺, es_istart⁺)
        ss_istart⁺ = startindex(f⁺, ss_table, ss_istart⁺)
        ss_istart⁻ = startindex(f⁻, ss_table, ss_istart⁻)
        I += increment(f, ts_table, es_table⁺, es_table⁻, ss_table,
                       es_istart⁺, ss_istart⁺, ss_istart⁻, intervals)
        h = h0/2^level
        prevIh = Ih
        Ih = I*h
        E = norm(prevIh - Ih)
        !(E > max(norm(Ih)*rtol, atol)) && level > 0 && break
    end
    Ih, E
end
(q::QuadDE{T,N})(f::Function, a::Real, b::Real, c::Real...; kwargs...) where {T<:AbstractFloat,N} = q(f, (a, b, c...); kwargs...)


function init(f::Function, q::QuadDE{T,N}, a::Real, b::Real) where {T<:AbstractFloat,N}
    if a > b
        return -init(x -> f(x), q, b, a)
    end

    if a == b
        return zero(f(a))
    end

    if a == -Inf && b == Inf
        # integrating over [-∞, ∞]
        x0, w0 = q.qss.origin
        return f(x0)*w0
    elseif b == Inf
        # integrating over [a, ∞]
        x0, w0 = q.qes.origin
        _a = T(a)
        return f(x0 + _a)*w0
    elseif a == -Inf
        # integrating over [-∞, b]
        return init(u -> f(-u), q, -b, Inf)
    else
        # integrating over [a, b]
        return init(x -> f(x), q.qts, a, b)
    end
end
function init(f::Function, q::QuadDE{T,N}, intervals) where {T<:AbstractFloat,N}
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


function increment(f::Function,
                   ts_table::QuadTSWeightTable{T},
                   es_table⁺::QuadESWeightTable{T},
                   es_table⁻::QuadESWeightTable{T},
                   ss_table::QuadSSWeightTable{T},
                   es_istart⁺::Integer, ss_istart⁺::Integer, ss_istart⁻::Integer,
                   a::Real, b::Real) where {T<:AbstractFloat}
    if a > b
        return -increment(f, ts_table, es_table⁺, es_table⁻, ss_table,
                          es_istart⁺, ss_istart⁺, ss_istart⁻, b, a)
    end

    if a == b
        return zero(f(a))
    end

    if a == -Inf && b == Inf
        # integrating over [-∞, ∞]
        return increment(f, ss_table, ss_istart⁺, ss_istart⁻)
    elseif b == Inf
        # integrating over [a, ∞]
        if a == 0
            return increment(f, es_table⁺, es_table⁻, es_istart⁺)
        else
            _a = T(a)
            return increment(u -> f(u + _a), es_table⁺, es_table⁻, es_istart⁺)
        end
    elseif a == -Inf
        # integrating over [-∞, b]
        if b == 0
            return increment(u -> f(-u), es_table⁺, es_table⁻, es_istart⁺)
        else
            _b = T(b)
            return increment(u -> f(-u + _b), es_table⁺, es_table⁻, es_istart⁺)
        end
    else
        # integrating over [a, b]
        return increment(f, ts_table, a, b)
    end
end
function increment(f::Function,
                   ts_table::QuadTSWeightTable{T},
                   es_table⁺::QuadESWeightTable{T},
                   es_table⁻::QuadESWeightTable{T},
                   ss_table::QuadSSWeightTable{T},
                   es_istart⁺::Integer, ss_istart⁺::Integer, ss_istart⁻::Integer,
                   intervals) where {T<:AbstractFloat}
    n = length(intervals)
    a = T(intervals[1])
    b = T(intervals[2])
    I = increment(f, ts_table, es_table⁺, es_table⁻, ss_table,
                  es_istart⁺, ss_istart⁺, ss_istart⁻, a, b)
    for i in 3:n
        a = T(intervals[i-1])
        b = T(intervals[i])
        I += increment(f, ts_table, es_table⁺, es_table⁻, ss_table,
                       es_istart⁺, ss_istart⁺, ss_istart⁻, a, b)
    end
    I
end
