module DoubleExponentialFormulas

using LinearAlgebra: norm

export
    quadde,
    quadts,
    quades,
    quadss,
    QuadTS

const h0inv = 8

function generate_table end
function trapez end

function quadde end

include("quadts.jl")
function quadts end

function quades end

function quadss end


end # module
