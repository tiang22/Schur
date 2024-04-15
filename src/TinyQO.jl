module TinyQO

using LinearAlgebra
using SparseArrays
using DormandPrince
using DocStringExtensions

export mesolve, Lindblad, qihao_lindblad
export TimeDependent, MesolveRes

include("outerprod.jl")
include("simulator.jl")

end