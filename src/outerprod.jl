export LowRankMatrix, OuterProduct, outerprod
using LinearAlgebra
abstract type LowRankMatrix{T} <: AbstractMatrix{T} end

"""
$(TYPEDEF)

If `ATL(R) <: AbstractVector`, it represents an outer product `x.left*x.right'`.
Else if `ATL(R) <: AbstractMatrix`, then it is an outer product `x.left*x.right'`.
"""
struct OuterProduct{T,ATL<:AbstractArray{T},ATR<:AbstractArray{T}} <: LowRankMatrix{T}
    left::ATL
    right::ATR
    function OuterProduct(left::ATL, right::ATR) where {T,ATL<:AbstractArray{T},ATR<:AbstractArray{T}}
        size(left, 2) != size(right, 2) && throw(
            DimensionMismatch(
                "The seconds dimension of left ($(size(left,2))) and right $(size(right,2)) does not match.",
            ),
        )
        return new{T,ATL,ATR}(left, right)
    end
end

const BatchedOuterProduct{T,MTL,MTR} =
    OuterProduct{T,MTL,MTR} where {MTL<:AbstractMatrix,MTR<:AbstractMatrix}
const SimpleOuterProduct{T,VTL,VTR} =
    OuterProduct{T,VTL,VTR} where {VTL<:AbstractVector,VTR<:AbstractVector}

LinearAlgebra.rank(op::OuterProduct) = size(op.left, 2)
Base.getindex(op::OuterProduct{T,<:AbstractVector}, i::Int, j::Int) where {T} =
    op.left[i] * conj(op.right[j])
Base.getindex(op::BatchedOuterProduct, i::Int, j::Int) =
    sum(k -> op.left[i, k] * conj(op.right[j, k]), 1:rank(op))
Base.size(op::OuterProduct) = (size(op.left, 1), size(op.right, 1))
Base.size(op::OuterProduct, i::Int) =
    i == 1 ? size(op.left, 1) : (i == 2 ? size(op.right, 1) : throw(DimensionMismatch("")))

Base.:(*)(a::OuterProduct, b::AbstractVector) = a.left * (a.right' * b)
Base.:(*)(a::OuterProduct, b::AbstractMatrix) =
    OuterProduct(a.left, (a.right' * b)')
Base.:(*)(a::AbstractMatrix, b::OuterProduct) = OuterProduct(a * b.left, b.right)
Base.:(*)(a::LinearAlgebra.AdjointAbsVec, b::OuterProduct) = (a * b.left) * b.right'
Base.:(*)(a::OuterProduct, b::OuterProduct) =
    OuterProduct(a.left * (a.right' * b.left), b.right)
LinearAlgebra.rmul!(a::OuterProduct, b::Number) = (rmul!(a.left, b); a)
Base.:(*)(a::Number, b::OuterProduct) = OuterProduct(a * b.left, b.right)
Base.:(*)(b::OuterProduct, a::Number) = a * b

Base.conj!(op::OuterProduct) = OuterProduct(conj!(op.left), conj!(op.right))
LinearAlgebra.transpose!(op::OuterProduct) = OuterProduct(conj!(op.right), conj!(op.left))
Base.adjoint(op::OuterProduct) = OuterProduct(op.right, op.left)

outerprod(left::AbstractVector, right::AbstractVector) = OuterProduct(left, right)
outerprod(left::AbstractMatrix, right::AbstractMatrix) = OuterProduct(left, right)
