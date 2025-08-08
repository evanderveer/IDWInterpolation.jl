module IDWInterpolation

using NearestNeighbors: KDTree, knn
using LinearAlgebra
using StaticArrays

include("kernels.jl")
using .Kernels

export interpolate, Kernels

function interpolate(
	points::AbstractMatrix{T}, 
	values::AbstractVector{T}, 
	resolution::Tuple, 
	num_neighbors::Integer, 
	kernel::Kernels.Kernel
	) where T <: Real
	if size(points, 2) != length(resolution)
		error("number of resolution values provided must be equal to dimensionality of data")
	end
	tree = KDTree(permutedims(points, (2,1)))

	mins = Tuple(minimum(points, dims=1))
	maxs = Tuple(maximum(points, dims=1))
	nodes = homogeneous_hypercube(resolution, mins, maxs)

	idxs, dists = knn(tree, nodes, num_neighbors)

	interpolated_matrix = Matrix{T}(undef, length(nodes), size(points, 2) + 1)

	for i in axes(idxs, 1)
		interpolated_matrix[i, 1:end-1] .= nodes[i]

		weights = kernel.(dists[i])
		weight_sum = sum(weights)
		vals = values[idxs[i]]
		interpolated_matrix[i, end] = sum(vals .* weights) / weight_sum
	end

	return interpolated_matrix
end

function interpolate(
	points::AbstractMatrix{T}, 
	values::AbstractVector{T}, 
	resolution::Integer, 
	num_neighbors::Integer, 
	kernel::Kernels.Kernel
	) where T <: Real
	interpolate(
		points, 
		values, 
		Tuple(resolution for _ in 1:size(points, 2)), 
		num_neighbors, 
		kernel
		)
end

function interpolate(
	matrix::AbstractMatrix{<:Real}, 
	resolution::Int, 
	num_neighbors::Integer, 
	kernel::Kernels.Kernel
	)

	if size(matrix, 2) < 2
		error("data matrix must have > 1 column")
	end
	points = matrix[:, 1:end-1]
	values = vec(matrix[:, end])
	interpolate(
		points,
		values, 
		resolution, 
		num_neighbors, 
		kernel
		)
end


"""

Adapted from KernelInterpolation.jl
Copyright (c) 2023-present Joshua Lampert <joshua.lampert@uni-hamburg.de> and contributors
"""
function homogeneous_hypercube(
	n::NTuple{Dim, Int},
	x_min::NTuple{Dim, RealT} = ntuple(_ -> 0.0, Dim),
	x_max::NTuple{Dim, RealT} = ntuple(_ -> 1.0, Dim);
	dim = Dim
	) where {Dim, RealT}

	@assert Dim == dim
	nodes = Vector{MVector{Dim, float(RealT)}}(undef, prod(n))
	for (i, indices) in enumerate(Iterators.product(ntuple(j -> 1:n[j], Dim)...))
		node = MVector{Dim, float(RealT)}(undef)
		for j in 1:dim
			node[j] = x_min[j] + (x_max[j] - x_min[j]) * (indices[j] - 1) / (n[j] - 1)
		end
		nodes[i] = node
	end
	return nodes
end

end
