module IDWInterpolation

using NearestNeighbors: KDTree, knn
using LinearAlgebra
using StaticArrays
using StatsBase: weights, mean

include("kernels.jl")
using .Kernels

export interpolate, Kernels

"""
    interpolate(points, values, resolution, num_neighbors, kernel)

Perform scattered data interpolation using inverse distance weighting (or other user-defined kernels).

# Arguments
- `points::AbstractMatrix{T}`: `(N × D)` matrix of sample point coordinates, where `D` is dimensionality.
- `values::AbstractVector{T}`: Length-`N` vector of sample values.
- `resolution::Int`: Number of grid divisions in each dimension.
- `num_neighbors::Integer`: Number of nearest neighbors to use for interpolation.
- `kernel::Kernels.Kernel`: Kernel object or callable for weighting distances.

# Returns
`Matrix{T}` of size `(M × (D+1))`, where `M = prod(resolution)`.  
First `D` columns are coordinates, last column is interpolated value.
"""
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

"""
    interpolate(points, values, resolution, num_neighbors, kernel)

Perform scattered data interpolation using inverse distance weighting (or other user-defined kernels).

# Arguments
- `matrix::AbstractMatrix{T}`: `(N × D+1)` matrix of sample point coordinates, where `D` is dimensionality. The last column are the sample values.
- `resolution::Int`: Number of grid divisions in each dimension.
- `num_neighbors::Integer`: Number of nearest neighbors to use for interpolation.
- `kernel::Kernels.Kernel`: Kernel object or callable for weighting distances.

# Returns
`Matrix{T}` of size `(M × (D+1))`, where `M = prod(resolution)`.  
First `D` columns are coordinates, last column is interpolated value.
"""
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
    interpolate(points, values, resolution, num_neighbors, kernel)

Perform scattered data interpolation using inverse distance weighting (or other user-defined kernels).

# Arguments
- `points::AbstractMatrix{T}`: `(N × D)` matrix of sample point coordinates, where `D` is dimensionality.
- `values::AbstractVector{T}`: Length-`N` vector of sample values.
- `resolution::NTuple{D,Int}`: Number of grid divisions in each dimension.
- `num_neighbors::Integer`: Number of nearest neighbors to use for interpolation.
- `kernel::Kernels.Kernel`: Kernel object or callable for weighting distances.

# Returns
`Matrix{T}` of size `(M × (D+1))`, where `M = prod(resolution)`.  
First `D` columns are coordinates, last column is interpolated value.
"""
function interpolate(
	points::AbstractMatrix{T}, 
	values::AbstractVector{T}, 
	resolution::NTuple{Dim, <:Integer}, 
	num_neighbors::Integer, 
	kernel::Kernels.Kernel
	) where {T <: Real, Dim}

	if size(points, 2) != length(resolution)
		error("number of resolution values provided must be equal to dimensionality of data")
	end
	
	# The grid on which to interpolate the data
	grid_points = interpolation_grid(points, resolution)

	# Get the neighbors for each grid point
	tree = KDTree(permutedims(points, (2,1)))
	idxs, dists = knn(tree, grid_points, num_neighbors)

	out_size = (length(grid_points), Dim + 1)
	out = Matrix{T}(zeros(T, out_size...))

	# Preallocate
	w = Vector{T}(undef, num_neighbors)
	v = Vector{T}(undef, num_neighbors)

	@inbounds for i in axes(idxs, 1)
		@views out[i, 1:Dim] .= grid_points[i]

		@views w .= kernel.(dists[i])
		@views v .= values[idxs[i]]
		out[i, end] = mean(v, weights(w))
	end

	out
end

function interpolation_grid(points, resolution)
	bound(f) = Tuple(f(points, dims=1))
	homogeneous_hypercube(resolution, bound(minimum), bound(maximum))
end

"""
    homogeneous_hypercube(n, x_min, x_max)

Generate evenly spaced grid points filling a hypercube.

# Arguments
- `n::NTuple{Dim,Int}`: Number of points along each dimension.
- `x_min::NTuple{Dim,Real}`: Lower bounds for each dimension (default `0.0`).
- `x_max::NTuple{Dim,Real}`: Upper bounds for each dimension (default `1.0`).

# Returns
`Vector{MVector{Dim,Float64}}` containing all grid points.

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
	grid_points = Vector{MVector{Dim, float(RealT)}}(undef, prod(n))
	for (i, indices) in enumerate(Iterators.product(ntuple(j -> 1:n[j], Dim)...))
		node = MVector{Dim, float(RealT)}(undef)
		for j in 1:dim
			node[j] = x_min[j] + (x_max[j] - x_min[j]) * (indices[j] - 1) / (n[j] - 1)
		end
		grid_points[i] = node
	end
	return grid_points
end

end
