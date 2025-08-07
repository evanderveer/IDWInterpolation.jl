module IDWInterpolation

using NearestNeighbors: KDTree
using LinearAlgebra
using StaticArrays

include("kernels.jl")
using .Kernel

export interpolate, Kernel

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

function make_matrix(
	nodes, 
	values, 
	N
	)

	xs = unique(node[1] for node in nodes)
	ys = unique(node[2] for node in nodes)

	matrix = reshape(values, (N,N))
	return(xs, ys, matrix)
end

function interpolate(
	points, 
	values, 
	resolution, 
	num_neighbors, 
	shape_parameter
	)
	if size(points, 2) != length(resolution)
		error("number of resolution values provided must be equal to dimensionality of data")
	end
	tree = KDTree(Float64.(points)')

	mins = Tuple(minimum(points, dims=1))
	maxs = Tuple(maximum(points, dims=1))
	nodes = homogeneous_hypercube(resolution, mins, maxs)

	idxs, dists = knn(tree, nodes, num_neighbors)

	ivalues = Vector{Float64}(undef, length(nodes))
	for i in 1:length(idxs)
		weights = Kernel.gaussian.(dists[i], shape_parameter)
		weight_sum = sum(weights)
		vals = values[idxs[i]]
		ivalues[i] = sum(vals .* weights) / weight_sum
	end
	
	return (nodes, ivalues)
end

function interpolate(
	points, 
	values, 
	resolution::Int, 
	num_neighbors, 
	shape_parameter
	)
	interpolate(points, values, [resolution for _ in 1:size(points, 1)], num_neighbors, shape_parameter)
end

end
