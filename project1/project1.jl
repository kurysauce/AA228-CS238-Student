using Graphs
using Printf
using DelimitedFiles  
using SpecialFunctions  
using Distributions

function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function mutual_information(x::Vector{Int}, y::Vector{Int})
    n = length(x)
    px = Dict(k => count(i -> i == k, x) / n for k in unique(x))
    py = Dict(k => count(i -> i == k, y) / n for k in unique(y))
    pxy = Dict((a, b) => count(i -> x[i] == a && y[i] == b, 1:n) / n for a in unique(x), b in unique(y))

    mi = 0.0
    for (a, b) in keys(pxy)
        if pxy[(a, b)] > 0
            mi += pxy[(a, b)] * log(pxy[(a, b)] / (px[a] * py[b]))
        end
    end
    return mi
end

function write_dot(gph_filename, dot_filename)
    open(gph_filename, "r") do infile
        open(dot_filename, "w") do outfile
            println(outfile, "digraph G {")
            for line in eachline(infile)
                src, dst = split(line, ",")
                println(outfile, "    $src -> $dst;")
            end
            println(outfile, "}")
        end
    end
end

function sub2ind(dims::Vector{Int}, indices::Vector{Int})
    linear_index = 1
    stride = 1
    for i in 1:length(dims)
        linear_index += (indices[i] - 1) * stride
        stride *= dims[i]
    end
    return linear_index
end

function statistics(vars, G, D::Matrix{Int})
    n = length(vars)
    r = [length(unique(D[:, i])) for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]

    M = [zeros(q[i], r[i]) for i in 1:n]

    for o in eachcol(D)
        for i in 1:n
            k = o[i]

            if k > r[i] || k < 1
                k = min(max(k, 1), r[i])
            end

            parents = inneighbors(G, i)

            j = 1
            if !isempty(parents)
                parent_indices = o[parents]
                if all(x -> x > 0 && x <= r[i], parent_indices)
                    j = sub2ind(r[parents], parent_indices)
                else
                    continue
                end
            end

            if j <= q[i] && k <= r[i]
                M[i][j, k] += 1.0
            end
        end
    end

    return M
end

function flexible_prior(vars, G, D::Matrix{Int}, alpha_scale=0.5)
    n = length(vars)
    r = [length(unique(D[:, i])) for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]

    priors = [ones(q[i], r[i]) .* alpha_scale for i in 1:n]
    for i in 1:n
        freq_dict = Dict{Int, Int}()
        for value in D[:, i]
            if value > r[i] || value < 1
                value = min(max(value, 1), r[i])
            end
            freq_dict[value] = get(freq_dict, value, 0) + 1
        end
        for value in keys(freq_dict)
            priors[i][:, value] .= freq_dict[value] * alpha_scale
        end
    end
    return priors
end

function bayesian_score_component(M, α)
    p = sum(loggamma.(α + M))
    p -= sum(loggamma.(α))
    p += sum(loggamma.(sum(α, dims=2)))
    p -= sum(loggamma.(sum(α, dims=2) + sum(M, dims=2)))

    return p
end

function bayesian_score(vars, G, D)
    M = statistics(vars, G, D)
    α = flexible_prior(vars, G, D)

    score = sum(bayesian_score_component(M[i], α[i]) for i in 1:length(vars))

    return score
end

function k2_algorithm_with_mi(vars, D, max_parents)
    n = length(vars)
    G = SimpleDiGraph(n)

    best_score = bayesian_score(vars, G, D)

    for i in 2:n
        mi_scores = Dict()
        for j in 1:(i-1)
            mi_scores[j] = mutual_information(D[:, i], D[:, j])
        end

        sorted_parents = sort(collect(keys(mi_scores)), by=x -> -mi_scores[x])

        for j in sorted_parents[1:min(max_parents, length(sorted_parents))]
            if outdegree(G, j) < max_parents && !has_edge(G, j, i)
                add_edge!(G, j, i)
                new_score = bayesian_score(vars, G, D)

                if new_score > best_score
                    best_score = new_score
                else
                    rem_edge!(G, j, i)
                end
            end
        end
    end

    return G, best_score
end




function compute(infile, outfile, scorefile, dotfile)
    # Load your data here from the CSV file
    data_tuple = readdlm(infile, ','; header=true)
    header = data_tuple[2]  # Extract header (variable names)
    data = data_tuple[1]    

    vars = header[1, :]  # Extract variable names dynamically
    D = Int64.(data)

    max_parents = 4
    G, best_score = k2_algorithm_with_mi(vars, D, max_parents)  # Get the graph and the score

    # Generalize the idx2names mapping
    idx2names = Dict(i => header[i] for i in 1:length(header))
    
    write_gph(G, idx2names, outfile)  
    write_dot(outfile, dotfile)  # Create a uniquely named .dot file for the graph
end


if length(ARGS) != 4
    error("usage: julia project1.jl <infile>.csv <outfile>.gph <scorefile>.txt <dotfile>.dot")
end

inputfilename = ARGS[1]
outputfilename = ARGS[2]
scorefilename = ARGS[3]
dotfilename = ARGS[4]



compute(inputfilename, outputfilename, scorefilename, dotfilename)
