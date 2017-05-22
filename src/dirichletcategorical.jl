
"""
prior and posterior are Dirichlets
likelihoods are Categorical
"""
type DirichletCategorical{S} <: BayesModel
    sample_sizes::Dict{S, Int64}
    prior::Distributions.Dirichlet{Float64}
    likelihoods::Dict{S, Distributions.Categorical{Float64}}
    posteriors::Dict{S, Distributions.Dirichlet{Float64}}
end

Base.show(io::IO, dc::DirichletCategorical) = print(io, typeof(dc))

"""
Returns a fitted DirichletCategorical.
Requires a dictionary of inputs, where the dictionary is indexed by
the different ids, and whose value is an array of categories.
Categories are integers from 1 to n_categories.
"""
function DirichletCategorical{S}(counts::Dict{S, Array{Int64, 1}}, n_categories::Int64)
    sample_sizes = Dict{S, Int64}()
    likelihoods = Dict{S, Distributions.Categorical{Float64}}()
    posteriors = Dict{S, Distributions.Dirichlet{Float64}}()
    P = zeros(length(counts), n_categories)

    for (i, (name, count)) in enumerate(counts)
        sample_sizes[name] = length(count)
        likelihoods[name] = fit(Categorical, n_categories, count) 
        P[i, :] = zero_out(likelihoods[name].p)
    end

    prior = fit(Dirichlet, P')

    for (name, count) in counts
        # conjugate magic
        posteriors[name] = Dirichlet(likelihoods[name].p * sample_sizes[name] + prior.alpha)
    end

    DirichletCategorical(sample_sizes, prior, likelihoods, posteriors)
end

"""
Convenience function fitting the DirichletCategorical on a dataframe; the ids are
in column id and the categories each id belongs to are in the column category
"""
function dirichlet_from_df(df::DataFrames.DataFrame, id::Symbol, category::Symbol )
    unique_ids = unique(df[id])
    unique_categories = unique(df[category])

    n_ids = length(unique_ids)
    n_cat = length(unique_categories)

    counts = Dict{typeof(unique_ids[1]), Array{Int64, 1}}()

    for name in unique_ids
        categories = df[df[id] .== name, :][category]
        counts[name] = categories
    end

    return DirichletCategorical(counts, n_cat)
end
