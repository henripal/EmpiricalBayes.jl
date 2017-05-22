
"""
Prior, posterior and likelihood are Normal.
Note that this presupposes that we know the variance of the Normal.
"""
type NormalNormal{S} <: BayesModel
    sample_sizes::Dict{S, Int64}
    prior::Distributions.Normal{Float64}
    likelihoods::Dict{S, Distributions.Normal{Float64}}
    posteriors::Dict{S, Distributions.Normal{Float64}}
end

Base.show(io::IO, nn::NormalNormal) = print(io, typeof(nn))
"""
Returns a fitted NormalNormal.
values is a dictionary indexed by ids, whose values are arrays
of floats (hypothesized to follow some normal distribution with
known variance)
"""
function NormalNormal{S}(values::Dict{S, Array{Float64, 1}})
    sample_sizes = Dict{S, Int64}()
    likelihoods = Dict{S, Distributions.Normal{Float64}}()
    posteriors = Dict{S, Distributions.Normal{Float64}}()
    means = zeros(length(values))

    # this is awful. 
    all_values = Float64[]
    for (_, value) in values
       all_values = vcat(all_values, value) 
    end

    σ = sqrt(var(all_values)) #n-1?

    for (i, (name, value)) in enumerate(values)
        sample_sizes[name] = length(value)
        likelihoods[name] = Normal(mean(value), σ)
        means[i] = likelihoods[name].μ
    end

    prior = fit(Normal, means)

    for (name, value) in values
        # conjugate magic
        n = sample_sizes[name]
        posterior_variance = 1/(1/prior.σ^2 + n/σ^2)
        posterior_mean = (prior.μ/prior.σ^2 + sum(value)/σ^2)*posterior_variance
        posteriors[name] = Normal(posterior_mean, sqrt(posterior_variance))
    end

    NormalNormal(sample_sizes, prior, likelihoods, posteriors)
end

"""
Convenience function fitting the NormalNormal on a dataframe; the ids are
in column id and the sample values are in the column category
"""
function normal_from_df(df::DataFrames.DataFrame, id::Symbol, value_col::Symbol )
    unique_ids = unique(df[id])

    n_ids = length(unique_ids)

    values = Dict{typeof(unique_ids[1]), Array{Float64, 1}}()

    for name in unique_ids
        categories = df[df[id] .== name, :][value_col]
        values[name] = categories
    end

    return NormalNormal(values)
end
