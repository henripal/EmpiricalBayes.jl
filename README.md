# EmpiricalBayes.jl

Basic julia package for conducting Empirical Bayes analysis on DataFrames or raw data.
Usage is extensively shown in my analysis of metacritic critic data [MetacriticBayes](https://github.com/henripal/MetacriticBayes).

For now this supports:
- Dirichlet / Categorical
- Normal / Normal (with known variance...)

It can directly be used on a dataframes with normal or categorical data in one of the columns and id in another column, and provides likelihoods, posteriors, and common prior.

