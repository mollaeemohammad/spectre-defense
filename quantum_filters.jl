include("dkk17.jl")
include("bench_util.jl")

function rcov_quantum_filter(reps, eps, k, α=4, τ=0.1; limit1=2, limit2=1.5)
    d, n = size(reps)
    reps_pca, U = pca(reps, k)
    if k == 1
        reps_estimated_white = reps_pca
        Σ′ = ones(1, 1)
    else
        selected, t = timed_median_call(cov_estimation_iterate, reps_pca, eps/n, τ, nothing; n_reps=5, limit=round(Int, limit1*eps))
        println("estimator time (median of 5): $t seconds")
        reps_pca_selected = reps_pca[:, selected]
        Σ = cov(reps_pca_selected', corrected=false)
        reps_estimated_white = Σ^(-1/2)*reps_pca
        Σ′ = cov(reps_estimated_white')
    end
    M = k > 1 ? exp(α*(Σ′- I)/(opnorm(Σ′) - 1)) : ones(1, 1)
    M /= tr(M)
    estimated_poison_ind = k_lowest_ind(
        -[x'M*x for x in eachcol(reps_estimated_white)],
        round(Int, limit2*eps)
    )
    return .! estimated_poison_ind
end

function rcov_auto_quantum_filter(reps, eps, α=4, τ=0.1; limit1=2, limit2=1.5, k_max=100)
    reps_pca, U = pca(reps, k_max)
    best_opnorm, best_selected, best_k = -Inf, nothing, nothing
    for k in round.(Int, range(1, sqrt(k_max), length=round(Int, sqrt(k_max))) .^ 2)
        selected = rcov_quantum_filter(reps, eps, k, α, τ; limit1=limit1, limit2=limit2)
        Σ = cov(reps_pca[:, selected]')
        Σ′ = cov((Σ^(-1/2)*reps_pca)')
        M = exp(α*(Σ′- I)/(opnorm(Σ′) - 1))
        M /= tr(M)
        op = tr(Σ′ * M) / tr(M)
        poison_removed = sum((.! selected)[end-eps+1:end])
        @show k, op, poison_removed
        if op > best_opnorm
            best_opnorm, best_selected, best_k = op, selected, k
        end
    end
    @show best_k, best_opnorm
    return best_selected
end
