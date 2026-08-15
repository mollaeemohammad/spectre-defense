using LinearAlgebra, Statistics

function robust_mahalanobis_iterate(S′, ε, τ=0.1, k=nothing; iters=5, limit=nothing)
    d, n = size(S′)
    idxs = collect(1:n)
    remove_total = limit === nothing ? round(Int, 1.5*ε*n) : limit
    per_iter = max(1, round(Int, remove_total / iters))

    for it in 1:iters
        μ = vec(mean(S′, dims=2))          # <-- fix: force 1D vector
        Σ = cov(S′', corrected=false) + 1e-6I
        invΣ = inv(Σ)
        d2 = [((S′[:,i]-μ)' * invΣ * (S′[:,i]-μ)) for i in 1:size(S′,2)]  # now scalar per entry
        n_remove = min(per_iter, size(S′,2)-1)
        keep = sortperm(d2)[1:end-n_remove]
        S′ = S′[:, keep]
        idxs = idxs[keep]
    end

    select = falses(n)
    for i in idxs
        select[i] = true
    end
    return select
end