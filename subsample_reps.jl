using NPZ, Random

name        = ARGS[1]                       # e.g. "r18-sgd-04-1xp500"
eps_full    = parse(Int, ARGS[2])           # 500
target_eps  = parse(Int, ARGS[3])           # e.g. 100
target_label = 4                             # match your run's target label

reps = npzread("output/$name/label_$(target_label)_reps.npy")
println("loaded reps size: ", size(reps))    # sanity check orientation — should be (n_samples, d)

n_clean = size(reps, 1) - eps_full
clean_reps = reps[1:n_clean, :]
poison_reps_full = reps[n_clean+1:end, :]

Random.seed!(1)
keep = randperm(eps_full)[1:target_eps]
poison_reps_sub = poison_reps_full[keep, :]

reps_sub = vcat(clean_reps, poison_reps_sub)
npzwrite("output/$name/label_$(target_label)_reps_eps$(target_eps).npy", reps_sub)
println("wrote subsampled reps: eps=$target_eps, total n=$(size(reps_sub,1))")