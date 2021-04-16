using Plots
pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]

include("pdfs.jl")
include("mcmc.jl")

# run
normal_lik(x) = sum(log_pdf_normal.(pandas_bm,x[1],x[2]))
u_kernels(x) = [(sliding_window(x[1],10)),abs(sliding_window(x[2],10))]
gamma_prior(x) = 0+log_pdf_gamma(x[2],2,.02)
n_iter=100000
test_thetas, test_probs = sample_mcmc([100.0,20.0], u_kernels, gamma_prior, normal_lik, n_iter,se= 10 )
plot(test_thetas, layout = (2, 1), legend=false, title=["mu" "sigma"])
plot(test_probs, layout = (3, 1), legend=false, title=["log prior" "log likelihood" "log unnormalized posterior"])

# import DataFrames, CSV
# df = DataFrames.DataFrame(hcat(test_probs,test_thetas))
# DataFrames.rename!(df, ["mu","sigma","prior","lik","post"])
# CSV.write("trace.tsv", df; delim = '\t')