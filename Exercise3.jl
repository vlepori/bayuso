import CSV, DataFrames
using Plots

include("pdfs.jl")
include("mcmc.jl")

data = CSV.File(read("global_temperature_NASA_scaled_time_start_1900.txt")) |> DataFrames.DataFrame
#data[:,1]=data[:,1].+1880
Plots.plot(data[:,1],data[:,2])

function lik_wn(theta)
    mus = theta[1] .+ theta[2]*data[:,1]
    return sum(log_pdf_normal.(data[:,2],mus,theta[3]))
end
    
function prior_wn(theta)
    prior_mu_zero = 0
    prior_a = log_pdf_normal(theta[2],0,1) 
    prior_sigma = log_pdf_gamma(theta[3],2,.02)
    return prior_mu_zero+prior_a+prior_sigma
end

function kernels_wn(theta)
    n_mu0 = sliding_window(theta[1], 1)
    n_a = sliding_window(theta[2],.01)
    n_sigma = abs(sliding_window(theta[3], .05))
    return [n_mu0, n_a, n_sigma]
end

n_iter=9000000
init = [13.5,0.01,0.1] # mu0, a, sigma
test_thetas, test_probs = sample_mcmc(init, kernels_wn, prior_wn, lik_wn, n_iter, se=100 )
plot(test_thetas, layout = (3, 1), legend=false, title=["mu0" "a" "sigma"])
plot(test_probs, layout = (3, 1), legend=false, title=["log prior" "log likelihood" "log unnormalized posterior"])
histogram(test_thetas[:,1],legend=false, title = "mu0")
Plots.plot(data[:,1],data[:,2], legend = false, seriestype = :scatter)
for i in 1:200
    row = Int(round(size(test_thetas)[1]*rand()))
    plot!(0:119,x->test_thetas[row,1]+ x*test_thetas[row,2], alpha = 0.1, color = "grey") #
end
plot!()

