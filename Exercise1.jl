using Plots, SpecialFunctions

include("pdfs.jl")
include("mcmc.jl")

pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]

lambda_range = 0.001:0.0001:0.1
logliks = [sum(log_pdf_exp.(pandas_bm,i)) for i in lambda_range]
plot(lambda_range,logliks,legend = false, xlabel = "lambda", ylabel = "log likelihood")
vline!([lambda_range[argmax(logliks)]]) # MLE exponential model

# MLE for Normal distribution model
ns_mu=100
ns_sig = 50

mus = range(84,stop=125,length=ns_mu)
sigs = range(6,stop=50,length=ns_sig)

norm_logliks = [sum(log_pdf_normal.(pandas_bm,mu,sig)) for mu=mus, sig=sigs]
heatmap(mus,sigs,norm_logliks', title="loglik")
mle = argmax(norm_logliks) # maximum likelihood estimate
vline!([mus[mle[1]]], color="black", legend = false)
hline!([sigs[mle[2]]], color="black", legend = false)

# unnormalized posterior
norm_post = zeros(ns_mu,ns_sig)
for i in 1:ns_mu
    for j in 1:ns_sig
        logprior =  log_pdf_gamma(sigs[j],2,.02) * log_pdf_uniform(mus[i],10,1000)
        loglik = sum(log_pdf_normal.(pandas_bm,mus[i],sigs[j]))
        norm_post[i,j]= loglik+logprior
    end
end
heatmap(mus,sigs,norm_post',xlabel = "mu", ylabel="sigma", title = "unnormalized p(theta|data)")
map = argmax(norm_post) # maximum a posteriori
plot!([mus[map[1]]],[sigs[map[2]]] ,  color="red", legend = false, markershape = :x)
plot!([mus[mle[1]]],[sigs[mle[2]]] ,  color="black", legend = false, markershape = :x)