pdf_exp(x,lambda) =  lambda*exp(-lambda*x)
log_pdf_exp(x,lambda) = log(lambda) - lambda*x    

pandas_bm = [84.74, 84.88, 94.60, 96.37, 102.93, 109.11, 125.76]
log_pdf_exp.(pandas_bm,0.1) |> sum

lambda_range = 0.001:0.0001:0.1
logliks = [sum(log_pdf_exp.(pandas_bm,i)) for i in lambda_range]

import Plots
Plots.plot(lambda_range,logliks)
Plots.vline!([lambda_range[argmax(logliks)]])


