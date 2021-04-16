import SpecialFunctions
pdf_exp(x,lambda) =  lambda*exp(-lambda*x)
log_pdf_exp(x,lambda) = log(lambda) - lambda*x   
log_pdf_uniform(x,a,b) = ((x>a) & (x<b)) ? -log(b-a) : -Inf
log_pdf_normal(x,mu,sigma) = -.5*log(2pi*sigma^2) -((x-mu)^2/(2*sigma^2))
log_pdf_gamma(x,al,bt) = al*log(bt) - log(SpecialFunctions.gamma(al)) + (al - 1)*log(x) - bt*x
