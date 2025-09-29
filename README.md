### Quantized SG-MCMC

This paper investigates the use of quantization-aware Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) methods for compressing Bayesian deep learning models. We explore how low-precision quantization, applied to methods such as Stochastic Gradient Hamiltonian Monte Carlo (SGHMC) and Stochastic Gradient Langevin Dynamics (SGLD), can significantly reduce model size without sacrificing accuracy and the quality of uncertainty estimation. The impact of different gradient accumulation techniques and variance correction strategies on the performance of quantized algorithms is evaluated. Furthermore, we analyze the effectiveness of MCMC thinning techniques in reducing correlation between samples, and its effect on model compression. Empirical results demonstrate that low-precision quantization with SGMCMC, combined with thinning, offers an effective method for compressing Bayesian models, presenting a competitive alternative to other compression techniques.


# TODO:

1.) Mover API Antigua (Linen) a  API nueva (https://flax.readthedocs.io/en/latest/nnx_basics.html#)
2.) Evaluar ResNet con LayerNorm en vez de BatchNorm
3.) Evaluar SGMCMC con ResNet LayerNorm versus BatchNorm
4.) Evaluar Cuantizacion de ResNet LayerNorm