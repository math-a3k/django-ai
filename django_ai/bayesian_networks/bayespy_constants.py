# -*- coding: utf-8 -*-

DIST_GAUSSIAN = "Gaussian"  # Gaussian(mu, Lambda, \*\*kwargs)
DIST_GAUSSIAN_ARD = "GaussianARD"  # GaussianARD(mu, alpha[, ndim, shape])
DIST_GAMMA = "Gamma"  # Gamma(a, b, **kwargs)
DIST_WISHART = "Wishart"  # Wishart(n, V, **kwargs)
DIST_EXPONENTIAL = "Exponential"  # Exponential(l, **kwargs)
DIST_GAUSSIAN_GAMMA = "GaussianGamma"  # GaussianGamma(*args, **kwargs)
DIST_GAUSSIAN_WISHART = "GaussianWishart"  # GaussianWishart(*args, **kw)
DIST_BERNOULLI = "Bernoulli"  # Bernoulli(p, **kwargs)
DIST_BINOMIAL = "Binomial"  # Binomial(n, p, **kwargs)
DIST_CATEGORICAL = "Categorical"  # Categorical(p, **kwargs)
DIST_MULTINOMIAL = "Multinomial"  # Multinomial(n, p, **kwargs)
DIST_POISON = "Poisson"  # Poisson(l, **kwargs)
DIST_BETA = "Beta"  # Beta(alpha, **kwargs)
DIST_DIRICHLET = "Dirichlet"  # Dirichlet(*args, **kwargs)
# Other Stochastics Nodes (like point estimators) - Should be Stoch Type?
DIST_MIXTURE = "Mixture"  # Mixture(z, node_class, *params[, cluster_plat])
DIST_MAXIMUMLIKELIHOOD = "MaximumLikelihood"  # MaximumLikelihood(arr[, r])
DIST_CONCENTRATION = "Concentration"  # Concentration(D[, regularization])
DIST_GAMMA_SHAPE = "GammaShape"  # GammaShape(**kwargs)
DISTRIBUTION_CHOICES = (
    (DIST_GAUSSIAN, DIST_GAUSSIAN),
    (DIST_GAUSSIAN_ARD, DIST_GAUSSIAN_ARD),
    (DIST_GAMMA, DIST_GAMMA),
    (DIST_WISHART, DIST_WISHART),
    (DIST_EXPONENTIAL, DIST_EXPONENTIAL),
    (DIST_GAUSSIAN_GAMMA, DIST_GAUSSIAN_GAMMA),
    (DIST_GAUSSIAN_WISHART, DIST_GAUSSIAN_WISHART),
    (DIST_BERNOULLI, DIST_BERNOULLI),
    (DIST_BINOMIAL, DIST_BINOMIAL),
    (DIST_CATEGORICAL, DIST_CATEGORICAL),
    (DIST_MULTINOMIAL, DIST_MULTINOMIAL),
    (DIST_POISON, DIST_POISON),
    (DIST_BETA, DIST_BETA),
    (DIST_DIRICHLET, DIST_DIRICHLET),
    (DIST_MIXTURE, DIST_MIXTURE),
    (DIST_MAXIMUMLIKELIHOOD, DIST_MAXIMUMLIKELIHOOD),
    (DIST_CONCENTRATION, DIST_CONCENTRATION),
    (DIST_GAMMA_SHAPE, DIST_GAMMA_SHAPE),
)
DET_DOT = "Dot"  # Dot(*args, **kwargs)
DET_SUM_MULTIPLY = "SumMultiply"  # SumMultiply(*args[, iterator_axis])
DET_ADD = "Add"  # Add(*nodes, **kwargs)
DET_GATE = "Gate"  # Gate(Z, X[, gated_plate, moments])
DET_TAKE = "Take"  # Take(node, indices[, plate_axis])
DET_FUNCTION = "Function"  # Function(function, *nodes_gradients[, shape])
DET_CONCAT_GAUSSIAN = "ConcatGaussian"  # ConcatGaussian(*nodes, **kwargs)
DET_CHOOSE = "Choose"  # Choose(z, *nodes)
DETERMINISTIC_CHOICES = (
    (DET_DOT, DET_DOT),
    (DET_SUM_MULTIPLY, DET_SUM_MULTIPLY),
    (DET_ADD, DET_ADD),
    (DET_GATE, DET_GATE),
    (DET_TAKE, DET_TAKE),
    (DET_FUNCTION, DET_FUNCTION),
    (DET_CONCAT_GAUSSIAN, DET_CONCAT_GAUSSIAN),
    (DET_CHOOSE, DET_CHOOSE),
)
