% Logistic Regression

# Maximum Likelihood Estimation #
* [Source](http://czep.net/stat/mlelr.pdf)

Consider the following dataset:

| Z | a | b |
| - | - | - |
| 0 | 0 | 1 |
| 0 | 0 | 1 |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

* *Z* : random variable with possible outcomes 1 and 0
* total sample size: *M* = 6
* $Z_i$ : binomial random variables where 1 indicates success
* $\mathbf Z = (Z_1, \ldots, Z_6)$
* *N* = 3 populations = (0,1), (0,0), (1,1)
    * distinct combinations of values of the independent variables
* number of observations for each population: *n* = (2, 2, 2)
* $Y_i$ : random variable representing number of successes of *Z* for population *i*
* $y_i$ : number of observed successes for population *i*
* ***y*** = (0, 1, 1)
* probability of success for any given observation in *i*th population: $\pi_i = P(Z_i = 1|i)$
* $\mathbf \pi = (\pi_1, \pi_2, \pi_3)$

Design matrix **X**:

| x0| x1| x2|
| - | - | - |
| 1 | 0 | 1 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

* number of independent variables: *K* = 2
* parameters: $\beta = (\beta_0, \beta_1, \beta_2)$

Equations to solve:
$$
  \log\left(\frac{\pi_i}{1 - \pi_i}\right)=(x_{i0}, x_{i1}, x_{i2}) \cdot \beta, \quad i = 1,\ldots, N
$$

Joint probability density function of dependent variable *Y*:

* $P(Y = y) = P(Y_1 = y_1) \cdot \ldots \cdot P(Y_N = y_N)$
* $Y_i \sim Binomial(n_i, \pi_i)$

$$
  f(y | \beta) = \prod_{i=1}^N \binom{n_i}{y_i}\pi_i^{y_i}(1 - \pi_i)^{n_i - y_i}
$$

* Note, that there is no $\beta$ in $f$ but it indirectly defines $\pi$.
* Likelihood function: $L(\beta | y) = f(y | \beta)$

* $\beta_\max$ : set of parameters for which the probability of the observed data is greatest
$$ \beta_\max = \underset \beta {\operatorname{argmax}} L(\beta | y)$$

# Cross-Entropy #
The cross entropy for the distributions *p* and *q* over a given set is defined as $H(p, q) = \mathbb E_p[-\log q]$.

* Discrete case: $-\sum_x \log q(x)\,p(x)$
* Continuous case: see [wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)

## Cross-entropy error function and logistic regression ##
See [wikipedia](https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression).
With the *logistic function* $g(z) = \frac 1 {1 + e^{-z}}$ and $y\in\{0, 1\}$ the probability of finding $y = 1$ is given by:
$$\phantom.
  q_{y = 1} = \hat y \equiv g(\mathbf w \cdot \mathbf x)
$$

Let $y \in \{0, 1\}$ be the true label and $\hat y$ the estimation.
Having set up the notation $p \in \{y, 1-y\}$ and $q\in \{\hat y, 1 - \hat y\}$,
we can use cross entropy to get a measure for similarity between *p* and *q*:
$$\phantom.
  H(p,q)\ = -\sum_i p_i \log q_i = -y \log \hat y - (1 - y) \log(1- \hat y)
$$

The typical loss function that one uses in logistic regression is computed by taking the average of all cross-entropies in the sample:
$$
  L(\mathbf w) = \frac 1 N \sum_{n=1}^{N} H(p_{n},q_{n})
$$

The *logistic loss* is sometimes called *cross-entropy loss*.
It is also known as *log loss*.

### Further reading ###
* [How is logistic loss and cross-entropy related?](https://math.stackexchange.com/a/1672834)
* [Regression for a ratio or fraction between 0 and 1](https://stats.stackexchange.com/q/29038)
* [cross-entropy error vs classification error vs mean squared error](https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/)
* [What's an intuitive way to think of cross entropy?](https://www.quora.com/Whats-an-intuitive-way-to-think-of-cross-entropy)
