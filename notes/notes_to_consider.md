

# Behavioral Similarity
Behavioral similarity is often policy-dependent, meaning that the relevance of the similarity can change with different policies. Two states might be similar under one policy but behave differently under another.


# Probability Simplex

The probability simplex over a set X is a mathematical concept used to describe the space of all possible probability distributions over X.

- For example: for a set X with 3 elements, the probability simplex would be a 2-dimensional triangle in 3-dimensional space. Each vertex of the triangle represents a distribution where one of the elements of X has a probability of 1, and the other points on the simplex represent distributions where the probabilities are shared among the elements of X.
- Defined in a high-dimensional space where the dimensions are the probabilities of each outcome, subject to the constraints that these probabilities sum to 1 and each probability is non-negative.
- A standard probability distribution is a particular assignment of probabilities to events or outcomes in a sample space. It is one point in the probability simplex.
- In the example, the dimensions of each point in the triangle will be a discrete standard probability distribution.

# Coupling

In the 1-Wasserstein framework, a coupling between two probability distributions is essentially a joint distribution that has the given distributions as its marginals.

the coupling reflects the least effort needed to align one distribution with another under the defined metric.

In bisimulation metrics, especially in the context of Markov decision processes (MDPs) or other stochastic models, the 1-Wasserstein term with couplings is used to:

- Quantify the similarity between states: Two states are considered similar if for any action, the resulting distributions of next states can be coupled in a way that keeps the average distance (according to the metric of the state space) small.
- Reflect behavioral similarity: This is not just about the static properties of states but about how they behave under the decision-making process, capturing both the dynamics and the outcomes (rewards, next states) of the system.

Let's consider two random variables $X_1$ and $X_2$, where $X_1$ follows a probability distribution $P_1$ on a space $S_1$, and $X_2$ follows $P_2$ on a space $S_2$.

### Original Marginal Distributions

- $X_1 \sim P_1$: $X_1$ has a distribution $P_1$, where $P_1(x)$ defines the probability of $X_1$ taking a value $x$ in $S_1$.
- $X_2 \sim P_2$: $X_2$ has a distribution $P_2$, where $P_2(y)$ defines the probability of $X_2$ taking a value $y$ in $S_2$.

### Coupling Definition

Coupling these two variables involves defining a joint distribution $P$ on the product space $S_1 \times S_2$ such that:

- $P(X_1 = x, X_2 = y)$ is the joint probability distribution of $X_1$ and $X_2$.
- The marginal distributions of $P$ must match the original distributions of $X_1$ and $X_2$:

$$\sum_{y \in S_2} P(X_1 = x, X_2 = y) = P_1(x) \quad \forall x \in S_1$$
$$\sum_{x \in S_1} P(X_1 = x, X_2 = y) = P_2(y) \quad \forall y \in S_2$$


In this coupled system:

1. **Joint Distribution $P$**: Represents how we can simultaneously consider $X_1$ and $X_2$, even if they originated from different spaces or processes.
2. **Marginal Properties**: Ensure that when we consider the behavior of one variable irrespective of the other, it aligns with the variable's original distribution.

### Purpose of Coupling

The main goal here is to analyze the properties of $X_1$ and $X_2$ in a unified way. For instance, in studying convergence, dependence, or to provide a probabilistic bound on their behavior, coupling allows us to mathematically reason about these properties more effectively.

By doing so, coupling provides a powerful method to compare random variables and to analyze their stochastic relationship in a controlled manner that respects their inherent probabilistic laws.
