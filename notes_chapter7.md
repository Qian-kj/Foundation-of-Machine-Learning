# Chapter 7: AdaBoost and Ensemble Methods

## 1. Aim
Combining several predictors to create a more accurate one.

### Objective1
Show how AdaBoost can reduce the empirical error of boosting and its relationship with some known algorithms.

### Objective2
Generalization properties of AdaBoost based on VC-dimension and the notion of margin.

### Objective3
Game-theoretic interpretation of AdaBoost for properties and equivalence analysis between weak learning assumption and a separability condition.

## 2. Weak Learning
For non-trivial learning tasks, it is difficult to devise an accurate algorithm satisfying the strong PAC-learning requirements. A weak learner is an algorithm that returns hypotheses slightly better than random guessing.

## 3. Ideas of Boosting
Transforming Weak to Strong Learners: Boosting techniques use a weak learning algorithm to create a strong learner, achieving high accuracy. Ensemble methods, by combining several weak classifiers, enhance overall predictive performance.

### Definition of Weak Learning
A concept class $C$ is weakly PAC-learnable if there exists an algorithm $A$, a constant $\gamma > 0$, and a polynomial function $\text{poly}(\cdot, \cdot, \cdot)$ such that for any $\delta > 0$, for all distributions $D$ on $X$, and for any target concept $c \in C$, the following holds for any sample size $m$:

$$
\Pr_{S \sim D^m}[ \Pr_{(x,y) \sim D}[ h_S(x) \neq y ] \leq 1/2 - \gamma] \geq 1 - \delta
$$

where $h_S$ is the hypothesis returned by algorithm $A $when trained on sample $S$,

### Components of Boosting
- **Base Classifiers**: Weak hypotheses or models that are combined to form a more accurate predictor.
- **Combining Base Classifiers**: The strategy to combine these base classifiers to improve overall accuracy is the core of boosting methods.

## 4. AdaBoost

### 4.1 Introduction

### Process of AdaBoost (Pseudocode)

#### Initialization
Start with uniform weights on the training examples.

#### Iteration
For each round $t$:
- Train a weak learner on the weighted data.
- Compute the weighted error rate $\epsilon_t$,
- Update the weights: Increase weights for misclassified examples and decrease for correctly classified ones.

#### Final Hypothesis
Combine the weak classifiers using weights proportional to their accuracy.

### Illustration
![[Pasted image 20240614093535.png]]
### Key Components
- **Base Classifier Set $H$**: This is the set from which base classifiers are selected. Base classifiers map from the input space $X$ to the output space $\{-1, +1\}$,
- **Labeled Sample $S$**: A sample $S$ consists of pairs $(x_i, y_i)$, where $x_i \in X$ and $y_i \in \{-1, +1\}$,
- **Initial Distribution $D_1$**: Initially, the weight distribution $D_1$ over the training examples is uniform, meaning each example is equally likely to be chosen.

### Algorithm Steps

#### Initialize Weights
Each example has an equal weight initially.

#### Iterate for $T$ Rounds
For each boosting round $t$:
- Train Base Classifier $h_t$: Select the classifier $h_t$ that minimizes the weighted error.
- Compute Error $\epsilon_t$: Compute the weighted error of the classifier.
- Compute Classifier Weight $\alpha_t$: Compute the weight for classifier $h_t$,
- Update Weights: Increase weights for misclassified examples and decrease for correctly classified ones.

### Weight Update
The weight update is performed using the formula:

$$
D_{t+1}(i) = \frac{D_t(i) \exp(-\alpha_t y_i h_t(x_i))}{Z_t}
$$

where $Z_t$ is a normalization factor ensuring $D_{t+1}$ sums to 1.

### Final Classifier
Combine the base classifiers using their weights:

$$
f_T = \sum_{t=1}^{T} \alpha_t h_t
$$

### Weight Distribution $D_{t+1}(i)$
This represents the weight of the $i$-th example after $t+1$ rounds. The exponential term $e^{-y_i f_t(x_i)}$ indicates how the prediction $f_t(x_i)$ (a weighted sum of all base classifiers up to round $t$) aligns with the true label $y_i$,

### Normalization Factor $Z_t$
$Z_t$ ensures that the distribution $D_t$ sums to 1 over all examples. This is necessary for $D_t$ to be a valid probability distribution.

## Recursive Expansion
By recursively expanding weight distribution using the definition of the distribution over the point $x_i$, we get:

$$
D_{t+1}(i) = \frac{D_t(i) \exp(-\alpha_t y_i h_t(x_i))}{Z_t}
$$

Here, $D_t(i)$ is the weight of the $i$-th example at the $t$-th round. The exponential term $e^{-\alpha_t y_i h_t(x_i)}$ adjusts the weight based on the classification performance of the base classifier $h_t$,

## Generalization

### Base Classifier Generalization
Allows the use of a broader range of weak learning algorithms, not limited to those that minimize weighted error.

### Extended Range of Outputs
Enables base classifiers to provide more detailed predictions, which can capture confidence levels or probabilities.

### Flexible Coefficients
Adapts the weight of each classifier in the final ensemble more flexibly, potentially improving the boosting process.

### Non-Binary Hypotheses
Uses both the sign and magnitude of the base classifier's output, enhancing the richness of information used to form the final prediction.

## 4.2 Bound on the Empirical Error

### Empirical Error
Empirical error decreases exponentially with the number of boosting rounds.

### Theorem 7.2

#### Theorem Breakdown
Empirical Error: This is the error of the final classifier $f$ on the training set $S$,

#### Exponential Decrease
- The empirical error decreases exponentially with the number of boosting rounds $T$,
- The rate of decrease depends on the terms $\epsilon_t$,

#### Condition on $\gamma$
- If $\gamma \leq (1/2 - \epsilon_t)$ holds for all rounds $t$, the bound simplifies to $\exp(-2\gamma^2 T)$,
- This shows that if the weighted error $\epsilon_t$ of each classifier is consistently less than 0.5 by at least $\gamma$, the empirical error decreases rapidly.

#### Proof Initial Bound on Misclassification
Use the inequality:

$$
\mathbb{I}[h(x) \neq y] \leq \exp(-y f(x))
$$

to bound the indicator function by an exponential function. This allows the empirical error to be bounded by an exponential function of the cumulative prediction:

$$
\sum_{i=1}^{m} D_{t+1}(i) \leq \prod_{t=1}^{T} Z_t
$$

#### Recursive Weight Update
Express the bound in terms of the product of normalization factors $Z_t$:

$$
\prod_{t=1}^{T} Z_t \leq \exp(-2\gamma^2 T)
$$

#### Normalization Factor $Z_t$
The normalization factor is expressed as:

$$
Z_t = \sum_{i=1}^{m} D_t(i) \exp(-\alpha_t y_i h_t(x_i))
$$

Simplifying this using the optimal choice of $\alpha_t$:

$$
Z_t = 2\sqrt{\epsilon_t(1 - \epsilon_t)}
$$

#### Product of Normalization Factors
The product of these normalization factors over $T$ rounds gives:

$$
\prod_{t=1}^{T} Z_t \leq \exp(-2\gamma^2 T)
$$

#### Upper Bound on Empirical Error
Combining the results, we get:

$$
\sum_{i=1}^{m} \mathbb{I}[f(x_i) \neq y_i] \leq \exp(-2\gamma^2 T)
$$

#### Simplified Bound with $\gamma$
If $\gamma \leq (1/2 - \epsilon_t)$, then:

$$
\sum_{i=1}^{m} \mathbb{I}[f(x_i) \neq y_i] \leq \exp(-2\gamma^2 T)
$$

#### Example
Suppose you have a dataset with 100 examples and you're using AdaBoost with decision stumps (simple decision trees) as weak learners. After running the algorithm for $T$ rounds, you compute the weighted errors $\epsilon_t$ for each round. If these errors are consistently around 0.1, then:

- The bound on the empirical error is:

$$
\sum_{i=1}^{m} \mathbb{I}[f(x_i) \neq y_i] \leq \exp(-2\gamma^2 T)
$$

- If $T = 10$, then:

$$
\sum_{i=1}^{m} \mathbb{I}[f(x_i) \neq y_i] \leq \exp(-2 \cdot 0.1

^2 \cdot 10) = \exp(-0.2) \approx 0.82
$$

This shows that with enough rounds, the empirical error can be reduced significantly.

## 4.3 Relationship with Coordinate Descent

### Coordinate Descent
An optimization algorithm that minimizes a function by successively performing one-dimensional minimizations along coordinate directions. At each iteration, it selects a direction and a step size along that direction to minimize the objective function.

### Objective Function for AdaBoost
Consider an ensemble function $f$ of the form:

$$
f(x) = \sum_{j=1}^{T} \alpha_j h_j(x)
$$

where $\alpha_j \geq 0$ are the coefficients and $h_j$ are the base classifiers. The objective function $F$ that AdaBoost aims to minimize is:

$$
F(\alpha) = \sum_{i=1}^{m} \exp(-y_i f(x_i))
$$

### Convexity and Differentiability
- $F$ is a convex function of $\alpha $since it is a sum of convex functions.
- $F$ is differentiable because the exponential function is differentiable.

### Coordinate Descent in AdaBoost
In each round $t$ of AdaBoost, a direction $e_k$ and a step size $\eta $are selected to update the coefficient vector $\alpha$,

$$
\alpha_{t+1} = \alpha_t + \eta e_k
$$

### Ensemble Function Update
Let $g_t$ denote the ensemble function at iteration $t$:

$$
g_t = g_{t-1} + \eta h_k
$$

The coordinate descent update $g_t = g_{t-1} + \eta h_k$ matches the AdaBoost update.

### Selection of Direction and Step Size
- The direction $e_k$ is selected to maximize the absolute value of the derivative of $F$:

$$
e_k = \arg\max_k \left| \frac{\partial F}{\partial \alpha_k} \right|
$$

where,

$$
\frac{\partial F}{\partial \alpha_k} = -\sum_{i=1}^{m} y_i h_k(x_i) \exp(-y_i f(x_i))
$$

- The step size $\eta$ is chosen to minimize $F$ along the selected direction:

$$
\eta = \arg\min_{\eta} F(\alpha + \eta e_k)
$$

- The step size $\eta$ is determined by setting the derivative of $F$ with respect to $\eta$ to zero:

$$
\frac{\partial F}{\partial \eta} = 0
$$

## 4.4 Practical Use

### Choice of Base Classifiers
Decision stumps (depth-one decision trees) are frequently used with AdaBoost due to their simplicity and efficiency.

### Computational Complexity
- Pre-sorting each component: $O(m \log m)$ per component, leading to $O(m N \log m)$ total.
- Finding the best threshold: $O(m)$ per component.
- Total complexity for $T$ rounds: $O(m N \log m + m N T)$,

### Performance and Limitations
Boosting stumps can be very effective but are not universally optimal, particularly for problems that are not linearly separable (e.g., XOR problem).

### XOR Problem
The XOR function is a binary operation that outputs true (or 1) only when the inputs differ (i.e., one is true and the other is false):

- (0, 0) labeled as 0
- (0, 1) labeled as 1
- (1, 0) labeled as 1
- (1, 1) labeled as 0

## 5. Theoretical Results

### VC-Dimension-Based Analysis
Provides an upper bound on the VC-dimension of the hypothesis set used by AdaBoost, suggesting potential overfitting with large $T$,

### L1-Geometric Margin
Defines and uses geometric margins to derive learning guarantees for AdaBoost.

### Margin-Based Analysis
Derives generalization bounds based on margins, indicating that the empirical margin loss decreases exponentially with the number of boosting rounds.

### Margin Maximization
Discusses the relationship between AdaBoost and margin maximization, showing that AdaBoost seeks to maximize the margin, though it may not always achieve the optimal margin.

### Game-Theoretic Interpretation
Frames AdaBoost in a game-theoretic context, showing the equivalence between the weak learning assumption and the separability condition.

## 5.1 VC-Dimension-Based Analysis

### Recall VC Dimension
The VC dimension is a measure of the capacity or complexity of a set of functions or classifiers. It represents the largest number of points that can be shattered by the hypothesis class.

For a class of linear classifiers in 2D (lines in a plane), the VC dimension is 3. This means any three points can be shattered by lines, but not necessarily four points.

### Final Hypothesis Class $F_T$

### Theorem on VC-Dimension

### Implications
- **Growth of VC Dimension**: The upper bound on the VC dimension grows as $O(dT \log T)$, As the number of rounds $T$ increases, the capacity of the hypothesis class $F_T$ increases, potentially leading to overfitting.
- **Generalization Error**: Empirical observations show that the generalization error of AdaBoost often decreases with more rounds of boosting, so margin-based analyses are needed.

## 5.2 L1-Geometric Margin

### Geometric Margin
Definition: A measure of the distance between data points and the decision boundary of a classifier.

### Confidence Margin
For a real-valued function $f $at a point $x $with label $y$, the confidence margin is defined as $y f(x)$,

### Geometric Margin for SVMs
In SVMs, the geometric margin is a lower bound on the confidence margin of a linear hypothesis with a normalized weight vector $w$, where $\|w\|_2 = 1$,

### L1-Norm
The Manhattan norm to measure the "size" or "length" of a vector.

### Function Representation

### Definition 7.3

### Differences from SVM
Geometric margin differs from that used in the SVMs by the norm applied to the weight vector.

### Implications

### Normalized Function
The normalized version of the function $f$ returned by AdaBoost is denoted as:

$$
\bar{f}(x) = \frac{f(x)}{\|f\|_1}
$$

### Confidence Margin and L1-Geometric Margin
For a correctly classified point $x$ with label $y$, the confidence margin of $\bar{f}$ at $x$ coincides with the L1-geometric margin of $f$:

$$
y \bar{f}(x) = \frac{y f(x)}{\|f\|_1}
$$

### Convex Combination
Since the coefficients $\alpha_t$ are non-negative, $\rho f(x)$ is a convex combination of the base hypothesis values $h_t(x)$, If the base hypotheses $h_t$ take values in $[-1, +1]$, then $\rho f(x)$ is in $[-1, +1]$.

## 5.3 Margin-Based Analysis

### Key Concepts

#### Margins
- The margin of a classifier at a point $(x_i, y_i)$ is the product $y_i f(x_i)$, where $f(x_i)$ is the value of the classifier before taking the sign.
- The margin provides a measure of confidence in the prediction; larger margins indicate greater confidence.

#### Rademacher Complexity
- Rademacher complexity is a measure of the richness of a hypothesis set $H$ of real-valued functions.
- For a hypothesis set $H$, the empirical Rademacher complexity $\hat{R}_S(H)$ is defined over a sample $S$,

#### Convex Hull
- The convex hull $\text{conv}(H)$ of a set $H$ is the smallest convex set that contains all functions in $H$,
- For a hypothesis set $H$ of real-valued functions, $\text{conv}(H)$ includes all convex combinations of functions in $H$,

### Rademacher Complexity of Convex Linear Ensembles

#### Definition
For any hypothesis set $H$ of real-valued functions, its convex hull is defined by:

$$
\text{conv}(H) = \left\{ \sum_{i=1}^{n} \alpha_i h_i : h_i \in H, \alpha_i \geq 0, \sum_{i=1}^{n} \alpha_i = 1 \right\}
$$

#### Lemma 7.4

### Implication

#### Conv(H)
Consider a hypothesis set $H = \{h_1, h_2\}$, where $h_1$ and $h_2$ are two different functions. The convex hull $\text{conv}(H)$ includes all functions that can be written as $\mu_1 h_1 + \mu_2 h_2$ where $\mu_1 \geq 0$, $\mu_2 \geq 0$, and $\mu_1 + \mu_2 \leq 1$, This means any function that

 is a weighted combination of $h_1$and $h_2$ with non-negative weights summing to at most 1 is in the convex hull.

### Lemma 7.4
If the original hypothesis set $H$ has a certain capacity to fit random noise (as measured by its empirical Rademacher complexity), then forming convex combinations of functions from $H $does not increase this capacity.

## Margin-Based Generalization Bounds

### Theorem 7.7 (Generalization Bound)

### Corollary 7.5 (Ensemble Rademacher Margin Bound)

### Corollary 7.6 (Ensemble VC-Dimension Margin Bound)

### Implication

#### Theorem 7.7
Provides a bound on the empirical margin loss, showing it decreases exponentially with the number of boosting rounds, as long as the weak classifiers' errors are less than 0.5.

#### Corollary 7.5
Uses Rademacher complexity to provide a bound on the true risk, incorporating empirical margin loss and complexity measures.

#### Corollary 7.6
Uses VC-dimension to provide a bound on the true risk, incorporating empirical margin loss and capacity measures of the hypothesis set.

## Generalization Error and Margin

### Empirical Margin Loss
Positive Edge (γ): Indicates consistent improvement over random guessing, leading to zero empirical margin loss for sufficiently large $T$,

### L1-Geometric Margin
AdaBoost achieves an L1-geometric margin of $γ$, which can be as large as half the maximum geometric margin for a separable dataset.

### Generalization Error
Decreases as the geometric margin increases, explaining the continued reduction in error even after achieving zero training error.

### Maximum Margin
AdaBoost does not always reach the maximum possible L1-geometric margin and may settle for a smaller margin in practice.

## 5.4 Margin Maximization

### Definition
The maximum margin for a linearly separable sample $S = \{(x_1, y_1), \ldots, (x_m, y_m)\}$ is given by:

$$
\gamma = \max_{\alpha} \min_{i} y_i (\alpha \cdot x_i)
$$

where $\alpha $is the weight vector.

### Optimization Problem

#### Resulting Linear Program (LP)
This is a linear programming problem, a convex optimization problem with a linear objective function and linear constraints.

#### Solver
There are various methods for solving large linear programming problems, including the Simplex method, Interior-point methods, and Special-purpose solutions.

### Practical Implications
Despite theoretical guarantees, margin theory alone is insufficient to explain performance. In many cases, AdaBoost may outperform the LP solution despite its suboptimal margin.

## 5.5 Game-Theoretic Interpretation

### Aim
Using von Neumann's minimax theorem to draw connections between the maximum margin, the optimal edge, and the weak learning condition.

### Edge of a Base Classifier
This measures the performance of the base classifier $h_t $relative to random guessing under the distribution $D$,

### Weak Learning Condition
AdaBoost's weak learning condition requires that there exists a $γ > 0$ such that for any distribution $D$ over the training sample and any base classifier $h_t$, the edge $γ_t(D)$ is at least $γ$:

$$
\gamma_t(D) = \mathbb{E}_{(x,y) \sim D}[y h_t(x)] \geq \gamma
$$

### Definition 7.9

### Zero-Sum Game
In Boosting, the row player selects training instances, and the column player selects base classifiers.

### Definition 7.10

### Mixed Strategy
A mixed strategy for the row player is a distribution $D $over the training points. A mixed strategy for the column player is a distribution over the base classifiers, derived from a non-negative vector $\alpha$,

### Theorem 7.11 von Neumann's Minimax Theorem
This equality shows that there exists a mixed strategy for each player such that the expected loss for one is the same as the expected payoff for the other.

### AdaBoost as a Zero-Sum Game

### Formula

### The Equivalence
Relates the maximum margin and the best possible edge.

### 5.6 Implications

#### Weak Learning Condition and Margin
The weak learning condition ($γ^* > 0 $) implies $ρ^* > 0$, indicating the existence of a classifier with a positive margin.

#### Algorithmic Goals
AdaBoost aims to achieve a non-zero margin, although it may not always achieve the optimal margin.

#### Strength of Weak Learning Assumption
While appearing weak, the weak learning assumption is strong as it implies linear separability with a margin $2γ^* > 0$, a condition not often met in practical datasets.

## L1-Regularization

### Motivation Issues
- When the training sample is not linearly separable, AdaBoost may not achieve a positive edge, meaning the weak learning condition does not hold.
- Even if AdaBoost does achieve a positive edge, it may be very small ($γ$ is very small). This can cause the algorithm to focus excessively on hard-to-classify examples, leading to large mixture weights for some base classifiers.

### Consequences
- This concentration on a few hard examples can result in a poor overall performance, as a few base classifiers with large weights dominate the final ensemble.
- The resulting classifier may heavily rely on these few classifiers, leading to overfitting and reduced generalization performance.

### Solutions: L1-Regularized AdaBoost

#### Early-Stopping
One method to prevent these issues is early-stopping, which involves limiting the number of boosting rounds $T$,

#### L1-Regularization
Another effective method is to control the magnitude of the mixture weights by incorporating a regularization term into the objective function of AdaBoost. The regularization term is based on the L1-norm of the weight vector, leading to what is known as L1-regularized AdaBoost.

### Objective Function Explanation
The term $\sum_{i=1}^{m} \exp(-y_i f(x_i))$ is the same as the original AdaBoost objective function, representing the empirical risk. The term $λ \| \alpha \|_1$ is the regularization term, with $λ$ being a regularization parameter that controls the trade-off between fitting the training data and keeping the weights small.

### Optimization
The objective function $G $is convex, allowing for efficient optimization methods such as coordinate descent to find the optimal weights.

## Generalization Guarantees

### Inequality 1
True risk $\leq$ Empirical margin loss + Rademacher Complexity + Confidence Terms

### Inequality 2
For $ρ > 1$ The bound trivially holds because the first term on the right-hand side is zero.

### Hölder’s Inequality
Let $p$ and $q$ be conjugate exponents, meaning that they satisfy:

$$
\frac{1}{p} + \frac{1}{q} = 1
$$

For any sequences $\{a_i\}$ and $\{b_i\}$, Hölder’s Inequality states that:

$$
\sum_{i} |a_i b_i| \leq \left( \sum_{i} |a_i|^p \right)^{1/p} \left( \sum_{i} |b_i|^q \right)^{1/q}
$$

### Using Hölder’s Inequality

### Inequality 3
This form of the bound uses the general upper bound $\sum_{i} \exp(a_i) \leq m \exp \left( \frac{1}{m} \sum_{i} a_i \right)$ for all $u \in \mathbb{R}$,

### Inequality 4
This bound is useful for deriving an optimal strategy for selecting the weights $\alpha$ and the parameter $ρ$,

## Optimization

### Selecting Weights $\alpha $
The minimization with respect to $ρ$ does not lead to a convex optimization problem due to the interdependence of the second and third terms. Instead, $ρ$ is treated as a free parameter, typically determined via cross-validation.

### Optimization for $\alpha $
The primary focus is on minimizing the empirical risk. The bound suggests selecting $\alpha $as the solution to the following problem:

$$
\min_{\alpha} \sum_{i=1}^{m} \exp(-y_i f(x_i)) + λ \| \alpha \|_1
$$

### Lagrange Formulation
We want to find the weights $\alpha$ that minimize the empirical risk of the classifier while also satisfying the constraint on the L1-norm of $\alpha$, So,

$$
\min_{\alpha} \sum_{i=1}^{m} \exp(-y_i f(x_i)) \quad \text{s.t.} \quad \| \alpha \|_1 \leq C
$$

## Implication
Controlling the L1-norm of the weight vector can achieve a better generalization error bound compared to the standard AdaBoost when the data is not linearly separable or the edge is very small.

## 6. Takeaways
- **Simplicity**: AdaBoost is simple to implement and computationally efficient.
- **Theoretical Foundations**: Supported by robust theoretical analysis, although it doesn’t always achieve maximum margin, indicating potential for refined theoretical insights.
- **Parameter Selection**: Critical parameters include the number of boosting rounds (T) and the choice of base classifiers.
- **Noise Sensitivity**: Sensitive to noise, as it increases the weight of misclassified examples. Regularization techniques like L1-regularization and alternative loss functions (e.g., logistic loss) can mitigate this.
- **Outlier Detection**: AdaBoost’s propensity to assign large weights to hard-to-classify examples can be used for detecting outliers.
- **Base Classifiers**: Decision stumps are a common choice due to their simplicity and speed.
- **Computational Complexity**: Efficient in practice.
- **Handling Non-Separable Data**: Regularization methods and less aggressive loss functions help in managing non-separable data.
