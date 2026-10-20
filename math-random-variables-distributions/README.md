# Random Variables & Distributions

Beginner | probability | foundations

### The problem, from first principles

**Random variables**

Every ML model that outputs a probability, a classifier's softmax, a language model's next-token distribution, a VAE's latent code, is treating some quantity as a **random variable**: a variable whose value isn't fixed, but drawn from a distribution. A **discrete** random variable takes one of a countable set of values (a class label, a token id, a die roll); its distribution is a **probability mass function (PMF)**, a table of `P(X = x)` for each possible `x`. A **continuous** random variable takes any value in a range (a pixel intensity, a sensor reading); it doesn't have a PMF at all, a single exact value has probability zero, instead it has a **probability density function (PDF)**, covered in the next question (`01-random-variables`). This question stays entirely in the discrete case and nails down the two things every valid PMF must satisfy, plus the single most useful summary of a random variable: its **expected value**.

**PMFs & PDFs**

`01-random-variables` introduced the PMF for a discrete random variable: a table of `P(X = x)` for each possible value. But most real quantities a model touches, a pixel value, an activation, a continuous latent variable, aren't discrete and a PMF literally cannot describe them: for a continuous variable, the probability of hitting any _exact_ real number is zero (there are infinitely many of them to spread the probability over). A **probability density function (PDF)** solves this by describing probability as area under a curve instead of height at a point, `P(a <= X <= b)` is the integral of the density between `a` and `b`, not a single lookup. This question implements one canonical example of each: the discrete **Binomial** PMF (counting successes over `n` trials) and the continuous **Uniform** PDF (equally likely anywhere in an interval).

### From theory to code

**Random variables**

Implement `is_valid_pmf(probabilities)`, checking the two axioms every PMF must satisfy, then `expected_value_discrete(outcomes, probabilities)`, computing `E[X]`. The signatures and docstrings are already in the editor.

**PMFs & PDFs**

Implement `binomial_pmf(n, p, k)`, returning `P(X = k)` for a `Binomial(n, p)` random variable, then `uniform_pdf(x, a, b)`, returning the constant density of a `Uniform(a, b)` random variable at point `x`. The signatures and docstrings are already in the editor.

### Constraints

**Random variables**

- `probabilities` is a list/tuple of floats.
- `outcomes` and `probabilities` are the same length, and `probabilities[i]` is `P(X = outcomes[i])`.
- "Sums to 1" means within `1e-9` of exactly 1, to tolerate floating-point roundoff.

**PMFs & PDFs**

- `n` is a positive integer, `k` is an integer in `[0, n]`, `p` is a float in `[0, 1]`.
- `a < b` for `uniform_pdf`; `x` can be any real number, including outside `[a, b]`.

### Hints

**Random variables**

<details>
<summary>Hint 1</summary>

`is_valid_pmf` checks two separate conditions and both must hold: every probability is `>= 0`, and they sum to (approximately) `1`.

</details>

<details>
<summary>Hint 2</summary>

`expected_value_discrete` is a single weighted sum, pair each outcome with its probability and add up `outcome * probability`, exactly what the formula in Theory says.

</details>

**PMFs & PDFs**

<details>
<summary>Hint 1</summary>

`binomial_pmf` is `C(n, k) * p**k * (1-p)**(n-k)` directly, `math.comb(n, k)` gives you the binomial coefficient without writing factorial by hand.

</details>

<details>
<summary>Hint 2</summary>

`uniform_pdf` is a single `if`: return `1 / (b - a)` when `a <= x <= b`, and `0.0` otherwise, the density is exactly constant everywhere inside the interval and exactly zero everywhere outside it.

</details>
