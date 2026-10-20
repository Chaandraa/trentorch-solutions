import math


def is_valid_pmf(probabilities):
    """
    probabilities: list/tuple of floats, one per outcome of a discrete
        random variable.

    Returns:
        True if `probabilities` is a valid probability mass function:
        every value is non-negative AND they sum to 1 (within 1e-9).
    """
    # Axiom 1: Check if any probability is negative
    for p in probabilities:
        if p < 0:
            return False

    # Axiom 2: Check if the sum equals 1 within the specified float tolerance
    if abs(sum(probabilities) - 1.0) >= 1e-9:
        return False

    return True


def expected_value_discrete(outcomes, probabilities):
    """
    outcomes: list/tuple of numbers, the values a discrete random
        variable X can take.
    probabilities: list/tuple of floats, P(X = outcomes[i]) for each i,
        same length as outcomes.

    Returns:
        E[X] = sum(outcome * probability), from Theory.
    """
    # Computes the weighted average by pairing outcomes and probabilities together
    return sum(x * p for x, p in zip(outcomes, probabilities))


def binomial_pmf(n, p, k):
    """
    n: number of independent trials (positive int)
    p: probability of success on each trial (float in)
    k: number of successes to compute the probability of (int in [0, n])

    Returns:
        P(X = k) for X ~ Binomial(n, p), from Theory.
    """
    # Computes C(n, k) * p^k * (1-p)^(n-k) securely without arithmetic leaks
    combinations = math.comb(n, k)
    return combinations * (p**k) * ((1 - p) ** (n - k))


def uniform_pdf(x, a, b):
    """
    x: point to evaluate the density at
    a, b: the interval's lower and upper bounds (a < b)

    Returns:
        The value of the continuous Uniform(a, b) density at x: 1/(b-a)
        inside [a, b], and 0 outside it.
    """
    # Evaluates piecewise probability distribution limits
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0
