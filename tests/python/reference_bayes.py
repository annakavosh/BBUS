"""Verification-only pure-Python Bayesian reference implementation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ReferenceCalculation:
    prior: float
    likelihood_ratio: float
    prior_odds: float
    posterior_odds: float
    posterior: float
    belief_delta: float


def _probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("probability must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError("probability must be in [0, 1]")
    return value


def _extended_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def log_likelihood_ratio(
    p_e_given_h: float, p_e_given_not_h: float
) -> float:
    p_e_given_h = _probability(p_e_given_h)
    p_e_given_not_h = _probability(p_e_given_not_h)
    log_h = -math.inf if p_e_given_h == 0.0 else math.log(p_e_given_h)
    log_not_h = (
        -math.inf
        if p_e_given_not_h == 0.0
        else math.log(p_e_given_not_h)
    )
    return log_h - log_not_h


def exact_binary_reference(
    prior: float, p_e_given_h: float, p_e_given_not_h: float
) -> ReferenceCalculation:
    prior = _probability(prior)
    p_e_given_h = _probability(p_e_given_h)
    p_e_given_not_h = _probability(p_e_given_not_h)
    scale = max(p_e_given_h, p_e_given_not_h)
    if scale == 0.0:
        raise ArithmeticError("observation has zero likelihood")

    hypothesis_mass = prior * (p_e_given_h / scale)
    not_hypothesis_mass = (1.0 - prior) * (p_e_given_not_h / scale)
    evidence_mass = hypothesis_mass + not_hypothesis_mass
    if evidence_mass == 0.0:
        raise ArithmeticError("observation has zero probability under the prior")

    posterior = hypothesis_mass / evidence_mass
    return ReferenceCalculation(
        prior=prior,
        likelihood_ratio=_extended_ratio(p_e_given_h, p_e_given_not_h),
        prior_odds=_extended_ratio(prior, 1.0 - prior),
        posterior_odds=_extended_ratio(hypothesis_mass, not_hypothesis_mass),
        posterior=posterior,
        belief_delta=posterior - prior,
    )


def log_odds_reference(
    prior: float, log_ratio: float
) -> ReferenceCalculation:
    prior = _probability(prior)
    if math.isnan(log_ratio):
        raise ValueError("log likelihood ratio must not be NaN")
    if prior == 0.0 and log_ratio == math.inf:
        raise ArithmeticError("infinite likelihood ratio conflicts with prior")
    if prior == 1.0 and log_ratio == -math.inf:
        raise ArithmeticError("infinite likelihood ratio conflicts with prior")

    likelihood_ratio = _safe_exp(log_ratio)
    prior_odds = _extended_ratio(prior, 1.0 - prior)
    if prior == 0.0:
        return ReferenceCalculation(
            prior, likelihood_ratio, prior_odds, 0.0, 0.0, 0.0
        )
    if prior == 1.0:
        return ReferenceCalculation(
            prior, likelihood_ratio, prior_odds, math.inf, 1.0, 0.0
        )

    log_prior_odds = math.log(prior) - math.log1p(-prior)
    log_posterior_odds = log_prior_odds + log_ratio
    posterior_odds = _safe_exp(log_posterior_odds)
    if log_posterior_odds >= 0.0:
        posterior = 1.0 / (1.0 + math.exp(-log_posterior_odds))
    else:
        exponential = math.exp(log_posterior_odds)
        posterior = exponential / (1.0 + exponential)

    return ReferenceCalculation(
        prior=prior,
        likelihood_ratio=likelihood_ratio,
        prior_odds=prior_odds,
        posterior_odds=posterior_odds,
        posterior=posterior,
        belief_delta=posterior - prior,
    )
