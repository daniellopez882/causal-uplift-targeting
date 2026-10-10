# ADR 0001: Decide on cross-fitted AIPW policy value, not on Qini alone

Status: accepted, 2026-10-09. The decision rule in the fourth bullet was amended by ADR 0005.

## Context

The question the project answers is operational: with a budget of 10, 20 or 30% of customers,
who gets which e-mail, and how many extra conversions does that buy? Qini and AUUC summarise how
well a score ranks one treatment against control over all cut-offs. They do not say what a specific
budgeted, multi-arm assignment is worth, they have no natural unit, and on Hillstrom (conversion
rate 0.6 to 1.3%) their sampling noise is large relative to the differences between learners.

The data come from a randomised test, so the assignment probabilities are known. That makes
off-policy estimators unbiased without modelling assignment.

## Decision

* The headline metric is the value of a budgeted policy, estimated with augmented inverse
  propensity weighting (AIPW) and reported as incremental outcomes per 1,000 contacts and per
  1,000 customers. IPW is reported next to it as a check that does not depend on an outcome model.
* Everything is cross-fitted with 5 folds: learners and the AIPW outcome model are fitted on four
  folds and scored on the fifth, and the budget is applied inside each held-out fold.
* Policies are probability matrices, so random targeting is scored exactly in expectation over its
  randomisation instead of from one random draw.
* Confidence intervals are percentile bootstraps over customers that resample all policies jointly.
  The comparison with random targeting is therefore a paired difference with its own interval, and
  the report only recommends model-based targeting when that interval excludes zero.
* When the response model is among the evaluated learners, every policy also gets a paired AIPW
  difference against the response model's policy at the same budget. Beating random targeting
  shows that a ranking helps; only this second difference shows whether modelling the effect
  helped more than modelling the outcome. It is reported, not used by the decision rule.
* Qini coefficient and AUUC stay in the report as ranking diagnostics, with the same bootstrap.

## Consequences

* Numbers are in business units and comparable across learners, arms and budgets.
* AIPW stays unbiased if the outcome model is wrong, because propensities are known; a Monte Carlo
  test checks this with a deliberately wrong outcome model.
* The bootstrap holds the fitted models fixed. Intervals cover evaluation noise but not the variance
  from refitting the learners, so they are narrower than a full repeated-split procedure. The README
  and the report say so.
* Picking the best of several policies on the same data and reporting its estimate is optimistic.
  As first written, the rule did not correct for it; ADR 0005 moved the test to a simultaneous
  interval.

## Alternatives rejected

* Qini/AUUC as the decision metric: no budget, no arm choice, no unit.
* A single train/test split: wastes four fifths of an already small number of conversions.
* Normal-approximation intervals per policy: they ignore the pairing between policies scored on the
  same customers, which is where most of the precision of a comparison comes from.
* Refitting every learner inside each bootstrap replicate: about 1,000 times the compute of the
  current benchmark for an interval that is still conditional on the same 64,000 customers.
