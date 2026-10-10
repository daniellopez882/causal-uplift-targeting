# ADR 0005: Test the chosen policy on simultaneous intervals, against two no-model policies

Status: accepted, 2026-10-10. Amends the decision rule of ADR 0001.

## Context

The first full benchmark (2026-10-09) used the rule from ADR 0001: at each budget, take the learner
policy with the highest AIPW value and recommend targeting when its paired gain over random
targeting has a 95% percentile interval above zero. Reviewing those results showed two problems.

* Selection. The winner is the largest of nine policy estimates on the same customers. Its own
  percentile interval would cover its gain 95% of the time had it been named in advance; after
  picking the largest estimate, it covers less often, and a lower bound just above zero is exactly
  what that selection produces by chance.
* A weak reference. Random targeting sends a random e-mail, so half of the contacts get the Womens
  e-mail, whose average effect on conversion is well below the Mens one. Sending the better e-mail
  to a random share of customers needs no model, and it is the comparison a marketing team would
  actually face.

## Decision

* For every learner policy, the paired gain over each no-model policy also gets a simultaneous 95%
  interval: a max-t band built from the same bootstrap replicates (each gain standardised by its
  bootstrap standard deviation, critical value from the largest standardised deviation per
  replicate). It covers the gains of all learner policies at a budget at once, so it also covers
  the one picked after looking at them.
* A second no-model policy, "best arm at random", is scored at every budget: random customers, all
  sent the treatment with the highest AIPW average effect at that budget and cost. The arm is
  chosen on the rows used for scoring, which can only flatter this reference.
* Targeting is recommended only when the winner's simultaneous lower bounds against both
  no-model policies are above zero. Requiring both is an intersection-union test, so the pair
  needs no further adjustment.
* The verdict under the rule as first written is still computed (`first_rule` in
  `causal_uplift.report.recommend`) and published next to the revised one, because the revision
  was made after the first results had been seen.

## Consequences

* The test of the recommended policy accounts for its selection. Its point estimate does not: the
  value reported for the chosen policy is still optimistic.
* Bands are wider than single-policy intervals by an amount that depends on how correlated the
  candidates are. Policies that contact mostly the same customers cost little width; the T-learner
  and its scikit-uplift twin, which are identical, cost nothing.
* With unequal arm costs, the reference arm is chosen per budget by effect times the share the
  budget pays for, so it can change between budgets.
* Single-treatment data get no second reference: the best arm at random is then random targeting.

## Alternatives rejected

* Choose the policy on one half of the customers and estimate its value on the other half. That is
  unbiased, but it halves an already small number of conversions for both steps.
* Bonferroni over the candidates. Valid, but it treats nine highly correlated estimates as
  independent and gives wider intervals than the max-t band for the same guarantee.
* Keep the percentile rule and only say that the winner is optimistic, as the first README did.
  The warning did not change any verdict, and at least one verdict depends on it.
* Make the best arm at random the only reference. Random targeting is kept because it is the
  baseline the original question was posed against, and the first rule used it.
