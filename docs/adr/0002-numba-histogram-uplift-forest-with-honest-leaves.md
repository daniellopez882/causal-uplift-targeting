# ADR 0002: Grow the uplift forest on binned features in numba, with honest leaves

Status: accepted, 2026-10-09

## Context

The project needs its own divergence-split uplift trees (KL and Euclidean, Rzepakowski and
Jaroszewicz 2012), with honest estimation and support for more than one treatment arm.
scikit-learn trees cannot use a custom split criterion. A pure Python tree is too slow for a forest
of a few hundred trees on 51,000 training rows per fold, refitted 5 times per outcome in the
benchmark.

## Decision

* Features are quantile-binned once per fit into at most 32 uint8 codes. The split search for one
  feature is one pass that fills an (n_bins x n_arms) histogram of counts and outcome sums, then a
  scan over thresholds, so all thresholds of a feature cost O(rows + bins x arms).
* The tree is grown depth-first in a single `numba.njit(nogil=True)` kernel, so the forest grows
  trees in parallel threads without the GIL. Kernels are cached on disk.
* Node divergence is the mean over treatment arms of D(P_arm || P_control); a split is scored by the
  size-weighted divergence of the children minus that of the parent. Every arm must keep
  `min_samples_leaf` rows on both sides.
* Each tree takes a subsample without replacement and splits it per arm into a structure half (picks
  splits) and an estimation half (fills leaves). Leaf effects come only from the estimation half; an
  arm with fewer than `min_estimation_count` estimation rows in a node takes its parent's mean.
* A Python brute-force split search is the test oracle: hypothesis checks that the kernel finds the
  same best gain on random small problems.

## Consequences

* Binning caps resolution at 32 thresholds per feature. Hillstrom's only continuous feature
  (`history`) loses some detail; the others have fewer distinct values than that.
* Honest leaves halve the rows that set each leaf value, trading variance for less overfitting of
  the effect estimates. The synthetic benchmark runs an adaptive (non-honest) KL forest beside the
  honest one so the trade-off is measured rather than assumed.
* KL needs a binary outcome; continuous outcomes such as spend use the Euclidean criterion, which
  becomes the squared difference in means.
* The normalisation term of Rzepakowski and Jaroszewicz that penalises unbalanced splits is not
  implemented; the per-arm minimum leaf size is the only guard.
* With the default `min_samples_leaf=100` per arm, small data sets (a few thousand rows) give trees
  that cannot split, and the forest predicts the average effect for everyone.

## Alternatives rejected

* causalml's `UpliftRandomForestClassifier`: a finished implementation of the same idea, so using it
  would defeat the purpose, and its dependency set is large for a reference used in one test.
* Exact splits on sorted values: more thresholds for little gain at this sample size, and a sort per
  node and feature.
* A causal forest with an R-learner splitting rule: a different estimator; the R- and DR-learners
  already cover the residualised approach.
