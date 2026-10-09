# ADR 0003: Allocate contacts by Lagrangian relaxation instead of top-k or an ILP solver

Status: accepted, 2026-10-09

## Context

Hillstrom has two treatments, so a policy chooses an arm per customer as well as whom to contact.
Arms may also cost different amounts (a printed catalogue against an e-mail). Choosing assignments
that maximise predicted incremental outcomes under a total cost cap is a multiple-choice knapsack
problem. It is solved once per held-out fold and budget in the benchmark and once per scoring run
in the CLI.

## Decision

`causal_uplift.policy.allocate` uses Lagrangian relaxation. For a price lambda per unit of cost,
every customer independently takes the arm with the largest `benefit - lambda * cost`, or no contact
if that is not positive. Lambda is bisected to the smallest value whose spend fits the budget, and
the budget left by the last discrete jump is filled greedily. The dual function at the final lambda
is returned as `upper_bound`, a certificate of how far from optimal the allocation can be.

## Consequences

* With equal costs the result is "top-k customers by best predicted uplift, skipping customers whose
  best predicted uplift is not positive"; hypothesis checks it against brute force.
* With unequal costs the allocation is feasible and never better than the optimum, and the optimum
  never exceeds `upper_bound`; hypothesis checks both against brute force.
* Cost is O(n K log(1/tol)) with numpy only.
* Customers with a non-positive best predicted uplift are never contacted, so a policy may spend less
  than its budget. Reports show the realised contact rate.

## Alternatives rejected

* Plain top-k: ignores arm costs and handles the arm choice only implicitly.
* An integer programming solver (OR-Tools, PuLP): exact, but a heavy dependency for a problem whose
  LP relaxation is already tight to within about one customer.
* Thresholding at a fixed uplift: does not respect the budget.
