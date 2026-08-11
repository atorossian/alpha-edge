# Actuarial Portfolio Risk Modelling

## 1. Purpose

The actuarial portfolio risk module estimates whether a portfolio can survive adverse future paths, reach target wealth levels, recover from drawdowns, and support a given level of leverage.

The module is designed for Alpha Edge portfolio analysis. It does not replace the portfolio search engine. Instead, it evaluates simulated equity paths produced by other components and converts those paths into risk, solvency, and time-to-event metrics.

The key questions are:

* What is the probability of ruin?
* What is the probability of breaching a dangerous drawdown?
* How long does it take to reach ruin or a target goal?
* What is the probability of reaching the goal before ruin?
* How likely is the portfolio to recover after a drawdown breach?
* How much capital is required to absorb simulated losses?
* What is the current solvency ratio?
* What leverage level appears supportable under the model?

## 2. Conceptual Mapping from Actuarial Science to Portfolio Risk

Traditional actuarial models often study the solvency of an insurer. The insurer starts with a capital reserve, receives premiums, pays claims, and faces ruin if claims deplete capital below an acceptable level.

In Alpha Edge, we map those concepts as follows:

| Actuarial Concept    | Alpha Edge Interpretation                                            |
| -------------------- | -------------------------------------------------------------------- |
| Capital reserve      | Portfolio equity                                                     |
| Claims               | Negative return shocks / drawdowns                                   |
| Premium income       | Contributions, drift, portfolio gains                                |
| Ruin                 | Equity crossing a minimum acceptable threshold                       |
| Solvency margin      | Capital buffer against simulated path losses                         |
| Survival probability | Probability of not hitting ruin by a horizon                         |
| Time to ruin         | First day the portfolio breaches the ruin threshold                  |
| Capital adequacy     | Whether current capital is sufficient for the simulated risk profile |

This mapping is useful but imperfect. Financial market losses are not identical to insurance claims. Market losses are correlated, regime-dependent, clustered in crises, and affected by liquidity and leverage.

## 3. Equity Path Input

The module evaluates simulated portfolio equity paths.

Expected shape:

```text
rows    = simulation paths
columns = time steps
```

Column `0` is the initial value at `t=0`.

Example:

```text
[
  [100.0, 105.0, 110.0],
  [100.0, 95.0, 80.0],
  [100.0, 98.0, 120.0],
]
```

If `horizon_days=252`, then the path matrix must contain at least `253` columns, including `t=0`.

The module assumes that simulated paths have already been generated elsewhere. It does not currently generate simulations internally.

## 4. Ruin Event

Ruin is defined as the portfolio equity crossing a minimum acceptable floor.

The module uses an inclusive threshold:

```text
ruin occurs when equity <= ruin_threshold
```

The ruin threshold can be defined in two ways.

### 4.1 Absolute Threshold

Example:

```text
ruin_threshold = 10,000
```

Ruin occurs when portfolio equity falls to `10,000` or below.

### 4.2 Fraction of Initial Capital

Example:

```text
initial_value = 32,000
threshold_value = 0.50
```

The ruin threshold is:

```text
32,000 * 0.50 = 16,000
```

Ruin occurs when equity falls to `16,000` or below.

## 5. Ruin Probability

Ruin probability is the fraction of simulated paths that hit the ruin threshold at least once during the evaluation horizon.

```text
ruin_probability = paths_with_ruin / total_paths
```

Example:

```text
total paths = 10,000
paths with ruin = 400
ruin probability = 4.0%
```

## 6. Time to Ruin

Time to ruin is the first time index where a path hits the ruin threshold.

If a path never reaches ruin, its time to ruin is undefined and excluded from the mean and median time-to-ruin calculations.

The module reports:

* Expected time to ruin among paths that ruin.
* Median time to ruin among paths that ruin.

These values are conditional on ruin occurring.

## 7. Goal Event

A goal event occurs when portfolio equity reaches or exceeds a target value.

```text
goal occurs when equity >= goal_value
```

Example:

```text
goal_value = 50,000
```

The module reports:

* Probability of reaching the goal.
* Median time to goal among paths that reach the goal.

## 8. Goal Before Ruin

The module estimates the probability that a path reaches the target goal before it reaches ruin.

A path counts as successful if:

```text
goal happens and either ruin never happens or goal_time < ruin_time
```

If goal and ruin happen at the same time index, the path is not counted as goal-before-ruin.

This is intentionally conservative.

## 9. Drawdown

Drawdown is calculated relative to the running peak of each path.

```text
drawdown = equity / running_peak - 1
```

Examples:

```text
0.00  = no drawdown
-0.10 = -10% drawdown
-0.30 = -30% drawdown
```

The maximum drawdown of a path is the most negative drawdown observed during the horizon.

## 10. Drawdown Breach

A drawdown breach occurs when the maximum drawdown falls below a configured limit.

The configured limit is expressed as a positive fraction.

Example:

```text
drawdown_limit_pct = 0.30
```

This means the path breaches if:

```text
drawdown <= -0.30
```

The module uses an inclusive breach rule.

## 11. Drawdown Breach Probability

Drawdown breach probability is:

```text
drawdown_breach_probability = paths_with_breach / total_paths
```

This measures how frequently the simulated portfolio enters an unacceptable drawdown state.

This is different from ruin. A portfolio can breach a drawdown limit and later recover without reaching ruin.

## 12. Recovery After Drawdown Breach

The recovery model starts after the first drawdown breach.

For each breached path:

1. Find the first time the path breaches the drawdown limit.
2. Record the running peak at the time of breach.
3. Check whether the path later recovers to `breach_peak * recovery_level`.

With:

```text
recovery_level = 1.0
```

recovery means returning to the prior peak.

The module reports:

* Recovery probability among breached paths.
* Median recovery time among breached paths that recover.

If no path breaches the drawdown limit, recovery probability is reported as `None`.

## 13. Survival Curve

The survival curve reports the probability that an event has not occurred by each selected horizon.

Currently, the implemented survival curve is based on the ruin event.

For horizon `h`:

```text
event_probability(h) = P(ruin_time <= h)
survival_probability(h) = 1 - event_probability(h)
```

Example horizons:

```text
[21, 63, 126, 252, 756]
```

These approximate:

* 1 month.
* 3 months.
* 6 months.
* 1 year.
* 3 years.

## 14. Maximum Drawdown CVaR

The module reports a CVaR-style statistic over the maximum drawdown distribution.

For example, with `alpha=0.95`, it averages the worst 5% most negative maximum drawdowns.

This is useful because the average maximum drawdown can hide severe tail outcomes.

## 15. Capital Adequacy

Capital adequacy estimates how much capital is required to absorb simulated path losses.

For each path, the model calculates:

```text
capital_loss = initial_value - minimum_equity_on_path
```

The loss is floored at zero.

Example:

```text
initial_value = 100
minimum_equity = 70
capital_loss = 30
```

Then the required capital is calculated from a high quantile of the simulated capital loss distribution.

If:

```text
target_ruin_probability = 0.05
```

then required capital is based on the 95th percentile of simulated path losses.

```text
capital_required = quantile(losses, 1 - target_ruin_probability)
```

If `min_solvent_capital_ratio` is greater than `1`, the required capital is scaled upward.

```text
capital_required = loss_quantile * min_solvent_capital_ratio
```

## 16. Capital Buffer Gap

Capital buffer gap is:

```text
capital_buffer_gap = current_capital - capital_required
```

Interpretation:

```text
positive gap = capital surplus
negative gap = capital shortfall
```

Example:

```text
current_capital = 100
capital_required = 120
capital_buffer_gap = -20
```

This means the portfolio is undercapitalized by `20` under the model.

## 17. Solvency Ratio

Solvency ratio is:

```text
solvency_ratio = current_capital / capital_required
```

Interpretation:

```text
solvency_ratio > 1.0 = capital surplus
solvency_ratio = 1.0 = exactly capitalized
solvency_ratio < 1.0 = capital shortfall
```

If `capital_required` is zero, the ratio is reported as `None` because the denominator is not meaningful.

## 18. Safe Leverage Estimate

The first version uses a conservative heuristic:

```text
safe_leverage = current_leverage * solvency_ratio
```

Then the result is capped by:

```text
max_allowed_leverage
```

Example:

```text
current_leverage = 1.0
solvency_ratio = 1.5
max_allowed_leverage = 2.0
safe_leverage = 1.5
```

Example with cap:

```text
current_leverage = 1.0
solvency_ratio = 3.0
max_allowed_leverage = 2.0
safe_leverage = 2.0
```

Example with undercapitalization:

```text
current_leverage = 2.0
solvency_ratio = 0.5
safe_leverage = 1.0
```

This means the model estimates that current leverage should be reduced.

## 19. Preliminary Risk Grade

The current risk grade is intentionally simple.

It compares:

* Ruin probability against target ruin probability.
* Drawdown breach probability against target drawdown breach probability.

Grades are:

```text
A, B, C, D, F
```

This should be considered preliminary. Later versions may incorporate:

* Solvency ratio.
* Capital buffer gap.
* Safe leverage.
* Recovery probability.
* CVaR max drawdown.
* Regime-adjusted risk.
* EVT tail metrics.

## 20. Current Implementation Files

Current module structure:

```text
src/alpha_edge/risk/actuarial/
    __init__.py
    path_metrics.py
    solvency.py
    engine.py
```

Schemas are centralized in:

```text
src/alpha_edge/core/schemas.py
```

Current tests:

```text
tests/unit/risk/actuarial/test_schemas.py
tests/unit/risk/actuarial/test_path_metrics.py
tests/unit/risk/actuarial/test_solvency.py
```

## 21. Current Model Limitations

This module produces model-based estimates, not guarantees.

Important limitations:

1. Results depend on the quality of simulated equity paths.
2. Historical returns may not capture future crisis behavior.
3. Bootstrap paths can understate regime shifts and structural breaks.
4. Correlations may increase during market stress.
5. Leverage introduces nonlinear risk.
6. Liquidation and margin mechanics are not fully modelled yet.
7. The first safe leverage estimate is a conservative heuristic, not an optimization.
8. The current model does not yet include EVT tail extrapolation.
9. The current model does not yet include Bayesian credibility adjustment.
10. The current model does not yet include regime-dependent actuarial hazard curves.

## 22. Intended Future Extensions

Future extensions may include:

* Regime-aware ruin probability.
* Hazard rate estimation.
* Survival analysis beyond ruin.
* Drawdown hazard curves.
* EVT-based extreme capital requirement.
* Bayesian credibility adjustment.
* Dynamic contributions.
* Withdrawal modelling.
* Margin liquidation modelling.
* Path-dependent leverage reduction.
* Integration into portfolio search scoring.
* Integration into daily report.
* Persistence into datalake/warehouse outputs.

## 23. Interpretation Guidance

The actuarial module should be used as a risk-control layer.

It should not be interpreted as a prediction that a specific outcome will occur.

A high ruin probability means the portfolio is fragile under the simulated assumptions.

A low ruin probability does not mean the portfolio is safe under all future conditions.

A positive capital buffer means the portfolio appears adequately capitalized under the model.

A negative capital buffer means the simulated loss distribution suggests more capital is required.

A safe leverage estimate below current leverage should be treated as a warning.

## 24. Design Principle

The module should remain:

* Transparent.
* Testable.
* Conservative.
* Documented.
* Modular.
* Independent from storage.
* Independent from portfolio search implementation details.

Portfolio search can later consume these metrics, but the actuarial module itself should operate on clean equity paths and configuration objects.
