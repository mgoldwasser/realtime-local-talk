# Q1 2026 — Equity Factor Exposure Review

**Author:** Quantitative Strategy
**Date:** 2026-02-18

## Summary

The TAA composite is currently overweight **quality** and **low-volatility**,
neutral on **value**, underweight **size** (large cap tilt) and underweight
**momentum**. This is broadly consistent with a late-cycle stance.

## Factor scores (Z-score vs Russell 3000)

| Factor          | Current | 6-month avg | Comment                          |
|-----------------|---------|-------------|----------------------------------|
| Quality         | +0.42   | +0.31       | Driven by IG balance sheet tilts |
| Low-volatility  | +0.28   | +0.19       | Healthcare + utilities OW        |
| Value           | -0.04   | -0.11       | Roughly neutral                  |
| Size (small)    | -0.34   | -0.29       | Persistent large-cap bias        |
| Momentum        | -0.22   | -0.05       | Trimmed AI mega-cap winners      |

## Discussion

The intentional trim of mega-cap AI winners in late 2025 has rotated the
momentum factor from neutral to small underweight. Risk model attribution
shows momentum contributing -3 bps to the recent 3-month tracking error.

The quality and low-volatility tilts are an explicit defensive stance that
the committee approved in October. We see no near-term reason to unwind,
but if the soft-landing narrative strengthens we would expect to trim
low-vol back toward neutral.

## Implementation drift

Factor exposures have drifted ~0.05 Z on quality and low-vol since the
December rebalance. We are below the 0.10 Z drift threshold that would
trigger an interim rebalance.

## Risks

Momentum mean-reversion is the main near-term risk if mega-cap AI names
extend their rally. We accept this in exchange for the diversification
benefit of the quality tilt.
