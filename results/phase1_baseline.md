# Phase 1: Baseline Failure Report
**Date:** 2026-01-11
**Objective:** Quantify ResNet-50 degradation in non-standard weather.

## Summary of Findings
| Domain | Avg Entropy | Status | Interpretation |
| :--- | :--- | :--- | :--- |
| **Sunny** (Baseline) | 4.1044 | ❌ HIGH CONFUSION | *Unexpected.* The model is struggling with glare or shadows? |
| **Rain** | 1.6730 | ⚠️ CONFUSED | Expected degradation. |
| **Night** | 0.1744 | ⚠️ FALSE CONFIDENCE | **Critical Finding:** The model is "Confident" (Low Entropy) but likely wrong. This is dangerous (False Positive). |

## Conclusion
The standard ResNet-50 is unreliable. It is either highly confused (Sunny/Rain) or dangerously overconfident (Night). This proves the need for Gated LoRA.
