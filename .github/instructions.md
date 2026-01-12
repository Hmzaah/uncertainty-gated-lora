# AGENT INSTRUCTIONS (STRICT)

**📋 Task Management:**
Before starting work, ALWAYS read `TASKS.md` to see what is pending. 
If you complete a task, mark it with `[x]` in your commit.

## 🎯 Current Mission: Phase 2 - Architecture & Training
**Goal:** Build a modular, config-driven "Adapter Factory" and train the first set of "Expert" adapters.
**Status:** Phase 1 (Baseline Failure) is PROVEN. We are now building the solution.

**The Strategy:**
1.  **Build the Factory:** Implement `core/factory.py` to generate models dynamically (LoRA/DoRA/PiSSA) from YAML configs.
2.  **Train Experts:** Use the factory to train three distinct adapters:
    - `adapters/rain_lora` (Target: Rain Domain)
    - `adapters/night_lora` (Target: Night Domain)
    - `adapters/sunny_lora` (Target: Sunny Domain)
3.  **Strict Constraint:** DO NOT hardcode model parameters in Python scripts. All hyperparameters (rank, alpha, dropout) must live in `configs/`.

# 🏛️ SYSTEM ARCHITECTURE & STRICT GUIDELINES

## 1. The "Two-Framework" Rule
To support future ablation studies (e.g., comparing LoRA vs DoRA), the code is strictly separated:

* **Framework A: The Builder (`core/factory`)**
    * **Role:** Constructs the model architecture.
    * **Rule:** NEVER hardcode `LoraConfig` inside a training script.
    * **Input:** Must read from `configs/adapters/*.yaml`.
    * **Logic:** `Factory.get_model(config_path) -> Model`

* **Framework B: The Evaluator (`metrics/monitor`)**
    * **Role:** Benchmarks the model.
    * **Rule:** Every inference script MUST use `metrics.monitor.PerformanceMonitor`.
    * **Output:** MUST generate a CSV report in `results/logs`.

## 2. The "Config-First" Workflow
Do not pass hyperparameters via command-line arguments (like `--rank 16`).
1.  Create a specific config file: `configs/adapters/rain_dora_rank32.yaml`.
2.  Pass only the file path to the script: `python scripts/train.py --config configs/...`

## Project Context
This project investigates "Uncertainty-Gated LoRA." However, currently, we are establishing the **Baseline**.
We hypothesize that standard models fail in non-stationary environments. We need hard data to prove this.

## Technical Constraints
-   **Model:** `ResNet-50` (Frozen Backbone) + `PEFT` Adapters.
-   **Configuration:** All model params must be in `.yaml` format.
-   **Hardware:** Code must be device-agnostic (Auto-detect CUDA/MPS/CPU).
-   **Metrics:** Latency (ms), VRAM (MB), and Shannon Entropy must be logged via `PerformanceMonitor`.

## Project Change Log
(Do not edit below this line manually. The system updates this automatically.)
-------------------------------------------------------------------------------
- **[2026-01-09 05:06] Sumanth Gopisetty:** ci: Add auto-scribe workflow to track history
- **[2026-01-09 06:06] Sumanth Gopisetty:** docs: Add tasks.md quest board
- **[2026-01-09 07:06] SumanthGopisetty109:** Merge pull request #1 from Hmzaah/feature/project-skeleton

feat(phase1): establish baseline and add data collection scripts
- **[2026-01-09 13:13] Hmzaah:** Merge pull request #2 from Hmzaah/feature/project-skeleton

feat(report): add visualization reproducer and latex generator
- **[2026-01-11 05:11] Sumanth Gopisetty:** feat: Enforce modular framework and add PerformanceMonitor
- **[2026-01-11 06:04] Sumanth Gopisetty:** f a text editor pops up: Press 'Ctrl+X', then 'Y', then 'Enter')

git push origin main:
Merge branch 'main' of https://github.com/GSumanth109/uncertainty-gated-lora
