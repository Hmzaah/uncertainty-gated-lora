# 📋 Project Quest Board

**Instructions for Agents:**
1.  **Claiming:** Read this file. If a task is `[ ]`, write your name next to "Assigned:".
2.  **Reporting:** DO NOT paste logs here.
    * *Correct:* "See `results/phase1_baseline.md` for details."
    * *Incorrect:* (Pasting 50 lines of CSV data).
3.  **Completion:** When work is committed, change `[ ]` to `[x]`.

---

## ✅ Phase 1: The Baseline (Complete)
**Goal:** Prove that ResNet-50 fails in Rain/Night.
*See `results/phase1_baseline.md` for the final report.*

- [x] **Task 1.1:** Find 10 "Sunny" images -> `data/test_sunny/`
- [x] **Task 1.2:** Find 10 "Rainy" images -> `data/test_rain/`
- [x] **Task 1.3:** Find 10 "Night" images -> `data/test_night/`
- [x] **Task 1.4:** Run Baseline Script & Log Entropy.
- [ ] **Task 1.5:** Generate LaTeX Report.
    - *Action:* Create `report/gen_latex.py` that reads the results and outputs a PDF table.
    - *Assigned:* ---

## 🛠 Phase 2: System Architecture (The Engine)
**Goal:** Build the modular factory so we can swap LoRA/DoRA easily.

- [ ] **Task 2.1: The Config Schema**
    - *Action:* Create `configs/README.md` defining the YAML structure (rank, alpha, method).
    - *Assigned:* - [ ] **Task 2.2: The Adapter Factory (`core/factory.py`)**
    - *Action:* Implement `AdapterFactory` class that reads YAML and builds `peft.LoraConfig`.
    - *Assigned:* - [ ] **Task 2.3: The Universal Trainer (`scripts/train_modular.py`)**
    - *Action:* Create a training script that uses `AdapterFactory` to train on any domain.
    - *Assigned:* ---

## 🧪 Phase 3: The Experiments (The Fuel)
**Goal:** Use the engine to generate our "Experts".
*(Requires Phase 2 completion)*

- [ ] **Task 3.1: Train Sunny Adapter**
    - *Config:* `configs/adapters/sunny_lora.yaml`
    - *Assigned:* - [ ] **Task 3.2: Train Rain Adapter**
    - *Config:* `configs/adapters/rain_lora.yaml`
    - *Assigned:* - [ ] **Task 3.3: Train Night Adapter**
    - *Config:* `configs/adapters/night_lora.yaml`
    - *Assigned:* ---

## 📊 Phase 4: Evaluation (The Scoreboard)
**Goal:** Measure the win.

- [ ] **Task 4.1:** Implement Gating Logic (`core/gate/entropy.py`).
- [ ] **Task 4.2:** Run Final Benchmark (`monitor.py`).
