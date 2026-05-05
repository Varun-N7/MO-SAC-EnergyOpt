# MO-RL-PeakShaving — Project Details for Planner (Week-by-Week)

Use this breakdown in **Microsoft Planner** (or any task tool): create a **Plan** named **MO-RL-PeakShaving**, then add **Buckets** per phase and **Tasks** per week. Adjust start/end dates to match your semester or deadline.

---

## Project summary (one line)

**Adaptive Multi-Objective Reinforcement Learning for peak load shaving in solar-integrated smart grids with consumer comfort constraints (PPO, Gymnasium, Flask web UI).**

---

## Phase 1 — Setup & environment (Weeks 1–2)

| Week | Deliverables / Completed | Tasks for Planner |
|------|---------------------------|-------------------|
| **Week 1** | Project setup, repo structure, config | • Create project folder structure (env, train, evaluation, models, results, data). • Add `config.py` (battery, PV, load, price, comfort, reward, training params). • Add `requirements.txt` (gymnasium, stable-baselines3, numpy, pandas, matplotlib, flask, flask-cors). • Set up Python venv and install dependencies. • Add README with overview and structure. |
| **Week 2** | Smart grid simulation environment | • Implement `env/smart_grid_env.py` (Gymnasium env: 24h episodes, obs: hour, pv, load, price, soc, comfort; 5 discrete actions). • Implement `env/pv_model.py`, `env/load_model.py`, `env/price_model.py`. • Implement comfort model and hard constraint (terminate if comfort < 0.60). • Add `env/__init__.py`. • Test env with `test_env.py` or `quick_demo.py`. |

**Phase 1 completion:** Environment runs, observation/action spaces defined, comfort and termination working.

---

## Phase 2 — Training pipeline (Weeks 3–4)

| Week | Deliverables / Completed | Tasks for Planner |
|------|---------------------------|-------------------|
| **Week 3** | PPO training and callbacks | • Implement `train/train_ppo.py` (PPO from stable-baselines3, 50k timesteps). • Implement `train/callbacks.py` (evaluation callback, best-model save). • Save final model to `models/ppo_final_model.zip`, best to `models/ppo_best_model.zip`. • Tune hyperparameters in `config.py` (learning rate, n_steps, batch_size, n_epochs). |
| **Week 4** | Training validation and real-data option | • Run full training and confirm no crashes; check reward curve. • (Optional) Add `data/download_dataset.py` and real-world dataset support. • Implement `train/train_ppo_real_data.py` and `env/real_data_loader.py` if using real data. • Document training commands in README. |

**Phase 2 completion:** Trained PPO model saved; optional real-data training path.

---

## Phase 3 — Evaluation & baselines (Weeks 5–6)

| Week | Deliverables / Completed | Tasks for Planner |
|------|---------------------------|-------------------|
| **Week 5** | Baseline and evaluation scripts | • Implement `evaluation/baseline_controller.py` (rule-based heuristic). • Implement `evaluation/evaluate.py`: run 10 episodes for RL and baseline; compute peak kW, total cost, min comfort. • Save `results/logs/summary.csv` and `results/logs/episode_trace_rl.csv`. |
| **Week 6** | Plots and run-all pipeline | • Implement `evaluation/plot_results.py` (grid import vs PV, SOC, comfort, etc.). • Save plots to `results/plots/`. • Implement `run_all.py` (train → evaluate in sequence). • (Optional) `evaluation/evaluate_real_data.py` if using real data. • Document evaluation and run_all in README. |

**Phase 3 completion:** RL vs baseline metrics and plots; one-command run_all.

---

## Phase 4 — Web interface (Weeks 7–8)

| Week | Deliverables / Completed | Tasks for Planner |
|------|---------------------------|-------------------|
| **Week 7** | Flask app and API | • Implement `app.py` (Flask + CORS). • Routes: `/`, `/api/summary`, `/api/episode_trace`, `/api/training_stats`, `/api/plots/...`, etc. • Serve results from `results/logs/` and `results/plots/`. • Optional: load PPO model and run live evaluation endpoint. |
| **Week 8** | Front-end and polish | • Implement `templates/index.html` (dashboard: summary table, episode trace, training stats, plots). • Test in browser at http://localhost:5000. • Document how to run app in README. • Final README update: usage, config, results locations, dependencies. |

**Phase 4 completion:** Web UI showing summary, traces, stats, and plots.

---

## Phase 5 — Documentation & submission (Week 9+)

| Week | Deliverables / Completed | Tasks for Planner |
|------|---------------------------|-------------------|
| **Week 9+** | Report / presentation / submission | • Write report or slides: problem, method (env, reward, PPO), results (RL vs baseline), conclusions. • Add any data README (`data/README.md`) if using real datasets. • Prepare submission (code zip or repo link, model files if required). • Optional: short video demo of training and web UI. |

**Phase 5 completion:** Documentation and deliverables ready.

---

## Quick reference — project structure

```
MO-RL-PeakShaving/
├── README.md
├── requirements.txt
├── config.py
├── run_all.py
├── app.py
├── env/
│   ├── __init__.py
│   ├── smart_grid_env.py
│   ├── pv_model.py
│   ├── load_model.py
│   ├── price_model.py
│   ├── real_data_loader.py   (optional)
├── train/
│   ├── train_ppo.py
│   ├── train_ppo_real_data.py (optional)
│   └── callbacks.py
├── evaluation/
│   ├── baseline_controller.py
│   ├── evaluate.py
│   ├── evaluate_real_data.py (optional)
│   └── plot_results.py
├── data/
│   ├── download_dataset.py   (optional)
│   └── real_world/            (optional)
├── models/
├── results/
│   ├── plots/
│   └── logs/
└── templates/
    └── index.html
```

---

## Suggested Planner buckets and tasks (copy-paste)

**Bucket 1: Phase 1 – Setup & environment**  
- Task: Week 1 – Project setup, config, requirements, README  
- Task: Week 2 – Smart grid env (Gymnasium), PV/load/price/comfort models, test env  

**Bucket 2: Phase 2 – Training**  
- Task: Week 3 – PPO training script, callbacks, model save  
- Task: Week 4 – Training validation, optional real-data pipeline  

**Bucket 3: Phase 3 – Evaluation**  
- Task: Week 5 – Baseline controller, evaluate.py, summary + trace logs  
- Task: Week 6 – plot_results.py, run_all.py, README usage  

**Bucket 4: Phase 4 – Web UI**  
- Task: Week 7 – Flask app, API routes, serve logs/plots  
- Task: Week 8 – index.html dashboard, test at localhost:5000, README  

**Bucket 5: Phase 5 – Documentation**  
- Task: Week 9+ – Report/presentation, data README, submission prep  

---

*Adjust week numbers and dates in Planner to match your actual schedule. You can add due dates and assignees per task.*
