"""Professional Streamlit dashboard for MO-RL-PeakShaving.

Run with:
    streamlit run dashboard/app.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.sac_agent import ContinuousActionWrapper
from env.smart_grid_env import SmartGridEnv
from evaluation.baseline_controller import BaselineController

try:
    from stable_baselines3 import SAC
except Exception:
    SAC = None

USD_TO_GBP = 0.79
ACTION_NAMES = ["do_nothing", "charge_battery", "discharge_battery", "shift_load", "comfort_sacrifice"]
ACTION_LABELS = {
    0: "do_nothing",
    1: "discharge_battery",
    2: "charge_battery",
    3: "shift_load",
    4: "comfort_sacrifice",
}
ACTION_COLORS = {
    0: "#7F8C8D",
    1: "#3B82F6",
    2: "#22C55E",
    3: "#F59E0B",
    4: "#EF4444",
}
NAV_ITEMS = [
    ("Dashboard", "📊 Dashboard"),
    ("Pareto Analysis", "🎯 Pareto Analysis"),
    ("Algorithm Comparison", "⚡ Algorithm Comparison"),
]


def _season_from_month(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _safe_read_csv(path):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0E1117 0%, #10151F 100%);
            font-size: 1.03rem;
        }
        .top-accent {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #00C9A7, #845EF7);
            z-index: 9999;
        }
        [data-testid="stSidebar"] {
            min-width: 360px !important;
            max-width: 360px !important;
            background: #111824;
            border-right: 1px solid rgba(255,255,255,0.10);
        }
        [data-testid="stSidebar"] .sidebar-title {
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] .sidebar-bolt {
            color: #00C9A7;
            text-shadow: 0 0 12px rgba(0, 201, 167, 0.55);
            font-size: 1.75rem;
            margin-right: 0.25rem;
        }
        .sidebar-divider {
            border: none;
            height: 1px;
            background: rgba(255,255,255,0.12);
            margin: 0.55rem 0 0.75rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.14);
            background: #1E1E2E;
            color: #ffffff;
            padding: 0.65rem 0.8rem;
            text-align: left;
            transition: all 0.22s ease;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
            background: rgba(0, 201, 167, 0.20);
            color: #00C9A7;
            border-left: 3px solid #00C9A7;
            border-top: 1px solid rgba(0, 201, 167, 0.45);
            border-right: 1px solid rgba(0, 201, 167, 0.45);
            border-bottom: 1px solid rgba(0, 201, 167, 0.45);
            font-weight: 700;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
            border-color: rgba(0, 201, 167, 0.45);
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button[aria-label="▶ Run Episode"] {
            background: rgba(0, 201, 167, 0.92);
            color: #032B25;
            text-align: center;
            font-weight: 800;
            border: none;
        }
        .banner {
            padding: 1.45rem 1.4rem;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            background: rgba(18,24,32,0.95);
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            margin-bottom: 1.15rem;
        }
        .card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
            padding: 0.95rem 1.1rem;
            background: rgba(21,26,33,0.92);
            box-shadow: 0 6px 24px rgba(0,0,0,0.20);
            margin-bottom: 0.95rem;
            transition: transform 0.18s ease, border-color 0.2s ease;
        }
        .card:hover {
            transform: translateY(-1px);
            border-color: rgba(0, 201, 167, 0.38);
        }
        .kpi-card {
            border-left: 3px solid rgba(0, 201, 167, 0.8);
        }
        .pill {
            display: inline-block;
            margin-left: 0.4rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.20);
            font-size: 0.84rem;
            background: rgba(0, 201, 167, 0.14);
            color: #CCFFF6;
        }
        .pill-dot {
            display:inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            background: #00C9A7;
            box-shadow: 0 0 8px rgba(0,201,167,0.65);
            animation: pulseGlow 1.8s infinite;
        }
        @keyframes pulseGlow {
            0% { transform: scale(1); opacity: 0.7; }
            50% { transform: scale(1.22); opacity: 1; }
            100% { transform: scale(1); opacity: 0.7; }
        }
        .kpi-value { font-size: 1.82rem; font-weight: 700; }
        .kpi-delta-good { color: #00C9A7; font-weight: 600; }
        .kpi-delta-bad { color: #FF6B6B; font-weight: 600; }
        .hint { color: #9fb0c3; font-size: 0.93rem; }
        .stMarkdown p, .stCaption, .stSelectbox label, .stSlider label { font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(config.REAL_DATASET_PATH):
        return None
    df = pd.read_csv(config.REAL_DATASET_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df["datetime"] = pd.date_range("2023-01-01", periods=len(df), freq="h")
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].map(_season_from_month)
    df["episode_date"] = df["datetime"].dt.date
    return df


@st.cache_resource(show_spinner=False)
def load_ppo_model():
    for filename in [config.BEST_MODEL_FILENAME, config.FINAL_MODEL_FILENAME]:
        path = config.MODELS_DIR / filename
        if path.exists():
            return PPO.load(str(path))
    return None


@st.cache_resource(show_spinner=False)
def load_sac_model():
    if SAC is None:
        return None
    for filename in ["sac_algorithm_compare.zip", "sac_best_model.zip", "sac_final_model.zip"]:
        path = config.MODELS_DIR / filename
        if path.exists():
            return SAC.load(str(path))
    return None


def normalize_weights(cost, peak, comfort):
    total = cost + peak + comfort
    if total <= 1e-8:
        return {"cost": 1 / 3, "peak": 1 / 3, "comfort": 1 / 3, "battery": 0.0}
    return {
        "cost": cost / total,
        "peak": peak / total,
        "comfort": comfort / total,
        "battery": 0.0,
    }


def reward_components(pre, post, reward_weights):
    peak_penalty = 0.0
    if post["grid_import"] > config.PEAK_DEMAND_THRESHOLD:
        peak_w = config.ADAPTIVE_PEAK_WEIGHT_MULTIPLIER if post["hour"] in config.PEAK_HOURS else 1.0
        peak_penalty = config.PEAK_PENALTY_BASE * peak_w
        if post["hour"] in config.PEAK_HOURS:
            peak_penalty *= config.PEAK_PENALTY_MULTIPLIER
    peak_reward = reward_weights["peak"] * peak_penalty

    cost_reward = -(reward_weights["cost"] * post["grid_import"] * post["price"] * config.COST_WEIGHT)

    comfort_reward = 0.0
    if post["comfort"] < config.COMFORT_PENALTY_THRESHOLD:
        comfort_w = config.ADAPTIVE_COMFORT_WEIGHT_MULTIPLIER if post["comfort"] < config.ADAPTIVE_COMFORT_THRESHOLD else 1.0
        comfort_penalty = config.COMFORT_VIOLATION_PENALTY * comfort_w
        if post["comfort"] < config.SEVERE_COMFORT_THRESHOLD:
            comfort_penalty *= config.COMFORT_VIOLATION_MULTIPLIER
        comfort_reward = reward_weights["comfort"] * comfort_penalty

    return peak_reward, cost_reward, comfort_reward


def run_single_episode(policy_name, model, reward_weights, start_date, battery_capacity_kwh, initial_soc, compare_baseline):
    old_capacity = config.BATTERY_CAPACITY_KWH
    old_soc = config.INITIAL_SOC
    config.BATTERY_CAPACITY_KWH = float(battery_capacity_kwh)
    config.INITIAL_SOC = float(initial_soc)

    try:
        env = SmartGridEnv(use_real_data=True, dataset_path=config.REAL_DATASET_PATH, reward_weights=reward_weights)
        if policy_name == "SAC":
            env = ContinuousActionWrapper(env)

        obs, _ = env.reset(options={"start_date": str(start_date)})
        rows = []
        rewards_parts = []
        for _ in range(config.EPISODE_LENGTH):
            core = env.unwrapped if hasattr(env, "unwrapped") else env
            pre = {
                "hour": int(core.hour),
                "pv": float(core.pv),
                "load": float(core.base_load),
                "price": float(core.price),
                "soc": float(core.soc),
                "comfort": float(core.comfort),
                "battery_health": float(getattr(core, "battery_health", 1.0)),
                "grid_import": float(getattr(core, "grid_import", 0.0)),
            }
            if policy_name == "PPO":
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, np.ndarray):
                    action_to_step = int(np.asarray(action).reshape(-1)[0])
                else:
                    action_to_step = int(action)
            else:
                action, _ = model.predict(obs, deterministic=True)
                action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                action_to_step = np.array([float(action_arr[0])], dtype=np.float32)

            step_result = env.step(action_to_step)
            obs, reward, done, truncated, info = step_result
            action_id = int(core.episode_history[-1]["action"] if core.episode_history else 0)
            post = {
                "hour": int(pre["hour"]),
                "pv": pre["pv"],
                "load": pre["load"],
                "price": pre["price"],
                "action": action_id,
                "grid_import": float(info.get("grid_import", 0.0)),
                "soc": float(info.get("soc", pre["soc"])),
                "comfort": float(info.get("comfort", pre["comfort"])),
                "battery_health": float(info.get("battery_health", pre["battery_health"])),
                "reward": float(reward),
                "battery_flow": float(pre["load"] - pre["pv"] - float(info.get("grid_import", 0.0))),
            }
            peak_r, cost_r, comfort_r = reward_components(pre, post, reward_weights)
            rewards_parts.append(
                {
                    "hour": post["hour"],
                    "peak_component": peak_r,
                    "cost_component": cost_r,
                    "comfort_component": comfort_r,
                }
            )
            rows.append(post)
            if done or truncated:
                break

        agent_df = pd.DataFrame(rows)
        rewards_df = pd.DataFrame(rewards_parts)
        baseline_df = pd.DataFrame()

        if compare_baseline:
            base_env = SmartGridEnv(use_real_data=True, dataset_path=config.REAL_DATASET_PATH, reward_weights=reward_weights)
            b_obs, _ = base_env.reset(options={"start_date": str(start_date)})
            controller = BaselineController(base_env)
            b_rows = []
            for _ in range(config.EPISODE_LENGTH):
                action = int(controller.predict(b_obs))
                pre_gi = float(base_env.grid_import)
                b_obs, reward, done, truncated, info = base_env.step(action)
                b_rows.append(
                    {
                        "hour": int(base_env.episode_history[-1]["hour"]) if base_env.episode_history else int(base_env.hour),
                        "pv": float(info.get("pv", base_env.pv)),
                        "load": float(info.get("load", base_env.base_load)),
                        "price": float(info.get("price", base_env.price)),
                        "action": action,
                        "grid_import": float(info.get("grid_import", pre_gi)),
                        "soc": float(info.get("soc", base_env.soc)),
                        "comfort": float(info.get("comfort", base_env.comfort)),
                        "battery_health": float(info.get("battery_health", base_env.battery_health)),
                        "reward": float(reward),
                    }
                )
                if done or truncated:
                    break
            baseline_df = pd.DataFrame(b_rows)

        return agent_df, baseline_df, rewards_df
    finally:
        config.BATTERY_CAPACITY_KWH = old_capacity
        config.INITIAL_SOC = old_soc


def compute_metrics(df):
    if df.empty:
        return {"cost": 0.0, "peak": 0.0, "comfort": 0.0, "battery": 0.0}
    return {
        "cost": float((df["grid_import"] * df["price"]).sum() * USD_TO_GBP),
        "peak": float(df["grid_import"].max()),
        "comfort": float(df["comfort"].min() * 100.0),
        "battery": float(df["battery_health"].iloc[-1] * 100.0),
    }


def render_header(model_loaded, dataset_rows):
    st.markdown(
        f"""
        <div class="top-accent"></div>
        <div class="banner">
            <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;">
                <div>
                    <h2 style="margin:0;">Multi-Objective RL Energy Manager</h2>
                    <div class="hint" style="margin-top:0.32rem;">PPO Agent for Smart Grid Peak Shaving</div>
                </div>
                <div>
                    <span class="pill"><span class="pill-dot"></span>Environment: Ready</span>
                    <span class="pill"><span class="pill-dot"></span>Model: {"Loaded" if model_loaded else "Missing"}</span>
                    <span class="pill"><span class="pill-dot"></span>Dataset: {dataset_rows} records</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing():
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><b>Peak Shaving</b><br/>Minimize max grid import during stress hours.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>Cost Reduction</b><br/>Lower total electricity cost using TOU pricing.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>Comfort Preservation</b><br/>Keep user comfort above the constraint threshold.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        <b>Energy Flow</b><br/>
        <div style="font-size:1.2rem; margin-top:0.5rem;">PV ☀️ → Battery 🔋 → Home 🏠 → Grid ⚡</div>
        <div class="hint" style="margin-top:0.5rem;">Configure settings in the sidebar and click Run Episode.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="card">
            <b>System Architecture</b>
            <div style="margin-top:0.85rem; display:flex; align-items:center; justify-content:space-between; gap:0.45rem; flex-wrap:wrap;">
                <div style="min-width:130px; padding:0.6rem; border-radius:10px; border:1px solid rgba(0,201,167,0.45); background:rgba(0,201,167,0.14); text-align:center;">
                    <div><b>Real Dataset</b></div><div class="hint">hourly time-series</div>
                </div>
                <div style="color:#00C9A7; font-size:1.2rem;">→</div>
                <div style="min-width:130px; padding:0.6rem; border-radius:10px; border:1px solid rgba(132,94,247,0.5); background:rgba(132,94,247,0.16); text-align:center;">
                    <div><b>Environment</b></div><div class="hint">grid simulator</div>
                </div>
                <div style="color:#00C9A7; font-size:1.2rem;">→</div>
                <div style="min-width:130px; padding:0.6rem; border-radius:10px; border:1px solid rgba(0,201,167,0.45); background:rgba(0,201,167,0.14); text-align:center;">
                    <div><b>PPO Agent</b></div><div class="hint">policy decisions</div>
                </div>
                <div style="color:#00C9A7; font-size:1.2rem;">→</div>
                <div style="min-width:165px; padding:0.6rem; border-radius:10px; border:1px solid rgba(132,94,247,0.5); background:rgba(132,94,247,0.16); text-align:center;">
                    <div><b>Multi-Objective Reward</b></div><div class="hint">cost/peak/comfort</div>
                </div>
                <div style="color:#00C9A7; font-size:1.2rem;">→</div>
                <div style="min-width:130px; padding:0.6rem; border-radius:10px; border:1px solid rgba(0,201,167,0.45); background:rgba(0,201,167,0.14); text-align:center;">
                    <div><b>Dashboard</b></div><div class="hint">interactive insights</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(agent_metrics, base_metrics, show_degradation):
    cols = st.columns(4)
    cards = [
        ("💷 Total Cost (£)", "cost", True),
        ("⚡ Peak Demand (kW)", "peak", True),
        ("😊 Comfort Score (%)", "comfort", False),
        ("🔋 Battery Health (%)", "battery", False),
    ]
    for col, (label, key, lower_better) in zip(cols, cards):
        value = agent_metrics[key]
        delta_txt = "-"
        delta_cls = "kpi-delta-good"
        if base_metrics:
            delta = value - base_metrics[key]
            better = delta < 0 if lower_better else delta > 0
            delta_cls = "kpi-delta-good" if better else "kpi-delta-bad"
            delta_txt = f"{delta:+.2f} vs baseline"
        if key == "battery" and not show_degradation:
            delta_txt = "degradation hidden"
        col.markdown(
            f"""
            <div class="card kpi-card">
                <div>{label}</div>
                <div class="kpi-value">{value:.2f}</div>
                <div class="{delta_cls}">{delta_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def energy_flow_chart(agent_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agent_df["hour"], y=agent_df["pv"], mode="lines", name="PV Generation", line={"color": "#FACC15"}))
    fig.add_trace(
            go.Scatter(
            x=agent_df["hour"],
            y=agent_df["pv"],
            mode="none",
            fill="tozeroy",
            fillcolor="rgba(250,204,21,0.15)",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=agent_df["hour"], y=agent_df["load"], mode="lines", name="Load Demand", line={"color": "#EF4444"}))
    fig.add_trace(go.Scatter(x=agent_df["hour"], y=agent_df["grid_import"], mode="lines", name="Grid Import", line={"color": "#3B82F6"}))
    fig.add_trace(go.Scatter(x=agent_df["hour"], y=agent_df["battery_flow"], mode="lines", name="Battery C/D", line={"color": "#22C55E"}))
    for h in config.PEAK_HOURS:
        fig.add_vline(x=h, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="kW", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


def soc_price_chart(agent_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agent_df["hour"], y=agent_df["soc"] * 100, mode="lines", fill="tozeroy", name="SoC (%)", line={"color": "#00C9A7"}))
    fig.add_trace(
        go.Scatter(
            x=agent_df["hour"],
            y=agent_df["price"],
            mode="lines",
            line={"color": "#FB923C", "dash": "dash"},
            name="Price",
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=360,
        yaxis={"title": "SoC (%)"},
        yaxis2={"title": "Price", "overlaying": "y", "side": "right"},
        xaxis_title="Hour",
    )
    return fig


def action_distribution_chart(agent_df):
    action_counts = agent_df["action"].value_counts().reindex([0, 1, 2, 3, 4], fill_value=0)
    fig = go.Figure(
        go.Bar(
            y=[ACTION_LABELS[a] for a in action_counts.index],
            x=action_counts.values,
            orientation="h",
            marker_color=[ACTION_COLORS[a] for a in action_counts.index],
        )
    )
    fig.update_layout(template="plotly_dark", height=360, xaxis_title="Hours", yaxis_title="Action")
    return fig


def reward_breakdown_chart(reward_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=reward_df["hour"], y=reward_df["peak_component"], name="Peak"))
    fig.add_trace(go.Bar(x=reward_df["hour"], y=reward_df["cost_component"], name="Cost"))
    fig.add_trace(go.Bar(x=reward_df["hour"], y=reward_df["comfort_component"], name="Comfort"))
    fig.update_layout(barmode="relative", template="plotly_dark", height=360, xaxis_title="Hour", yaxis_title="Reward contribution")
    return fig


def radar_chart(agent_metrics, base_metrics):
    categories = ["Cost", "Peak", "Comfort", "Battery", "Reward"]
    agent_reward = st.session_state["episode_agent_df"]["reward"].sum() if "episode_agent_df" in st.session_state else 0.0
    base_reward = st.session_state["episode_base_df"]["reward"].sum() if "episode_base_df" in st.session_state and not st.session_state["episode_base_df"].empty else 0.0
    agent_vals = [agent_metrics["cost"], agent_metrics["peak"], agent_metrics["comfort"], agent_metrics["battery"], agent_reward]
    base_vals = [base_metrics["cost"], base_metrics["peak"], base_metrics["comfort"], base_metrics["battery"], base_reward]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=agent_vals, theta=categories, fill="toself", name="Agent", line={"color": "#00C9A7"}))
    fig.add_trace(go.Scatterpolar(r=base_vals, theta=categories, fill="toself", name="Baseline", line={"color": "#845EF7"}))
    fig.update_layout(template="plotly_dark", height=360)
    return fig


def render_timeline_table(agent_df):
    table_df = agent_df.copy()
    table_df["Action"] = table_df["action"].map(ACTION_LABELS)
    table_df = table_df.rename(
        columns={
            "hour": "Hour",
            "pv": "PV",
            "load": "Load",
            "price": "Price",
            "grid_import": "Grid Import",
            "soc": "SoC",
            "reward": "Reward",
        }
    )
    st.dataframe(
        table_df[["Hour", "PV", "Load", "Price", "Action", "Grid Import", "SoC", "Reward"]]
        .style.apply(
            lambda row: [
                (
                    "background-color: rgba(255,69,58,0.20)"
                    if (row["Grid Import"] >= table_df["Grid Import"].max() and c != "Action")
                    else ""
                )
                for c in row.index
            ],
            axis=1,
        )
        .apply(
            lambda row: [
                "background-color: rgba(245,158,11,0.25)" if (row["Action"] == "comfort_sacrifice" and c == "Action") else ""
                for c in row.index
            ],
            axis=1,
        ),
        use_container_width=True,
    )
    st.download_button(
        "Download CSV",
        data=table_df.to_csv(index=False),
        file_name="episode_timeline.csv",
        mime="text/csv",
    )


def render_copy_button(agent_metrics, base_metrics):
    summary_text = "\n".join(
        [
            "MO-RL Episode Summary",
            f"Cost (£): {agent_metrics['cost']:.2f} | Baseline: {base_metrics['cost']:.2f}" if base_metrics else f"Cost (£): {agent_metrics['cost']:.2f}",
            f"Peak (kW): {agent_metrics['peak']:.2f}",
            f"Comfort (%): {agent_metrics['comfort']:.2f}",
            f"Battery (%): {agent_metrics['battery']:.2f}",
        ]
    )
    components.html(
        f"""
        <button onclick="navigator.clipboard.writeText({summary_text!r})"
                style="background:#00C9A7;color:#032B25;border:none;padding:8px 12px;border-radius:8px;font-weight:700;cursor:pointer;">
            Copy Results to Clipboard
        </button>
        """,
        height=40,
    )


@st.cache_data(show_spinner=False)
def load_pareto_data(cache_key=0):
    _ = cache_key
    path = config.RESULTS_DIR / config.PARETO_DATA_FILENAME
    if not path.exists():
        return pd.DataFrame()
    try:
        import json

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results", [])
        if not isinstance(results, list):
            return pd.DataFrame()
        return pd.json_normalize(results)
    except Exception:
        return pd.DataFrame()


def compute_pareto_optimal(df):
    pts = df[["metrics.mean_total_cost", "metrics.mean_peak_demand", "metrics.mean_min_comfort"]].to_numpy()
    flags = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        for j, q in enumerate(pts):
            if i == j:
                continue
            dominates = (q[0] <= p[0] and q[1] <= p[1] and q[2] >= p[2]) and (q[0] < p[0] or q[1] < p[1] or q[2] > p[2])
            if dominates:
                flags[i] = False
                break
    return flags


def run_shell_command(command):
    try:
        completed = subprocess.run(command, shell=True, cwd=str(config.PROJECT_ROOT), capture_output=True, text=True, timeout=1200)
        return completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    except Exception as exc:
        return False, str(exc)


def render_pareto_page():
    st.subheader("Pareto Analysis")
    pareto_path = config.RESULTS_DIR / config.PARETO_DATA_FILENAME
    cache_key = pareto_path.stat().st_mtime_ns if pareto_path.exists() else 0
    rows = load_pareto_data(cache_key=cache_key)
    if rows.empty:
        st.warning("Pareto data not found.")
        if st.button("Run Pareto Sweep"):
            with st.spinner("Running Pareto sweep..."):
                ok, msg = run_shell_command("python experiments/pareto_sweep.py --seed 42 --timesteps 5000")
                if ok:
                    st.success("Pareto sweep completed. Reload page.")
                else:
                    st.error(msg)
        return

    required_cols = {
        "setting_id",
        "metrics.mean_total_cost",
        "metrics.mean_peak_demand",
        "metrics.mean_min_comfort",
    }
    if not required_cols.issubset(set(rows.columns)):
        st.error("Pareto file exists but has an unexpected format.")
        return

    optimal_mask = compute_pareto_optimal(rows)
    select_idx = st.slider("Select weight configuration", 0, len(rows) - 1, 0)

    fig3d = go.Figure()
    fig3d.add_trace(
        go.Scatter3d(
            x=rows["metrics.mean_total_cost"],
            y=rows["metrics.mean_peak_demand"],
            z=rows["metrics.mean_min_comfort"],
            mode="markers",
            marker={"size": 5, "color": np.where(optimal_mask, "#00C9A7", "#6B7280")},
            text=rows["setting_id"],
            name="All points",
        )
    )
    sel = rows.iloc[select_idx]
    fig3d.add_trace(
        go.Scatter3d(
            x=[sel["metrics.mean_total_cost"]],
            y=[sel["metrics.mean_peak_demand"]],
            z=[sel["metrics.mean_min_comfort"]],
            mode="markers",
            marker={"size": 9, "color": "#845EF7"},
            name="Selected",
        )
    )
    fig3d.update_layout(template="plotly_dark", height=500, scene={"xaxis_title": "Cost", "yaxis_title": "Peak", "zaxis_title": "Comfort"})
    st.plotly_chart(fig3d, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, (xcol, ycol, ttl) in zip(
        [c1, c2, c3],
        [
            ("metrics.mean_total_cost", "metrics.mean_peak_demand", "Cost vs Peak"),
            ("metrics.mean_total_cost", "metrics.mean_min_comfort", "Cost vs Comfort"),
            ("metrics.mean_peak_demand", "metrics.mean_min_comfort", "Peak vs Comfort"),
        ],
    ):
        f = go.Figure()
        f.add_trace(go.Scatter(x=rows[xcol], y=rows[ycol], mode="markers", marker={"color": np.where(optimal_mask, "#00C9A7", "#9CA3AF")}))
        f.update_layout(template="plotly_dark", height=280, title=ttl)
        col.plotly_chart(f, use_container_width=True)


def render_algorithm_page():
    st.subheader("Algorithm Comparison")
    comp_path = config.RESULTS_DIR / "algorithm_comparison.csv"
    curves_path = config.RESULTS_DIR / "algorithm_training_curves.csv"
    final_path = config.RESULTS_DIR / "final_summary.csv"
    if not comp_path.exists():
        st.warning("algorithm_comparison.csv not found.")
        if st.button("Run Comparison"):
            with st.spinner("Running algorithm comparison..."):
                ok, msg = run_shell_command("python experiments/compare_algorithms.py --seed 42 --timesteps 5000")
                if ok:
                    st.success("Comparison completed. Reload page.")
                else:
                    st.error(msg)
        return

    comp_df = pd.read_csv(comp_path)
    final_df = _safe_read_csv(final_path)
    baseline_row = final_df[final_df["algorithm"].str.contains("Baseline", case=False)] if not final_df.empty and "algorithm" in final_df.columns else pd.DataFrame()
    chart_df = comp_df.copy()
    if not baseline_row.empty:
        b = baseline_row.iloc[0]
        chart_df = pd.concat(
            [
                chart_df,
                pd.DataFrame(
                    [
                        {
                            "algorithm": "Rule-Based",
                            "mean_reward": b.get("mean_reward", np.nan),
                            "mean_cost": b.get("mean_cost", np.nan),
                            "mean_peak_demand": b.get("mean_peak_demand", np.nan),
                            "mean_comfort": b.get("mean_comfort", np.nan),
                            "training_time_sec": b.get("training_time_sec", np.nan),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    metrics = ["mean_reward", "mean_cost", "mean_peak_demand", "mean_comfort"]
    fig = go.Figure()
    for algo in chart_df["algorithm"]:
        row = chart_df[chart_df["algorithm"] == algo].iloc[0]
        fig.add_trace(go.Bar(name=algo, x=metrics, y=[row[m] for m in metrics]))
    fig.update_layout(template="plotly_dark", barmode="group", height=420)
    st.plotly_chart(fig, use_container_width=True)

    curves_df = _safe_read_csv(curves_path)
    if not curves_df.empty and "algorithm" in curves_df.columns:
        cf = go.Figure()
        for algo in curves_df["algorithm"].unique():
            sub = curves_df[curves_df["algorithm"] == algo]
            cf.add_trace(go.Scatter(x=sub["episode"], y=sub["episode_reward"], mode="lines", name=algo))
        cf.update_layout(template="plotly_dark", height=360, xaxis_title="Timesteps/Episode", yaxis_title="Reward")
        st.plotly_chart(cf, use_container_width=True)

    summary = chart_df.set_index("algorithm")[metrics]
    winner = {}
    winner["mean_reward"] = summary["mean_reward"].idxmax()
    winner["mean_cost"] = summary["mean_cost"].idxmin()
    winner["mean_peak_demand"] = summary["mean_peak_demand"].idxmin()
    winner["mean_comfort"] = summary["mean_comfort"].idxmax()
    st.dataframe(summary.style.apply(lambda col: ["background-color: rgba(0,201,167,0.22)" if idx == winner[col.name] else "" for idx in col.index], axis=0))


def dashboard_main_page(dataset, ppo_model, sac_model):
    available_algos = ["PPO"] if ppo_model is not None else []
    if sac_model is not None:
        available_algos.append("SAC")

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Dashboard"

    st.sidebar.markdown(
        '<div class="sidebar-title"><span class="sidebar-bolt">⚡</span>MO-RL Console</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    for page_key, page_label in NAV_ITEMS:
        if st.sidebar.button(
            page_label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if st.session_state["active_page"] == page_key else "secondary",
        ):
            st.session_state["active_page"] = page_key
    page = st.session_state["active_page"]
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if page != "Dashboard":
        return page

    st.sidebar.markdown("### Agent Configuration")
    algo = st.sidebar.radio("Algorithm", options=available_algos or ["PPO"], help="Choose which trained controller to execute.")
    l_cost = st.sidebar.slider("λ_cost", 0.0, 1.0, 0.33, 0.01, help="Weight assigned to minimizing electricity cost.")
    l_peak = st.sidebar.slider("λ_peak", 0.0, 1.0, 0.33, 0.01, help="Weight assigned to minimizing peak demand.")
    l_comfort = st.sidebar.slider("λ_comfort", 0.0, 1.0, 0.34, 0.01, help="Weight assigned to preserving user comfort.")
    weights = normalize_weights(l_cost, l_peak, l_comfort)
    donut = go.Figure(go.Pie(labels=["Cost", "Peak", "Comfort"], values=[weights["cost"], weights["peak"], weights["comfort"]], hole=0.62, marker={"colors": ["#00C9A7", "#3B82F6", "#845EF7"]}))
    donut.update_layout(template="plotly_dark", height=260, margin={"t": 10, "b": 10, "l": 10, "r": 10})
    st.sidebar.plotly_chart(donut, use_container_width=True)

    st.sidebar.markdown("### Episode Settings")
    season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"], help="Select which season segment from the 8760-hour dataset is sampled.")
    st.sidebar.caption("Episode length: 24 hours")
    battery_capacity = st.sidebar.slider("Battery capacity (kWh)", 5.0, 20.0, float(config.BATTERY_CAPACITY_KWH), 0.5, help="Overrides battery capacity for this run only.")
    initial_soc_pct = st.sidebar.slider("Initial SoC (%)", 0, 100, int(config.INITIAL_SOC * 100), 1, help="Initial battery state of charge for this episode.")

    st.sidebar.markdown("### Comparison")
    show_baseline = st.sidebar.toggle("Show Rule-Based Baseline", value=True, help="Runs baseline controller on same episode for side-by-side comparison.")
    show_deg = st.sidebar.toggle("Show Battery Degradation", value=True, help="Display battery health degradation metrics.")

    st.sidebar.markdown('<div class="hint">Press R to run episode</div>', unsafe_allow_html=True)
    run_button = st.sidebar.button("▶ Run Episode", use_container_width=True, type="primary")

    filtered = dataset[dataset["season"] == season]
    dates = sorted(filtered["episode_date"].unique().tolist()) if not filtered.empty else []
    selected_date = dates[0] if dates else None
    if selected_date is None:
        st.error("No episode date available for selected season.")
        return page

    if run_button:
        if (algo == "PPO" and ppo_model is None) or (algo == "SAC" and sac_model is None):
            st.error("Selected model is not available. Train model first and retry.")
        else:
            with st.spinner("Agent is making decisions..."):
                model = ppo_model if algo == "PPO" else sac_model
                agent_df, base_df, reward_df = run_single_episode(
                    policy_name=algo,
                    model=model,
                    reward_weights=weights,
                    start_date=selected_date,
                    battery_capacity_kwh=battery_capacity,
                    initial_soc=initial_soc_pct / 100.0,
                    compare_baseline=show_baseline,
                )
            st.session_state["episode_agent_df"] = agent_df
            st.session_state["episode_base_df"] = base_df
            st.session_state["episode_reward_df"] = reward_df
            st.session_state["episode_algo"] = algo

    if "episode_agent_df" not in st.session_state or st.session_state["episode_agent_df"].empty:
        render_landing()
        return page

    agent_df = st.session_state["episode_agent_df"]
    base_df = st.session_state.get("episode_base_df", pd.DataFrame())
    reward_df = st.session_state.get("episode_reward_df", pd.DataFrame())
    agent_metrics = compute_metrics(agent_df)
    base_metrics = compute_metrics(base_df) if show_baseline and not base_df.empty else None

    if base_metrics:
        if agent_metrics["cost"] < base_metrics["cost"] and agent_metrics["peak"] < base_metrics["peak"] and agent_metrics["comfort"] > base_metrics["comfort"]:
            st.balloons()

    render_kpi_cards(agent_metrics, base_metrics, show_deg)
    render_copy_button(agent_metrics, base_metrics)

    st.subheader("Energy Flow Chart")
    energy_flow_chart(agent_df)

    c1, c2 = st.columns(2)
    c1.subheader("Battery SoC & Price Signal")
    c1.plotly_chart(soc_price_chart(agent_df), use_container_width=True)
    c2.subheader("Action Distribution")
    c2.plotly_chart(action_distribution_chart(agent_df), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.subheader("Reward Breakdown")
    c3.plotly_chart(reward_breakdown_chart(reward_df), use_container_width=True)
    c4.subheader("Agent vs Baseline Radar")
    if show_baseline and base_metrics:
        c4.plotly_chart(radar_chart(agent_metrics, base_metrics), use_container_width=True)
    else:
        c4.info("Enable baseline comparison to view radar chart.")

    st.subheader("Episode Timeline")
    render_timeline_table(agent_df)
    return page


def main():
    st.set_page_config(page_title="MO-RL Peak Shaving | Smart Grid AI", page_icon="⚡", layout="wide")
    inject_css()

    dataset = load_dataset()
    ppo_model = load_ppo_model()
    sac_model = load_sac_model()
    dataset_rows = int(len(dataset)) if dataset is not None else 0
    render_header(model_loaded=(ppo_model is not None or sac_model is not None), dataset_rows=dataset_rows)

    if dataset is None:
        st.error(f"Dataset missing at `{config.REAL_DATASET_PATH}`. Please place the file and rerun.")
        st.stop()

    selected = dashboard_main_page(dataset, ppo_model, sac_model)
    if selected == "Pareto Analysis":
        render_pareto_page()
    elif selected == "Algorithm Comparison":
        render_algorithm_page()

    st.markdown("---")
    st.caption("MO-RL-PeakShaving | Final Year Project | Built with Streamlit & Stable-Baselines3")


if __name__ == "__main__":
    main()
