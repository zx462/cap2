#!/usr/bin/env python3
"""
Flask backend for WiFi simulation demo.
Runs RL and BEB eval scripts in parallel and streams progress via SSE.
"""

import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR = Path("/Users/gim-yeojun/cap_final6/mappo")
MODEL_DIR = (
    REPO_DIR
    / "model/WiFi_v9/mappo"
    / "wifi_v9_train_airtime50ms_m15m25_s3s5_parallel_vec4_d2lt_mldsucc1_sld07_10_ntop1_cidle05_1600k_lr1e4_ent5e3_seed1"
)

# ── Mu bounds per scenario (for clamping) ─────────────────────────────────────
SCENARIO_MU_BOUNDS = {
    "high": (0.09, 0.12),
    "mid":  (0.055, 0.085),
    "low":  (0.01, 0.04),
}

MU_SPREAD = 0.008   # ±0.008 around the exact mu → narrow range, episode별 소폭 변동

# ── Global state for current evaluation run ────────────────────────────────────
_event_queue: queue.Queue = queue.Queue()
_run_lock = threading.Lock()
_running = False


# ── Script builders ────────────────────────────────────────────────────────────

def _base_args(mu_min: float, mu_max: float, episodes: int, seed: int) -> list:
    return [
        "--env_name", "WiFi_v9",
        "--num_mld", "15",
        "--num_sld", "5",
        "--max_mld", "30",
        "--max_sld", "10",
        "--round_length", "500",
        "--mu_min", str(mu_min),
        "--mu_max", str(mu_max),
        "--eta", "1.0",
        "--zeta", "1.0",
        "--c_idle", "0.5",
        "--theta_scale", "1.0",
        "--sld_target_low_scale", "0.7",
        "--sld_target_high_scale", "1.0",
        "--sld_target_bonus", "0.0",
        "--mld_success_reward", "1.0",
        "--non_top_tx_penalty", "1.0",
        "--eval_episodes", str(episodes),
        "--eval_duration_sec", "30.0",
        "--slot_time_sec", "9e-6",
        "--debug_prob_steps", "0",
        "--seed", str(seed),
    ]


def _rl_cmd(mu_min: float, mu_max: float, episodes: int, seed: int) -> list:
    return [
        sys.executable, "-m", "onpolicy.scripts.eval.eval_wifi_v9_rl_mbps",
        "--experiment_name", "wifi_v9_rl_mbps_sim_demo",
        *_base_args(mu_min, mu_max, episodes, seed),
        "--stochastic",
        "--use_wandb",
        "--model_dir", str(MODEL_DIR),
    ]


def _beb_cmd(mu_min: float, mu_max: float, episodes: int, seed: int) -> list:
    return [
        sys.executable, "-m", "onpolicy.scripts.eval.eval_wifi_v9_beb_mbps",
        "--experiment_name", "wifi_v9_beb_mbps_sim_demo",
        *_base_args(mu_min, mu_max, episodes, seed),
        "--use_wandb",
    ]


# ── Output parsers ─────────────────────────────────────────────────────────────

RE_EPISODE = re.compile(
    r"\[(?P<policy>RL|BEB) Mbps Eval\] Episode (?P<ep>\d+)/(?P<total>\d+)"
    r" \| mbps/system=(?P<mbps_sys>[0-9.]+)"
    r" \| mbps/mld_total=(?P<mbps_mld>[0-9.]+)"
    r" \| mbps/sld_total=(?P<mbps_sld>[0-9.]+)"
    r" \| tx_ratio=(?P<tx>[0-9.]+)"
)

RE_SUMMARY_HDR = re.compile(r"\[(RL|BEB) Mbps Summary\]")
RE_SUMMARY_LINE = re.compile(r"^\s{2}(?P<key>[\w/.]+):\s+(?P<val>[0-9.eE+\-]+)")


def _stream_proc(proc: subprocess.Popen, policy: str, eq: queue.Queue):
    """Read stdout of a subprocess and push parsed events to queue."""
    in_summary = False
    summary: dict = {}

    for raw in proc.stdout:
        line = raw.rstrip("\n")

        # Raw log
        eq.put({"type": "log", "policy": policy, "line": line})

        # Episode result
        m = RE_EPISODE.search(line)
        if m:
            eq.put({
                "type": "episode",
                "policy": policy.lower(),
                "ep": int(m.group("ep")),
                "total": int(m.group("total")),
                "mbps_system": float(m.group("mbps_sys")),
                "mbps_mld": float(m.group("mbps_mld")),
                "mbps_sld": float(m.group("mbps_sld")),
                "tx_ratio": float(m.group("tx")),
            })
            continue

        # Summary header
        if RE_SUMMARY_HDR.search(line):
            in_summary = True
            continue

        # Summary key-value lines
        if in_summary:
            ms = RE_SUMMARY_LINE.match(line)
            if ms:
                summary[ms.group("key")] = float(ms.group("val"))
            else:
                in_summary = False

    proc.wait()
    rc = proc.returncode

    if summary:
        eq.put({
            "type": "summary",
            "policy": policy.lower(),
            "mbps_system": summary.get("mbps/system", 0.0),
            "mbps_mld": summary.get("mbps/mld_total", 0.0),
            "mbps_sld": summary.get("mbps/sld_total", 0.0),
            "success_24": summary.get("events/2_4GHz/success", 0.0),
            "success_5": summary.get("events/5GHz/success", 0.0),
            "collision_rate": summary.get("collision_rate/system_per_event", 0.0),
            "success_rate": summary.get("success_rate/system_per_event", 0.0),
            "returncode": rc,
        })
    else:
        eq.put({
            "type": "error",
            "policy": policy.lower(),
            "msg": f"Script ended (rc={rc}) with no summary output.",
            "returncode": rc,
        })


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/api/start", methods=["POST"])
def start():
    global _running, _event_queue

    data = request.get_json(force=True, silent=True) or {}
    scenario  = data.get("scenario", "mid")
    episodes  = int(data.get("episodes", 4))
    mu_exact  = float(data.get("mu_exact", -1))

    if scenario not in SCENARIO_MU_BOUNDS:
        return jsonify({"error": f"unknown scenario '{scenario}'"}), 400

    with _run_lock:
        if _running:
            return jsonify({"error": "already running"}), 409
        _running = True

    # Fresh queue for this run
    _event_queue = queue.Queue()
    eq = _event_queue

    lo, hi = SCENARIO_MU_BOUNDS[scenario]
    if mu_exact < 0:
        mu_exact = (lo + hi) / 2
    mu_exact = max(lo, min(hi, mu_exact))
    mu_min   = max(0.01, mu_exact - MU_SPREAD)
    mu_max   = min(0.12, mu_exact + MU_SPREAD)

    # 매 실행마다 랜덤 시드 → 같은 파일이어도 실행할 때마다 다른 결과
    run_seed = random.randint(1, 100000)

    def launch():
        global _running
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_DIR) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONUNBUFFERED"] = "1"

            rl_proc = subprocess.Popen(
                _rl_cmd(mu_min, mu_max, episodes, run_seed),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_DIR),
                env=env,
            )
            beb_proc = subprocess.Popen(
                _beb_cmd(mu_min, mu_max, episodes, run_seed),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(REPO_DIR),
                env=env,
            )

            rl_thread = threading.Thread(
                target=_stream_proc, args=(rl_proc, "RL", eq), daemon=True
            )
            beb_thread = threading.Thread(
                target=_stream_proc, args=(beb_proc, "BEB", eq), daemon=True
            )
            rl_thread.start()
            beb_thread.start()
            rl_thread.join()
            beb_thread.join()

        except Exception as exc:
            eq.put({"type": "error", "policy": "server", "msg": str(exc)})
        finally:
            eq.put({"type": "done"})
            with _run_lock:
                _running = False

    threading.Thread(target=launch, daemon=True).start()

    return jsonify({
        "status":   "started",
        "scenario": scenario,
        "mu_exact": round(mu_exact, 4),
        "mu_min":   round(mu_min, 4),
        "mu_max":   round(mu_max, 4),
        "seed":     run_seed,
        "episodes": episodes,
    })


@app.route("/api/progress")
def progress():
    """SSE endpoint — streams events from the current run."""

    def generate():
        # yield keep-alive comment
        yield ": connected\n\n"
        while True:
            try:
                evt = _event_queue.get(timeout=60)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue

            yield f"data: {json.dumps(evt)}\n\n"

            if evt.get("type") == "done":
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/status")
def status():
    return jsonify({"running": _running})


if __name__ == "__main__":
    print("=" * 60)
    print(" WiFi Simulation Backend  →  http://localhost:5050")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5050, threaded=True)
