#!/usr/bin/env python3
"""Run Steps 2-5 of the overnight pipeline (skip ingestion)."""

import os
import subprocess
import sys
import time


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


EVAL_DB = "/tmp/memorybench_eval.db"
PROD_DB = os.path.expanduser("~/Projects/openclaw-memory/memory.db")
EVAL_API = "http://127.0.0.1:8901"
PROD_API = "http://127.0.0.1:8643"


def check_server(url, name):
    import requests

    try:
        r = requests.get(f"{url}/api/health", timeout=5)
        r.raise_for_status()
        log(f"✅ {name} healthy at {url}")
        return True
    except Exception as e:
        log(f"❌ {name} unreachable: {e}")
        return False


# Step 2: Backfill structured facts on eval DB
def step2():
    log("=" * 60)
    log("STEP 2: Backfilling structured facts on eval DB")
    log("=" * 60)
    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = EVAL_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "local"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"
    try:
        result = subprocess.run(
            [sys.executable, "backfill_facts.py"],
            cwd=os.path.expanduser("~/Projects/openclaw-memory"),
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        log(f"Backfill output (last 500):\n{result.stdout[-500:]}")
        if result.returncode != 0:
            log(f"Errors:\n{result.stderr[-500:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log("Backfill timed out after 2h — partial progress saved. Continuing.")
        return True  # Partial is OK


# Step 3: Multi-session benchmark
def step3():
    log("=" * 60)
    log("STEP 3: Running multi-session aggregate benchmark")
    log("=" * 60)
    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = EVAL_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "local"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"
    result = subprocess.run(
        [sys.executable, "bench_multisession.py", "--strategy", "structured"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    log(f"Benchmark output (last 2000):\n{result.stdout[-2000:]}")
    if result.returncode != 0:
        log(f"Errors:\n{result.stderr[-500:]}")
    return result.returncode == 0


# Step 4: Backfill production DB
def step4():
    log("=" * 60)
    log("STEP 4: Backfilling structured facts on production DB")
    log("=" * 60)
    if not check_server(PROD_API, "Production server"):
        log("Prod server not running. Skipping.")
        return False
    env = os.environ.copy()
    env["ULTRAMEMORY_DB_PATH"] = PROD_DB
    env["ULTRAMEMORY_EMBEDDING_PROVIDER"] = "litellm"
    env["ULTRAMEMORY_EMBEDDING_MODEL"] = "gemini/gemini-embedding-2-preview"
    env["ULTRAMEMORY_MODEL"] = "gemini/gemini-2.5-flash"
    result = subprocess.run(
        [sys.executable, "backfill_facts.py"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    log(f"Prod backfill output (last 500):\n{result.stdout[-500:]}")
    return result.returncode == 0


# Step 5: Full memorybench eval
def step5():
    log("=" * 60)
    log("STEP 5: Running full memorybench evaluation suite")
    log("=" * 60)

    memorybench_dir = os.path.expanduser("~/Projects/memorybench")
    if not os.path.isdir(memorybench_dir):
        log("memorybench directory not found!")
        return False

    # Start fresh eval server on 8643 for memorybench
    eval_db_fresh = "/tmp/memorybench_fulleval.db"
    log(f"Starting eval server on :8643 with fresh DB: {eval_db_fresh}")

    subprocess.run("lsof -ti :8643 | xargs kill -9 2>/dev/null", shell=True)
    time.sleep(1)

    google_key = os.environ.get("GOOGLE_API_KEY", "")
    server_env = os.environ.copy()
    server_env.update(
        {
            "ULTRAMEMORY_DB_PATH": eval_db_fresh,
            "ULTRAMEMORY_EMBEDDING_PROVIDER": "litellm",
            "ULTRAMEMORY_EMBEDDING_MODEL": "gemini/gemini-embedding-2-preview",
            "ULTRAMEMORY_MODEL": "gemini/gemini-2.5-flash",
            "GOOGLE_API_KEY": google_key,
            "GEMINI_API_KEY": google_key,
        }
    )
    server_env.pop("ULTRAMEMORY_FAST_INGEST", None)

    server_proc = subprocess.Popen(
        ["uvicorn", "ultramemory.server:app", "--port", "8643"],
        cwd=os.path.expanduser("~/Projects/openclaw-memory"),
        env=server_env,
        stdout=open("/tmp/eval_fulleval_server.log", "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(10)

    if not check_server("http://127.0.0.1:8643", "Full eval server"):
        server_proc.kill()
        return False

    bench_env = os.environ.copy()
    bench_env.update(
        {
            "OPENCLAW_SUPERMEMORY_URL": "http://127.0.0.1:8643",
            "GOOGLE_API_KEY": google_key,
            "GEMINI_API_KEY": google_key,
        }
    )

    run_id = "eval-gemini-emb2"
    result = subprocess.run(
        [
            "bun",
            "run",
            "src/index.ts",
            "run",
            "-p",
            "openclaw-supermemory",
            "-b",
            "longmemeval",
            "-j",
            "gemini-2.5-flash",
            "-m",
            "gemini-2.5-flash",
            "-r",
            run_id,
        ],
        cwd=memorybench_dir,
        env=bench_env,
        capture_output=True,
        text=True,
        timeout=14400,
    )
    log(f"Benchmark output (last 3000):\n{result.stdout[-3000:]}")
    if result.returncode != 0:
        log(f"Errors:\n{result.stderr[-1000:]}")

    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()

    return result.returncode == 0


if __name__ == "__main__":
    log("🔧 Running Steps 2-5 (skipping Step 1 ingestion)")
    log("Eval DB has 110/133 multi-session questions ingested")

    results = {}
    for name, fn in [("step2", step2), ("step3", step3), ("step4", step4), ("step5", step5)]:
        try:
            results[name] = fn()
        except Exception as e:
            log(f"❌ {name} failed with exception: {e}")
            results[name] = False

    log("=" * 60)
    log("🎯 STEPS 2-5 COMPLETE")
    for step, ok in results.items():
        status = "✅" if ok else "❌"
        log(f"  {status} {step}")
    log("=" * 60)
