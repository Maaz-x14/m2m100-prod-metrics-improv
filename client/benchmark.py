#!/usr/bin/env python3
"""
benchmark.py — Latency & throughput test for the M2M100 transliteration API.

Usage:
    python benchmark.py --host http://<DROPLET_IP>:8000
    python benchmark.py --host http://localhost:8000 --runs 50
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# ── Test sentences (Urdu) ───────────────────────────────────────────────────────
SAMPLE_SENTENCES = [
    "مجھے کھانا چاہیے",
    "آپ کہاں جا رہے ہیں",
    "یہ کتاب میری ہے",
    "کل موسم اچھا تھا",
    "پاکستان زندہ باد",
    "میرا نام علی ہے اور میں لاہور میں رہتا ہوں",
    "آج صبح میں نے چائے پی اور پھر آفس گیا",
    (
        "اس ناول میں لکھا تھا کہ ایک لڑکے نے اپنی محنت اور لگن سے "
        "ایک بڑا مقام حاصل کیا اور اپنی زندگی میں بہت کمی آئی"
    ),
]


def single_request(client: httpx.Client, base_url: str, text: str, beams: int = 4):
    t0 = time.perf_counter()
    r = client.post(
        f"{base_url}/translate",
        json={"text": text, "num_beams": beams},
        timeout=60.0,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    return {
        "input": text,
        "output": data["translation"],
        "server_ms": data["latency_ms"],
        "wall_ms": round(wall_ms, 2),
    }


def batch_request(client: httpx.Client, base_url: str, texts: list[str], beams: int = 4):
    t0 = time.perf_counter()
    r = client.post(
        f"{base_url}/translate",
        json={"texts": texts, "num_beams": beams},
        timeout=120.0,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    return {
        "inputs": texts,
        "outputs": data["translations"],
        "server_ms": data["latency_ms"],
        "wall_ms": round(wall_ms, 2),
    }


def run_single_benchmarks(base_url: str, runs: int, beams: int):
    print(f"\n{'═'*64}")
    print(f"  SINGLE-REQUEST BENCHMARK  |  runs={runs}  beams={beams}")
    print(f"{'═'*64}")

    with httpx.Client() as client:
        # Check health
        health = client.get(f"{base_url}/health").json()
        print(f"  Device : {health['device']}")
        if health.get("gpu_mem_mb"):
            print(f"  GPU mem: {health['gpu_mem_mb']} MB")

        # Warm-up (server already does a warm-up at startup, but ensure HTTP path is warm)
        print(f"\n  Warm-up pass …")
        single_request(client, base_url, SAMPLE_SENTENCES[0], beams)

        wall_times = []
        gen_times  = []

        print(f"  Running {runs} timed requests …\n")
        for i in range(runs):
            sentence = SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)]
            result   = single_request(client, base_url, sentence, beams)
            wall_times.append(result["wall_ms"])
            gen_times.append(result["server_ms"]["generate_ms"])
            if i < 5 or i == runs - 1:
                print(f"  [{i+1:3d}] wall={result['wall_ms']:7.1f} ms  "
                      f"gen={result['server_ms']['generate_ms']:7.1f} ms  "
                      f"→  {result['output'][:50]}")

        print(f"\n  ── Wall-clock latency (ms) ──────────────────────────")
        print(f"     p50  : {statistics.median(wall_times):.1f}")
        print(f"     p90  : {sorted(wall_times)[int(runs*0.9)]:.1f}")
        print(f"     p99  : {sorted(wall_times)[int(runs*0.99)]:.1f}")
        print(f"     mean : {statistics.mean(wall_times):.1f}")
        print(f"     min  : {min(wall_times):.1f}")
        print(f"     max  : {max(wall_times):.1f}")
        print(f"\n  ── Server-side generate time (ms) ───────────────────")
        print(f"     p50  : {statistics.median(gen_times):.1f}")
        print(f"     mean : {statistics.mean(gen_times):.1f}")


def run_batch_benchmarks(base_url: str, batch_sizes: list[int], beams: int):
    print(f"\n{'═'*64}")
    print(f"  BATCH BENCHMARK  |  beams={beams}")
    print(f"{'═'*64}")
    with httpx.Client() as client:
        for bs in batch_sizes:
            texts  = (SAMPLE_SENTENCES * 10)[:bs]
            result = batch_request(client, base_url, texts, beams)
            per_item = result["wall_ms"] / bs
            print(f"  batch={bs:3d}  wall={result['wall_ms']:7.1f} ms  "
                  f"per-item={per_item:6.1f} ms  "
                  f"gen={result['server_ms']['generate_ms']:.1f} ms")


def run_concurrency_test(base_url: str, concurrency: int, total: int, beams: int):
    print(f"\n{'═'*64}")
    print(f"  CONCURRENCY TEST  |  threads={concurrency}  total={total}  beams={beams}")
    print(f"{'═'*64}")
    results = []

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [
                pool.submit(
                    single_request, client, base_url,
                    SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)], beams
                )
                for i in range(total)
            ]
            for f in as_completed(futs):
                try:
                    results.append(f.result()["wall_ms"])
                except Exception as e:
                    print(f"  ERROR: {e}")

    if results:
        print(f"  Completed: {len(results)}/{total}")
        print(f"  p50 wall : {statistics.median(results):.1f} ms")
        print(f"  p99 wall : {sorted(results)[int(len(results)*0.99)]:.1f} ms")
        print(f"  mean     : {statistics.mean(results):.1f} ms")


def edge_case_tests(base_url: str):
    print(f"\n{'═'*64}")
    print(f"  EDGE CASE TESTS")
    print(f"{'═'*64}")
    with httpx.Client() as client:
        # Empty
        r = client.post(f"{base_url}/translate", json={"text": ""})
        print(f"  Empty string        → HTTP {r.status_code}")

        # Very short
        result = single_request(client, base_url, "ha", beams=1)
        print(f"  Single syllable     → '{result['output']}' ({result['wall_ms']:.0f} ms)")

        # English (out-of-domain)
        result = single_request(client, base_url, "this is an english sentence", beams=4)
        print(f"  English input       → '{result['output']}' ({result['wall_ms']:.0f} ms)")

        # Long input
        long_text = "mera ghar bohot door hai aur rasta bhi mushkil hai " * 6
        result = single_request(client, base_url, long_text[:500], beams=4)
        print(f"  Long input (~500c)  → '{result['output'][:60]}…' ({result['wall_ms']:.0f} ms)")

        # Greedy vs beam
        for beams in (1, 4, 8):
            result = single_request(client, base_url, "aap ka shukriya", beams=beams)
            print(f"  beams={beams}  → '{result['output']}'  gen={result['server_ms']['generate_ms']:.0f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="http://localhost:8000")
    parser.add_argument("--runs",  type=int, default=30, help="Single-request benchmark runs")
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--skip-concurrency", action="store_true")
    args = parser.parse_args()

    print(f"\nTarget: {args.host}")

    run_single_benchmarks(args.host, args.runs, args.beams)
    run_batch_benchmarks(args.host, [2, 4, 8, 16], args.beams)
    edge_case_tests(args.host)

    if not args.skip_concurrency:
        run_concurrency_test(args.host, concurrency=4, total=20, beams=args.beams)


if __name__ == "__main__":
    main()
