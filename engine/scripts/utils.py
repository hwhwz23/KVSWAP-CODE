import os
import re
import json
import argparse
import math
import csv
from collections import defaultdict

def _to_regular_dict(d):
    if isinstance(d, defaultdict):
        d = {k: _to_regular_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: _to_regular_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [_to_regular_dict(x) for x in d]
    return d


def parse_common_params(file_name, seed):
    common = {"seed": seed, "batch": None, "seqlen": None, "genlen": None}

    # shadowkv: 32668_bsz1_gen100_chunk16_r40.log
    m_shadow = re.match(r"(?P<seqlen>\d+)_bsz(?P<batch>\d+)_gen(?P<genlen>\d+)_chunk\d+_r\d+\.log$", file_name)
    if m_shadow:
        common["batch"] = int(m_shadow.group("batch"))
        common["seqlen"] = int(m_shadow.group("seqlen"))
        common["genlen"] = int(m_shadow.group("genlen"))
        return common

    # kvswap/infinigen: 1-16284-100_L4_none_clear_1_400_0-curr-emb_p0.125.log
    m_kv = re.match(r"(?P<batch>\d+)-(?P<seqlen>\d+)-(?P<genlen>\d+)_", file_name)
    if m_kv:
        common["batch"] = int(m_kv.group("batch"))
        common["seqlen"] = int(m_kv.group("seqlen"))
        common["genlen"] = int(m_kv.group("genlen"))

    return common


def parse_path_info(base_dir, log_path):
    rel = os.path.relpath(log_path, base_dir)
    parts = rel.split(os.sep)

    # shadowkv path:
    # logs/shadowkv/{disk}/{model}/budget{budget}/seed{seed}/{file}.log
    if len(parts) >= 7 and parts[0] == "logs" and parts[1] == "shadowkv":
        disk = parts[2]
        model = parts[3]
        budget_part = parts[4]
        seed_part = parts[5]
        file_name = parts[6]

        budget_m = re.match(r"budget(\d+)$", budget_part)
        seed_m = re.match(r"seed(\d+)$", seed_part)
        file_m = re.match(r"\d+_bsz\d+_gen\d+_chunk(?P<chunk>\d+)_r(?P<r>\d+)\.log$", file_name)
        if not (budget_m and seed_m and file_m):
            return None

        return {
            "method": "shadowkv",
            "disk": disk,
            "model": model,
            "params": {
                "budget": int(budget_m.group(1)),
                "chunk": int(file_m.group("chunk")),
                "r": int(file_m.group("r")),
            },
            "common": parse_common_params(file_name, int(seed_m.group(1))),
        }

    # kvswap / infinigen path:
    # logs/log/{disk}/{model}/{mode}/tg{tg}-ru{ru}/seed{seed}/{file}.log
    if len(parts) >= 8 and parts[0] == "logs" and parts[1] == "log":
        disk = parts[2]
        model = parts[3]
        mode = parts[4]
        tgru = parts[5]
        seed_part = parts[6]
        file_name = parts[7]

        tgru_m = re.match(r"tg(?P<tg>\d+)-ru(?P<ru>\d+)$", tgru)
        seed_m = re.match(r"seed(?P<seed>\d+)$", seed_part)
        if not (tgru_m and seed_m):
            return None

        # method from mode
        if mode == "base":
            method = "infinigen"
        else:
            method = "kvswap"

        # budget from file name: ..._<budget>_0-curr-emb_p<ratio>.log
        budget = None
        ratio = None
        b_m = re.search(r"_(?P<budget>\d+)_0-curr-emb_p(?P<ratio>[0-9.]+)\.log$", file_name)
        if b_m:
            budget = int(b_m.group("budget"))
            ratio = float(b_m.group("ratio"))

        return {
            "method": method,
            "disk": disk,
            "model": model,
            "params": {
                "tg": int(tgru_m.group("tg")),
                "ru": int(tgru_m.group("ru")),
                "budget": budget,
                "ratio": ratio,
                "mode": mode,
            },
            "common": parse_common_params(file_name, int(seed_m.group("seed"))),
        }

    return None


def parse_log_metrics(log_path, method):
    metrics = {
        "throughput": None,       # shadowkv throughput or kvswap/infinigen decode throughput
        "reuse_rate": None,
        "swap_bw": None,
        "flush_bw": None,
    }

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if method == "shadowkv":
        m = re.search(r"Throughput:\s*([0-9.]+)\s*tokens/s", text)
        if m:
            metrics["throughput"] = float(m.group(1))
        return metrics

    # kvswap / infinigen
    m_decode = re.search(r"Throughput Total:\s*[0-9.]+\s*Prefill:\s*[0-9.]+\s*Decode:\s*([0-9.]+)", text)
    if m_decode:
        metrics["throughput"] = float(m_decode.group(1))

    m_reuse = re.search(r"Average reuse rate:\s*([0-9.]+)", text)
    if m_reuse:
        metrics["reuse_rate"] = float(m_reuse.group(1))

    m_swap = re.search(r"sum_swap_bw:\s*([0-9.]+)\s*MB/s", text)
    if m_swap:
        metrics["swap_bw"] = float(m_swap.group(1))

    m_flush = re.search(r"sum_flush_bw:\s*([0-9.]+)\s*MB/s", text)
    if m_flush:
        metrics["flush_bw"] = float(m_flush.group(1))

    return metrics


def collect_logs(base_dir):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for root, _, files in os.walk(base_dir):
        for name in files:
            if not name.endswith(".log"):
                continue
            log_path = os.path.join(root, name)

            path_info = parse_path_info(base_dir, log_path)
            if path_info is None:
                continue

            method = path_info["method"]
            disk = path_info["disk"]
            model = path_info["model"]
            params = path_info["params"]
            common = path_info["common"]
            metrics = parse_log_metrics(log_path, method)

            record = {
                "path": log_path,
                "params": params,
                "common": common,
                "metrics": metrics,
            }
            data[method][disk][model].append(record)

    return _to_regular_dict(data)


def _mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def summarize_fig10(records_by_method_disk_model):
    target_models = {"Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Qwen3-14B"}
    target_disks = {"nvme", "emmc"}
    target_batches = {1, 8}
    target_total_len = 32768

    grouped = defaultdict(list)

    for method, disk_dict in records_by_method_disk_model.items():
        if method == "__meta__":
            continue
        if method not in {"shadowkv", "kvswap"}:
            continue
        for disk, model_dict in disk_dict.items():
            if disk not in target_disks:
                continue
            for model, records in model_dict.items():
                if model not in target_models:
                    continue
                for rec in records:
                    params = rec.get("params", {})
                    common = rec.get("common", {})
                    metrics = rec.get("metrics", {})

                    # Public constraints
                    batch = common.get("batch")
                    seqlen = common.get("seqlen")
                    genlen = common.get("genlen")
                    if batch not in target_batches:
                        continue
                    if seqlen is None or genlen is None or (seqlen + genlen) != target_total_len:
                        continue

                    # Method-specific constraints
                    if method == "shadowkv":
                        if params.get("budget") != 400 or params.get("chunk") != 16 or params.get("r") != 40:
                            continue
                    else:  # kvswap
                        if params.get("budget") != 400:
                            continue
                        ratio = params.get("ratio")
                        if ratio is None or abs(ratio - 1.0) > 1e-9:
                            continue
                        if params.get("ru") != 400:
                            continue
                        expected_tg = 4 if disk == "nvme" else 8
                        if params.get("tg") != expected_tg:
                            continue

                    throughput = metrics.get("throughput")
                    if throughput is None:
                        continue

                    key = (method, disk, model, batch)
                    grouped[key].append(
                        {
                            "seed": common.get("seed"),
                            "throughput": throughput,
                            "path": rec.get("path"),
                        }
                    )

    # vLLM results from: {search_path}/vllm_results/{model}_results.csv
    # with columns: seqlen,batch,throughput
    # This function stores vllm in grouped_vllm[(model, batch)] = throughput
    # (single measurement, no seed dimension).
    grouped_vllm = {}
    # We attach search_path through a special key on input dict when available.
    search_path = records_by_method_disk_model.get("__meta__", {}).get("search_path")
    if search_path:
        vllm_dir = os.path.join(search_path, "vllm_results")
        if os.path.isdir(vllm_dir):
            for fn in os.listdir(vllm_dir):
                if not fn.endswith("_results.csv"):
                    continue
                model = fn[: -len("_results.csv")]
                if model not in target_models:
                    continue
                csv_path = os.path.join(vllm_dir, fn)
                try:
                    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            try:
                                seqlen = int(row.get("seqlen", "").strip())
                                batch = int(row.get("batch", "").strip())
                                throughput = float(row.get("throughput", "").strip())
                            except Exception:
                                continue
                            if seqlen != target_total_len or batch not in target_batches:
                                continue
                            grouped_vllm[(model, batch)] = {
                                "throughput": throughput,
                                "path": csv_path,
                            }
                except Exception:
                    # Keep robust behavior: skip broken csv without failing whole parser.
                    continue

    # Format output tree: method -> disk -> model -> batch -> stats
    # Keep full target combinations and output N/A when no matched samples.
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    methods = ["shadowkv", "kvswap", "vllm"]
    disks = ["nvme", "emmc"]
    models = ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "Qwen3-14B"]
    batches = [1, 8]

    for method in methods:
        method_disks = disks if method != "vllm" else ["N/A"]
        for disk in method_disks:
            for model in models:
                for batch in batches:
                    if method == "vllm":
                        v = grouped_vllm.get((model, batch))
                        if not v:
                            out[method][disk][model][str(batch)] = {
                                "decode_throughput_mean": "N/A",
                                "decode_throughput_std": "N/A",
                                "num_samples": 0,
                                "seeds": "N/A",
                            }
                        else:
                            out[method][disk][model][str(batch)] = {
                                "decode_throughput_mean": v["throughput"],
                                "decode_throughput_std": "N/A",
                                "num_samples": 1,
                                "seeds": "N/A",
                            }
                        continue

                    key = (method, disk, model, batch)
                    rows = grouped.get(key, [])
                    if not rows:
                        out[method][disk][model][str(batch)] = {
                            "decode_throughput_mean": "N/A",
                            "decode_throughput_std": "N/A",
                            "num_samples": 0,
                            "seeds": "N/A",
                        }
                        continue

                    values = [r["throughput"] for r in rows]
                    seeds = sorted({r["seed"] for r in rows if r.get("seed") is not None})
                    mean, std = _mean_std(values)
                    out[method][disk][model][str(batch)] = {
                        "decode_throughput_mean": mean,
                        "decode_throughput_std": std,
                        "num_samples": len(values),
                        "seeds": seeds if seeds else "N/A",
                    }

    return _to_regular_dict(out)


def summarize_fig11_tp(records_by_method_disk_model):
    """
    Fig-11 throughput extraction for Llama-3.1-8B-Instruct, total context 24576
    (seqlen+genlen for engine logs; vLLM uses seqlen=24576 in CSV).

    Named series:
      InfiGen*        — infinigen, ratio=0.5, tg=1, budget=400, ru=0,   batch 1,3
      InfiGen*+reuse  — infinigen, ratio=0.5, tg=1, budget=400, ru=400, batch 1,3
      ShadowKV        — shadowkv, chunk=8, rank=160, budget=400, batch 2,4
      KVSwap          — kvswap, ratio=1,   tg nvme=4/emmc=8, budget=400, ru=400, batch 8
      KVSwap-t        — kvswap, ratio=0.25, tg nvme=4/emmc=8, budget=400, ru=400, batch 8
      vLLM            — batch=8, seqlen=24576 (from vllm_results CSV)
    """
    target_model = "Llama-3.1-8B-Instruct"
    target_disks = {"nvme", "emmc"}
    target_total_len = 24576

    # (display_name, disk, batch_str) -> list of {throughput, seed, path}
    grouped = defaultdict(list)

    def add_row(name, disk, batch, throughput, seed, path):
        if throughput is None:
            return
        grouped[(name, disk, str(batch))].append(
            {"throughput": throughput, "seed": seed, "path": path}
        )

    for method, disk_dict in records_by_method_disk_model.items():
        if method == "__meta__":
            continue
        if method not in {"infinigen", "shadowkv", "kvswap"}:
            continue
        for disk, model_dict in disk_dict.items():
            if disk not in target_disks:
                continue
            for model, records in model_dict.items():
                if model != target_model:
                    continue
                for rec in records:
                    params = rec.get("params", {})
                    common = rec.get("common", {})
                    metrics = rec.get("metrics", {})
                    batch = common.get("batch")
                    seqlen = common.get("seqlen")
                    genlen = common.get("genlen")
                    if seqlen is None or genlen is None:
                        continue
                    if seqlen + genlen != target_total_len:
                        continue
                    tp = metrics.get("throughput")
                    seed = common.get("seed")

                    if method == "infinigen":
                        if params.get("mode") != "base":
                            continue
                        ratio = params.get("ratio")
                        if ratio is None or abs(ratio - 0.5) > 1e-6:
                            continue
                        if params.get("tg") != 1 or params.get("budget") != 400:
                            continue
                        ru = params.get("ru")
                        if batch not in (1, 3):
                            continue
                        if ru == 0:
                            add_row("InfiGen*", disk, batch, tp, seed, rec.get("path"))
                        elif ru == 400:
                            add_row("InfiGen*+reuse", disk, batch, tp, seed, rec.get("path"))

                    elif method == "shadowkv":
                        if params.get("budget") != 400:
                            continue
                        if params.get("chunk") != 8 or params.get("r") != 160:
                            continue
                        if batch not in (2, 4):
                            continue
                        add_row("ShadowKV", disk, batch, tp, seed, rec.get("path"))

                    elif method == "kvswap":
                        if params.get("budget") != 400 or params.get("ru") != 400:
                            continue
                        if batch != 8:
                            continue
                        expected_tg = 4 if disk == "nvme" else 8
                        if params.get("tg") != expected_tg:
                            continue
                        ratio = params.get("ratio")
                        if ratio is None:
                            continue
                        if abs(ratio - 1.0) < 1e-9:
                            add_row("KVSwap", disk, batch, tp, seed, rec.get("path"))
                        elif abs(ratio - 0.25) < 1e-9:
                            add_row("KVSwap-t", disk, batch, tp, seed, rec.get("path"))

    # vLLM: {search_path}/vllm_results/Llama-3.1-8B-Instruct_results.csv
    search_path = records_by_method_disk_model.get("__meta__", {}).get("search_path")
    vllm_tp = None
    vllm_path = None
    if search_path:
        csv_path = os.path.join(
            search_path, "vllm_results", f"{target_model}_results.csv"
        )
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            seqlen = int(row.get("seqlen", "").strip())
                            batch = int(row.get("batch", "").strip())
                            throughput = float(row.get("throughput", "").strip())
                        except Exception:
                            continue
                        if seqlen == target_total_len and batch == 8:
                            vllm_tp = throughput
                            vllm_path = csv_path
                            break
            except Exception:
                pass

    # Build output: name -> disk -> batch -> {mean, std, num_samples, seeds, paths}
    display_order = [
        "InfiGen*",
        "InfiGen*+reuse",
        "ShadowKV",
        "KVSwap",
        "KVSwap-t",
        "vLLM",
    ]
    out = {}

    for name in display_order:
        if name == "vLLM":
            if vllm_tp is not None:
                out[name] = {
                    "disk": "N/A",
                    "batch": "8",
                    "decode_throughput": vllm_tp,
                    "seqlen": target_total_len,
                    "path": vllm_path,
                }
            else:
                out[name] = {
                    "disk": "N/A",
                    "batch": "8",
                    "decode_throughput": "N/A",
                    "seqlen": target_total_len,
                    "path": "N/A",
                }
            continue

        out[name] = {"nvme": {}, "emmc": {}}
        for disk in ("nvme", "emmc"):
            for batch in (
                [1, 3] if name in ("InfiGen*", "InfiGen*+reuse") else [2, 4]
                if name == "ShadowKV"
                else [8]
            ):
                key = (name, disk, str(batch))
                rows = grouped.get(key, [])
                if not rows:
                    out[name][disk][str(batch)] = {
                        "decode_throughput_mean": "N/A",
                        "decode_throughput_std": "N/A",
                        "num_samples": 0,
                        "seeds": "N/A",
                        "paths": "N/A",
                    }
                    continue
                values = [r["throughput"] for r in rows]
                seeds = sorted({r["seed"] for r in rows if r.get("seed") is not None})
                mean, std = _mean_std(values)
                out[name][disk][str(batch)] = {
                    "decode_throughput_mean": mean,
                    "decode_throughput_std": std,
                    "num_samples": len(values),
                    "seeds": seeds if seeds else "N/A",
                    "paths": [r["path"] for r in rows],
                }

    return out


def _fig11_tp_mean(result, series_name, disk, batch_str):
    """decode_throughput_mean from result[series_name][disk][batch]; vLLM uses top-level decode_throughput.
    Missing / N/A -> 0.0."""
    if series_name == "vLLM":
        v = result.get("vLLM", {}).get("decode_throughput")
        if v in (None, "N/A"):
            return 0.0
        return float(v)
    st = result.get(series_name, {}).get(disk, {}).get(batch_str, {})
    m = st.get("decode_throughput_mean")
    if m in (None, "N/A"):
        return 0.0
    return float(m)


def build_fig11_tp_throughput_data(result):
    """
    Structure matches the reference fig-11 script: two panels NVMe / eMMC,
    labels with 2000MiB / 800MiB bars (batch + thr).
    """
    panels = {}

    def panel_for_disk(disk):
        return {
            "InfiGen*": {
                "2000MiB": {
                    "batch": 3,
                    "thr": _fig11_tp_mean(result, "InfiGen*", disk, "3"),
                },
                "800MiB": {
                    "batch": 1,
                    "thr": _fig11_tp_mean(result, "InfiGen*", disk, "1"),
                },
            },
            "InfiGen*\n+reuse": {
                "2000MiB": {
                    "batch": 3,
                    "thr": _fig11_tp_mean(result, "InfiGen*+reuse", disk, "3"),
                },
                "800MiB": {
                    "batch": 1,
                    "thr": _fig11_tp_mean(result, "InfiGen*+reuse", disk, "1"),
                },
            },
            "Loki": {
                "2000MiB": {
                    "batch": 3,
                    "thr": _fig11_tp_mean(result, "Loki", disk, "3"),
                },
                "800MiB": {
                    "batch": 1,
                    "thr": _fig11_tp_mean(result, "Loki", disk, "1"),
                },
            },
            "ShadowKV": {
                "2000MiB": {
                    "batch": 4,
                    "thr": _fig11_tp_mean(result, "ShadowKV", disk, "4"),
                },
                "800MiB": {
                    "batch": 2,
                    "thr": _fig11_tp_mean(result, "ShadowKV", disk, "2"),
                },
            },
            "KVSwap": {
                "2000MiB": {
                    "batch": 8,
                    "thr": _fig11_tp_mean(result, "KVSwap", disk, "8"),
                },
                "800MiB": {
                    "batch": 8,
                    "thr": _fig11_tp_mean(result, "KVSwap-t", disk, "8"),
                },
            },
        }

    panels["(a) NVMe"] = panel_for_disk("nvme")
    panels["(b) eMMC"] = panel_for_disk("emmc")
    return panels


def save_fig11_tp_figure(result, output_file):
    """
    Bar chart layout from Untitled-1 reference; thr / vLLM from summarize_fig11_tp.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    throughput_data = build_fig11_tp_throughput_data(result)
    vllm_throughput = _fig11_tp_mean(result, "vLLM", None, None)

    labels = ["InfiGen*", "InfiGen*\n+reuse", "Loki", "ShadowKV", "KVSwap", "vLLM"]
    sizes = ["2000MiB", "800MiB"]

    colors = {"800MiB": "#345B8C", "2000MiB": "#F28E2B"}
    vllm_color = "#9FD6D0"

    fig, axes = plt.subplots(2, 1, figsize=(4.5, 2.5), sharex=True)
    bar_width = 0.4
    x = np.arange(len(labels))

    for idx, (device, ax) in enumerate(zip(throughput_data.keys(), axes)):
        data = throughput_data[device]

        # Fixed y-axis: (a) NVMe 0–40, (b) eMMC 0–20
        if idx == 0:
            ax.set_ylim(0, 40)
            ax.set_yticks([0, 10, 20, 30, 40])
            y_top_for_label = 40
        else:
            ax.set_ylim(0, 20)
            ax.set_yticks([0, 5, 10, 15, 20])
            y_top_for_label = 20

        for i, label in enumerate(labels):
            if label == "vLLM":
                ax.bar(x[i], vllm_throughput, bar_width, color=vllm_color)
                if device == "(b) eMMC":
                    ax.text(
                        x[i],
                        y_top_for_label,
                        f"TP={vllm_throughput:.3g}",
                        ha="center",
                        va="top",
                        fontsize=9.5,
                    )
            else:
                entry_2200 = data[label]["2000MiB"]
                entry_900 = data[label]["800MiB"]
                batch_2200 = entry_2200.get("batch", 0)
                batch_900 = entry_900.get("batch", 0)

                ax.bar(
                    x[i] - bar_width / 2,
                    entry_2200["thr"],
                    bar_width,
                    color=colors["2000MiB"],
                )
                ax.bar(
                    x[i] + bar_width / 2,
                    entry_900["thr"],
                    bar_width,
                    color=colors["800MiB"],
                )

                ax.text(
                    x[i] - bar_width / 2,
                    entry_2200["thr"] - 0.2,
                    f"b={batch_2200}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
                ax.text(
                    x[i] + bar_width / 2 + 0.05,
                    entry_900["thr"] - 0.2,
                    f"b={batch_900}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_ylabel("TP (tokens/s)", fontsize=10)
        ax.set_title(device, fontsize=10, pad=0)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    axes[1].set_xticks(x)
    axes[1].tick_params(axis="x", pad=0)
    axes[0].tick_params(axis="x", pad=0)
    axes[0].tick_params(axis="y", pad=0)
    axes[1].tick_params(axis="y", pad=0)
    xtick_labels = [
        label.replace("KVSwap", "KVSwap/\nKVSwap-t").replace("ShadowKV", "ShadKV")
        for label in labels
    ]
    axes[1].set_xticklabels(xtick_labels, rotation=0, ha="center", fontsize=9.5)

    legend_elements = [Patch(facecolor=colors[s], label=s) for s in sizes]
    # Do not call plt.legend() with no args (warns when no labeled artists exist).
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fontsize=9.5,
        frameon=False,
        labelspacing=0.0,
        columnspacing=2,
        handletextpad=0,
    )

    plt.tight_layout(pad=0)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _fig10_throughput_for_bar(stats):
    """Numeric bar height; N/A -> 0.0 for plotting."""
    m = stats.get("decode_throughput_mean")
    if m == "N/A" or m is None:
        return 0.0
    return float(m)


def build_fig10_plot_data(result):
    """
    Build dicts expected by plot_fig10: for each model (group), keys
    ShadowKV+eMMC, ShadowKV+NVMe, KVSwap+eMMC, KVSwap+NVMe, vLLM.
    """
    groups = [
        "Llama-3.2-3B-Instruct",
        "Llama-3.1-8B-Instruct",
        "Qwen3-14B",
    ]

    def one_batch(batch_str):
        d = {}
        for g in groups:
            d[g] = {
                "ShadowKV+eMMC": _fig10_throughput_for_bar(
                    result["shadowkv"]["emmc"][g][batch_str]
                ),
                "ShadowKV+NVMe": _fig10_throughput_for_bar(
                    result["shadowkv"]["nvme"][g][batch_str]
                ),
                "KVSwap+eMMC": _fig10_throughput_for_bar(
                    result["kvswap"]["emmc"][g][batch_str]
                ),
                "KVSwap+NVMe": _fig10_throughput_for_bar(
                    result["kvswap"]["nvme"][g][batch_str]
                ),
                "vLLM": _fig10_throughput_for_bar(result["vllm"]["N/A"][g][batch_str]),
            }
        return d

    return one_batch("1"), one_batch("8")


def save_fig10_pdf(data_b1, data_b8, output_file):
    """
    Figure layout matches the reference script; only data is injected.
    Saves to output_file (use .pdf extension).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams["hatch.linewidth"] = 0.1
    fig, axes = plt.subplots(2, 1, figsize=(4, 1.9), sharex=True)

    def plot_fig10(data, ax, i):
        method_colors = {
            "ShadowKV": "#345B8C",
            "KVSwap": "#F28E2B",
            "vLLM": "#9FD6D0",
        }

        storage_hatch = {
            "eMMC": "//",
            "NVMe": "\\\\",
        }

        groups = list(data.keys())
        # Short x-axis labels (full names remain keys in `data`)
        group_display_labels = {
            "Llama-3.2-3B-Instruct": "3B",
            "Llama-3.1-8B-Instruct": "8B",
            "Qwen3-14B": "14B",
        }
        bar_width = 0.4
        subgroup_spacing = 0.1
        vllm_spacing = 0.1
        group_spacing = 0.50
        methods = ["ShadowKV", "KVSwap"]

        x_pos = 0.0
        group_centers = []

        for group in groups:
            group_start = x_pos

            # eMMC
            for method in methods:
                if method == "ShadowKV" or method == "InfiGen*+re" or method == "ShadowKV-t":
                    val = data[group][f"{method}+eMMC"]
                    ax.bar(
                        x_pos,
                        val,
                        width=bar_width,
                        color=method_colors[method],
                        hatch=storage_hatch["eMMC"],
                        linewidth=0.1,
                    )
                elif method in ["KVSwap", "KVSwap-t"]:
                    total_val = data[group][f"{method}+eMMC"]
                    diff_val = total_val
                    ax.bar(
                        x_pos,
                        diff_val,
                        bottom=0,
                        width=bar_width,
                        color=method_colors[method],
                        hatch=storage_hatch["eMMC"],
                        linewidth=0.1,
                    )
                x_pos += bar_width
            x_pos += subgroup_spacing

            # NVMe
            for method in methods:
                if method == "ShadowKV" or method == "InfiGen*+re" or method == "ShadowKV-t":
                    val = data[group][f"{method}+NVMe"]
                    ax.bar(
                        x_pos,
                        val,
                        width=bar_width,
                        color=method_colors[method],
                        hatch=storage_hatch["NVMe"],
                        linewidth=0.1,
                    )
                elif method in ["KVSwap", "KVSwap-t"]:
                    total_val = data[group][f"{method}+NVMe"]
                    diff_val = total_val
                    ax.bar(
                        x_pos,
                        diff_val,
                        bottom=0,
                        width=bar_width,
                        color=method_colors[method],
                        hatch=storage_hatch["NVMe"],
                        linewidth=0.1,
                    )
                x_pos += bar_width
            x_pos += vllm_spacing

            # vLLM
            ax.bar(
                x_pos,
                data[group]["vLLM"],
                width=bar_width,
                color=method_colors["vLLM"],
                linewidth=0.1,
            )
            x_pos += bar_width

            group_end = x_pos - bar_width / 2.0
            group_centers.append((group_start + group_end) / 2.0)
            x_pos += group_spacing

        ax.set_xticks(group_centers)
        ax.set_xticklabels(
            [group_display_labels.get(g, g) for g in groups],
            fontsize=9,
        )
        ax.set_ylabel("TP(tokens/s)", fontsize=9, labelpad=0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", pad=0, labelsize=9)
        ax.tick_params(axis="y", pad=0, labelsize=9)

        if i > 0:
            ax.set_ylim(0, 45)
            ax.set_yticks(np.arange(0, 50, step=10))
            ax.set_yticklabels([0, 10, 20, 30, 40], fontsize=9)
            return

        ax.set_ylim(0, 14)
        ax.set_yticks(np.arange(0, 14, step=4))
        ax.set_yticklabels([0, 4, 8, 12], fontsize=9)

        model_3b = "Llama-3.2-3B-Instruct"
        vllm_tp_3b_b1 = float(data.get(model_3b, {}).get("vLLM", 0.0) or 0.0)
        tp_label = f"{vllm_tp_3b_b1:.1f}" if vllm_tp_3b_b1 > 0 else "N/A"
        ax.text(
            1.8,
            12.2,
            tp_label,
            ha="center",
            va="center",
            fontsize=9,
            rotation=0,
            color="k",
        )

        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch=storage_hatch["eMMC"],
                edgecolor="k",
                linewidth=0.1,
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                hatch=storage_hatch["NVMe"],
                edgecolor="k",
                linewidth=0.1,
            ),
            plt.Rectangle((0, 0), 1, 1, facecolor=method_colors["ShadowKV"]),
            plt.Rectangle((0, 0), 1, 1, facecolor=method_colors["KVSwap"]),
            plt.Rectangle((0, 0), 1, 1, facecolor=method_colors["vLLM"]),
        ]
        labels = [
            "eMMC",
            "NVMe",
            "ShadKV",
            "KVSwap",
            "vLLM",
        ]

        ax.legend(
            handles,
            labels,
            loc="lower center",
            fontsize=9,
            frameon=False,
            ncol=5,
            bbox_to_anchor=(0.47, 1.04),
            handletextpad=0,
            columnspacing=0.3,
            labelspacing=0,
        )

    plot_fig10(data_b1, axes[0], 0)
    plot_fig10(data_b8, axes[1], 1)

    axes[0].set_title("(a) Batch=1", fontsize=9.5, pad=0)
    axes[1].set_title("(b) Batch=8", fontsize=9.5, pad=0)

    plt.tight_layout(pad=0, w_pad=0.0, h_pad=0.0)
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("search_path")
    parser.add_argument("mode")
    parser.add_argument("output_file")
    args = parser.parse_args()

    all_logs = collect_logs(args.search_path)
    all_logs["__meta__"] = {"search_path": args.search_path}

    if args.mode == "fig10":
        result = summarize_fig10(all_logs)
    elif args.mode in ("fig-11-tp", "fig11_tp"):
        result = summarize_fig11_tp(all_logs)
    else:
        raise ValueError(
            f"Unsupported mode: {args.mode}. Supported: fig10, fig-11-tp"
        )

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)

    if args.output_file != "-":
        out_path = os.path.abspath(args.output_file)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        is_plot = args.output_file.lower().endswith((".pdf", ".png"))
        if args.mode == "fig10" and is_plot:
            data_b1, data_b8 = build_fig10_plot_data(result)
            save_fig10_pdf(data_b1, data_b8, out_path)
        elif args.mode in ("fig-11-tp", "fig11_tp") and is_plot:
            save_fig11_tp_figure(result, out_path)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text + "\n")


if __name__ == "__main__":
    main()
