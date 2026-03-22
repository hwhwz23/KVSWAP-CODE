import json
import sys
from pathlib import Path
import os

def fmt(x: float) -> str:
    """Format absolute values with two decimals."""
    return f"{x:.2f}"

def fmt_delta(x: float) -> str:
    """Format deltas with sign and two decimals; omit '+' for 0.00."""
    if abs(x) < 5e-3:  # treat very small values as 0.00
        return "0.00"
    return f"{x:+.2f}"

def print_longbench_scores(json_path: str) -> None:
    """
    Extract and print scores for SQA, MQA, SUM, FSL, SYN, COD and AVG from a
    LongBench-style result JSON file. Also prints AVG.num.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = ["SQA", "MQA", "SUM", "FSL", "SYN", "CODE"]

    print(f"File: {path}")

    # Per-task scores (table format with AVG as last column), absolute values.
    avg = data.get("AVG", {})
    avg_score = avg.get("score", "N/A")
    avg_num = avg.get("num", "N/A")

    headers = tasks + ["AVG", "NUM"]
    col_width = 10
    header_line = "".join(f"{h:>{col_width}}" for h in headers)
    print(header_line)


    values = []
    for task in tasks:
        value = data.get(task, None)
        if isinstance(value, (int, float)):
            values.append(fmt(value))
        else:
            values.append("N/A")
    values.append(fmt(avg_score) if isinstance(avg_score, (int, float)) else "N/A")
    values.append(str(avg_num))
    value_line = "".join(f"{v:>{col_width}}" for v in values)
    print(value_line)

def print_ruler_scores(json_path: str) -> None:
    """
    Extract and print scores for ruler tasks from a RULER-style result JSON file.
    Expected keys include:
      - niah_single_1, niah_single_2, niah_multikey_1,
        niah_multiquery, niah_multivalue, qa_1, qa_2, vt
      - avg (overall score) and avg_num (number of instances)
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    task_keys = [
        "niah_single_1",
        "niah_single_2",
        "niah_multikey_1",
        "niah_multiquery",
        "niah_multivalue",
        "qa_1",
        "qa_2",
        "vt",
    ]
    task_labels = ["S1", "S2", "MK1", "MQ", "MV", "QA1", "QA2", "VT"]

    print(f"File: {path}")
    # Table with AVG and avg_num as last columns, absolute values.
    avg_score = data.get("avg", "N/A")
    avg_num = data.get("avg_num", "N/A")

    headers = task_labels + ["AVG", "NUM"]
    col_width = 10
    header_line = "".join(f"{h:>{col_width}}" for h in headers)
    print(header_line)

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    values = []
    for key in task_keys:
        value = data.get(key, None)
        if isinstance(value, (int, float)):
            values.append(fmt(value))
        else:
            values.append("N/A")
    values.append(fmt(avg_score) if isinstance(avg_score, (int, float)) else "N/A")
    values.append(str(avg_num))

    value_line = "".join(f"{v:>{col_width}}" for v in values)
    print(value_line)


def replace_method_name(method: str, file: str) -> str:
    if method == 'none':
        method_in_paper = 'Full-KV'
    elif method == 'infinigen':
        method_in_paper = 'Infinigen'
    elif method == 'infinigen_mergeh':
        method_in_paper = 'Infinigen*'
    elif method == 'loki':
        method_in_paper = 'Loki'
        if '_p0.125_' in file:
            pass
        elif '_p0.03125_' in file:
            method_in_paper += '-t'
        else:
            raise ValueError(f"Unknown method in Loki: {method}")
    elif method.startswith('shadowkv'):
        method_in_paper = 'ShadowKV'
        if method == 'shadowkv-16-40-48-4':
            pass
        elif method == 'shadowkv-60-16':
            method_in_paper += '-t'
        else:
            raise ValueError(f"Unknown method in ShadowKV: {method}")
    elif method == 'lr_proj_mh':
        method_in_paper = 'KVSwap'
        if '_p1_0-' in file:
            pass
        elif '_p0.25_' in file:
            method_in_paper += '-t'
        else:
            raise ValueError(f"Unknown method in KVSwap: {method}")
        if '_tg4_' in file or '_tg4.json' in file or '_tg4cot.json' in file:
            method_in_paper += '-NVMe'
        elif '_tg8_' in file or '_tg8.json' in file or '_tg8cot.json' in file:
            method_in_paper += '-eMMC'
        else:
            raise ValueError(f"Unknown method in KVSwap: {method}")
    else:
        raise ValueError(f"Unknown method: {method}")
    return method_in_paper


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <path-to-results> <mode>")
        sys.exit(1)

    mode = sys.argv[2]

    if mode == 'table2':
        print_ruler = True
        print_longbench = True
        method_list = ['Full-KV', 'Infinigen', 'Infinigen*', 'Loki', 'ShadowKV',
            'KVSwap-NVMe', 'KVSwap-eMMC', 'Loki-t', 'ShadowKV-t',
            'KVSwap-t-NVMe', 'KVSwap-t-eMMC']
    elif mode == 'table3-left':
        method_list = ['Full-KV', 'Loki', 'ShadowKV',
            'KVSwap-NVMe', 'KVSwap-eMMC', 'Loki-t', 'ShadowKV-t',
            'KVSwap-t-NVMe', 'KVSwap-t-eMMC']
        print_ruler = False
        print_longbench = False
        print_longbench_mqa = True
        model_list = ['Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']
    elif mode == 'table3-right':
        method_list = [
            "Full-KV",
            "Loki",
            "ShadowKV",
            "KVSwap-NVMe",
            "KVSwap-eMMC",
            "Loki-t",
            "ShadowKV-t",
            "KVSwap-t-NVMe",
            "KVSwap-t-eMMC",
        ]
        print_ruler = False
        print_longbench = False
        print_longbench_mqa = False
        model_list = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "InternVL3-14B"]
    elif mode == 'fig-11-acc':
        print_ruler = False
        print_longbench = False
        print_longbench_mqa = False
        method_list = []
    else:
        raise ValueError(f"Unknown mode: {mode}")

    def fmt(x: float) -> str:
        return f"{x:.2f}"

    def _format_path(template: str, **kwargs) -> str:
        """
        Best-effort formatter for user-provided path templates.
        Supports templates that may or may not include {task} / {model_name}.
        """
        try:
            return template.format(**kwargs)
        except KeyError:
            # fall back: strip unknown keys by trying common minimal variants
            for keys in [
                ("task", "model_name"),
                ("model_name",),
                ("task",),
                tuple(),
            ]:
                try:
                    return template.format(**{k: kwargs[k] for k in keys if k in kwargs})
                except KeyError:
                    continue
            raise

    def _load_longbench_method_paths(base_dir: str) -> dict:
        """
        Scan a LongBench result directory and return:
          { method_in_paper -> path_to_tasks_results_json }
        """
        out = {}
        if not os.path.isdir(base_dir):
            return out
        for method in os.listdir(base_dir):
            method_dir = os.path.join(base_dir, method)
            if not os.path.isdir(method_dir):
                continue
            for file in os.listdir(method_dir):
                if file.endswith(".json") and file.startswith("tasks_results_"):
                    method_in_paper = replace_method_name(method, file)
                    out[method_in_paper] = os.path.join(method_dir, file)
        return out

    def _read_longbench_task_value(json_path: str, task_key: str) -> float | None:
        path_obj = Path(json_path)
        with path_obj.open("r", encoding="utf-8") as f:
            d = json.load(f)
        v = d.get(task_key, None)
        return v if isinstance(v, (int, float)) else None

    def _read_longbench_avg_num(json_path: str) -> float | None:
        path_obj = Path(json_path)
        with path_obj.open("r", encoding="utf-8") as f:
            d = json.load(f)
        avg_block = d.get("AVG", {})
        n = avg_block.get("num", None)
        return n if isinstance(n, (int, float)) else None

    def _find_tasks_results_json(method_dir: str) -> str | None:
        if not os.path.isdir(method_dir):
            return None
        for fn in os.listdir(method_dir):
            if fn.startswith("tasks_results_") and fn.endswith(".json"):
                return os.path.join(method_dir, fn)
        return None

    def _read_avg_score_and_num(json_path: str) -> tuple[float | None, int | None]:
        path_obj = Path(json_path)
        with path_obj.open("r", encoding="utf-8") as f:
            d = json.load(f)
        avg_block = d.get("AVG", {})
        score = avg_block.get("score", None)
        num = avg_block.get("num", None)
        score_out = score if isinstance(score, (int, float)) else None
        num_out = int(num) if isinstance(num, (int, float)) else None
        return score_out, num_out

    def _parse_tab2_longbench_raw(tab2_path: str) -> dict[str, dict[str, float | int]]:
        """
        Parse `RESULTS/tab-2.txt` "LongBench Scores (raw per method)" section.
        Returns: { method -> {col_name -> value} } where values are floats except NUM (int).
        """
        text = Path(tab2_path).read_text(encoding="utf-8", errors="replace").splitlines()
        out: dict[str, dict[str, float | int]] = {}

        # Find the section start.
        start = None
        for i, line in enumerate(text):
            if line.strip() == "LongBench Scores (raw per method):":
                start = i + 1
                break
        if start is None:
            return out

        i = start
        while i < len(text):
            line = text[i].strip()
            if line.startswith("====="):
                break
            if line.startswith("Method: "):
                method = line[len("Method: "):].strip()
                # Seek header line (contains MQA SUM ... NUM)
                j = i + 1
                header_tokens = None
                value_tokens = None
                while j < len(text):
                    s = text[j].strip()
                    if s.startswith("Method: ") or s.startswith("====="):
                        break
                    # header line: starts with task names (SQA MQA SUM ...)
                    if header_tokens is None and ("MQA" in s and "SUM" in s and "NUM" in s):
                        header_tokens = s.split()
                        # next non-empty line should be values
                        k = j + 1
                        while k < len(text) and text[k].strip() == "":
                            k += 1
                        if k < len(text):
                            value_tokens = text[k].strip().split()
                        break
                    j += 1

                if header_tokens and value_tokens and len(header_tokens) == len(value_tokens):
                    row: dict[str, float | int] = {}
                    for h, v in zip(header_tokens, value_tokens):
                        if h.upper() == "NUM":
                            try:
                                row["NUM"] = int(float(v))
                            except Exception:
                                pass
                        else:
                            try:
                                row[h.upper()] = float(v)
                            except Exception:
                                pass
                    out[method] = row
                i = j
                continue
            i += 1
        return out

    def _find_latest_file(dir_path: str, suffix: str, must_contain: str | None = None) -> str | None:
        """Find the latest-modified file in dir_path that ends with suffix."""
        if not os.path.isdir(dir_path):
            return None
        candidates: list[str] = []
        for fn in os.listdir(dir_path):
            if not fn.endswith(suffix):
                continue
            if must_contain is not None and must_contain not in fn:
                continue
            candidates.append(os.path.join(dir_path, fn))
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p))
        return candidates[-1]

    def _parse_mlvu_eval_log(log_path: str) -> tuple[float | None, int | None]:
        """
        Parse an MLVU eval log and extract:
          - Total Score: <float>
          - Total number of files: <int>
        """
        if not log_path:
            return None, None
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        score = None
        num = None
        for line in lines:
            s = line.strip()
            if s.startswith("Total Score:"):
                try:
                    score = float(s.split("Total Score:", 1)[1].strip())
                except Exception:
                    pass
            elif s.startswith("Total number of files:"):
                try:
                    num = int(s.split("Total number of files:", 1)[1].strip())
                except Exception:
                    pass
        return score, num

    def _read_mlvu_method_scores(method_dir: str, tg_filter: str | None = None) -> dict[str, float | int | None]:
        """
        Read both *_subplot_all_eval.log and *_summary_all_eval.log under method_dir.
        If tg_filter is provided (e.g., 'tg4' or 'tg8'), only consider filenames containing it.
        Returns dict with keys: subplot, summary, avg, num where
        avg = (subplot + summary) / 2.
        """
        subplot_log = _find_latest_file(method_dir, "_subplot_all_eval.log", must_contain=tg_filter)
        summary_log = _find_latest_file(method_dir, "_summary_all_eval.log", must_contain=tg_filter)

        subplot_score, subplot_num = _parse_mlvu_eval_log(subplot_log) if subplot_log else (None, None)
        summary_score, summary_num = _parse_mlvu_eval_log(summary_log) if summary_log else (None, None)

        avg = None
        if isinstance(subplot_score, (int, float)) and isinstance(summary_score, (int, float)):
            avg = 0.5 * (float(subplot_score) + float(summary_score))

        num = summary_num if isinstance(summary_num, int) else subplot_num
        return {
            "subplot": subplot_score,
            "summary": summary_score,
            "avg": avg,
            "num": num,
        }

    if mode == "fig-11-acc":
        # Main table:
        # Accuracy := average of LongBench MQA and SUM (or precomputed AVG in longbench_mqa+sum).
        # All values are absolute (no deltas).
        base_dir = sys.argv[1]
        tab2_path = (Path(__file__).resolve().parent.parent / "RESULTS" / "tab-2.txt").as_posix()
        raw = _parse_tab2_longbench_raw(tab2_path)

        def acc_from_tab2(method: str) -> tuple[float | None, int | None]:
            row = raw.get(method)
            if not row:
                return None, None
            mqa = row.get("MQA")
            summ = row.get("SUM")
            num = row.get("NUM")
            if isinstance(mqa, (int, float)) and isinstance(summ, (int, float)):
                return (float(mqa) + float(summ)) / 2.0, int(num) if isinstance(num, int) else None
            return None, int(num) if isinstance(num, int) else None

        def acc_from_dir(method_subdir: str) -> tuple[float | None, int | None]:
            p = _find_tasks_results_json(os.path.join(base_dir, method_subdir))
            if not p:
                return None, None
            score, num = _read_avg_score_and_num(p)
            return score, num

        rows: list[tuple[str, float | None, int | None]] = []

        # From longbench_mqa+sum folders (use AVG.score and AVG.num directly)
        score, num = acc_from_dir("infinigen_mergeh")
        rows.append(("InfiniGen*(+reuse)", score, num))
        score, num = acc_from_dir("loki")
        rows.append(("Loki", score, num))
        score, num = acc_from_dir("shadowkv-8-160-48-4")
        rows.append(("ShadowKV", score, num))

        # From tab-2.txt raw per-method LongBench results (MQA and SUM columns)
        for name, method_key in [
            ("KVSwap(NVMe)", "KVSwap-NVMe"),
            ("KVSwap-t(NVMe)", "KVSwap-t-NVMe"),
            ("KVSwap(eMMC)", "KVSwap-eMMC"),
            ("KVSwap-t(eMMC)", "KVSwap-t-eMMC"),
            ("vLLM", "Full-KV"),
        ]:
            score, num = acc_from_tab2(method_key)
            rows.append((name, score, num))

        # Print main table (absolute values, no NUM)
        print("================================================")
        print("Method".ljust(18) + "Accuracy(MQA+SUM)".rjust(18))
        print("-" * 40)
        for method, score, num in rows:
            if method in {"KVSwap(eMMC)", "vLLM"}:
                print("-" * 40)
            s = fmt(score) if isinstance(score, (int, float)) else "N/A"
            print(method.ljust(18) + s.rjust(12))
        print("-" * 40)

        # Between main table and detail results
        for _ in range(20):
            print("----"*80)
        print("Below are detail results:")

        # Detail table (includes NUM)
        print("Method".ljust(18) + "Accuracy(MQA+SUM)".rjust(18) + "  " + "NUM".rjust(6))
        print("-" * 40)
        for method, score, num in rows:
            s = fmt(score) if isinstance(score, (int, float)) else "N/A"
            n = str(num) if isinstance(num, int) else "N/A"
            print(method.ljust(18) + s.rjust(12) + "  " + n.rjust(6))
        print("-" * 40)

        sys.exit(0)

    if mode == "table3-right":
        # Table 3 (right) - MLVU (subplot + summary).
        # Full-KV prints absolute values; other methods print delta to Full-KV.
        ordered_models = ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "InternVL3-14B"]
        model_to_col = {
            "Qwen2.5-VL-3B-Instruct": "QVL3B",
            "Qwen2.5-VL-7B-Instruct": "QVL7B",
            "InternVL3-14B": "IVL14B",
        }

        # Collect per-model scores.
        # For ShadowKV and KVSwap, directories / filters depend on the model.
        per_model: dict[str, dict[str, dict[str, float | int | None]]] = {m: {} for m in ordered_models}
        for model in ordered_models:
            base_dir = _format_path(sys.argv[1], model_name=model)

            # Model-specific mapping for ShadowKV(-t) and KVSwap variants.
            if model == "Qwen2.5-VL-3B-Instruct":
                shadow_dir = "shadowkv-32-24"
                shadow_t_dir = "shadowkv-80-6"
                kvswap_nvme_tg = "p0.25_"
                kvswap_t_nvme_tg = "p0.0625_"
            elif model == "Qwen2.5-VL-7B-Instruct":
                shadow_dir = "shadowkv-32-48"
                shadow_t_dir = "shadowkv-80-10"
                kvswap_nvme_tg = "p0.5_"
                kvswap_t_nvme_tg = "p0.125_"
            else:  # InternVL3-14B: keep previous defaults
                shadow_dir = "shadowkv-8-160-48-4"
                shadow_t_dir = "shadowkv-60-16"
                kvswap_nvme_tg = "tg4"
                kvswap_t_nvme_tg = "tg4"

            # Full-KV and Loki do not depend on model-specific subnames.
            per_model[model]["Full-KV"] = _read_mlvu_method_scores(os.path.join(base_dir, "none"))
            per_model[model]["Loki"] = _read_mlvu_method_scores(os.path.join(base_dir, "loki"))
            per_model[model]["ShadowKV"] = _read_mlvu_method_scores(os.path.join(base_dir, shadow_dir))

            # KVSwap NVMe / eMMC (and their -t variants) all live in lr_proj_mh, but
            # we disambiguate via filename filters.
            lr_dir = os.path.join(base_dir, "lr_proj_mh")
            per_model[model]["KVSwap(NVMe)"] = _read_mlvu_method_scores(lr_dir, tg_filter=kvswap_nvme_tg)
            per_model[model]["KVSwap(eMMC)"] = _read_mlvu_method_scores(lr_dir, tg_filter="tg8")

            per_model[model]["Loki-t"] = _read_mlvu_method_scores(os.path.join(base_dir, "loki"))
            per_model[model]["ShadowKV-t"] = _read_mlvu_method_scores(os.path.join(base_dir, shadow_t_dir))
            per_model[model]["KVSwap(NVMe)-t"] = _read_mlvu_method_scores(lr_dir, tg_filter=kvswap_t_nvme_tg)
            per_model[model]["KVSwap(eMMC)-t"] = _read_mlvu_method_scores(lr_dir, tg_filter="tg8")

        baseline_avg = {m: per_model[m].get("Full-KV", {}).get("avg") for m in ordered_models}

        print("================================================")
        print("Table 3 (right) - MLVU (average of subplot and summary). Full-KV absolute; others are deltas to Full-KV:")
        col_width_method = 18
        col_width = 10
        header = f"{'Methods':<{col_width_method}}" + "".join(
            f"{model_to_col[m]:>{col_width}}" for m in ordered_models
        )
        print(header)
        print("-" * (col_width_method + col_width * len(ordered_models)))

        def main_cell(disp: str, model: str) -> str:
            v = per_model[model].get(disp, {}).get("avg")
            if not isinstance(v, (int, float)):
                return "N/A"
            if disp == "Full-KV":
                return fmt(v)
            b = baseline_avg.get(model)
            if isinstance(b, (int, float)):
                return fmt_delta(v - b)
            return "N/A"

        def sep() -> None:
            print("-" * (col_width_method + col_width * len(ordered_models)))

        order_for_table = [
            "Full-KV",
            "Loki",
            "ShadowKV",
            "KVSwap(NVMe)",
            "KVSwap(eMMC)",
            "Loki-t",
            "ShadowKV-t",
            "KVSwap(NVMe)-t",
            "KVSwap(eMMC)-t",
        ]

        for disp in order_for_table:
            row = f"{disp:<{col_width_method}}" + "".join(
                f"{main_cell(disp, m):>{col_width}}" for m in ordered_models
            )
            print(row)
            if disp in ["Full-KV", "KVSwap(eMMC)", "KVSwap(eMMC)-t"]:
                sep()

        print("================================================")
        for _ in range(20):
            print("----"*80)
        print("Below are detail results:")
        print("Per-model raw MLVU scores (absolute; avg = (subplot + summary) / 2):")
        for model in ordered_models:
            print("-" * (col_width_method + col_width * len(ordered_models)))
            print(f"Model: {model}")
            print(
                f"{'Methods':<{col_width_method}}"
                f"{'subplot':>{col_width}}"
                f"{'summary':>{col_width}}"
                f"{'avg':>{col_width}}"
                f"{'NUM':>{col_width}}"
            )
            for disp in order_for_table:
                info = per_model[model].get(disp, {})
                subplot = info.get("subplot")
                summary = info.get("summary")
                avg = info.get("avg")
                num = info.get("num")
                subplot_s = fmt(subplot) if isinstance(subplot, (int, float)) else "N/A"
                summary_s = fmt(summary) if isinstance(summary, (int, float)) else "N/A"
                avg_s = fmt(avg) if isinstance(avg, (int, float)) else "N/A"
                num_s = str(num) if isinstance(num, int) else "N/A"
                print(f"{disp:<{col_width_method}}{subplot_s:>{col_width}}{summary_s:>{col_width}}{avg_s:>{col_width}}{num_s:>{col_width}}")

        print("================================================")
        sys.exit(0)

    if print_ruler:
        data = {}
        task = 'ruler'
        path = sys.argv[1].format(task=task)
        for method in os.listdir(path):
            for file in os.listdir(os.path.join(path, method)):
                if file.endswith(".json") and file.startswith("results_"):
                    method_in_paper = replace_method_name(method, file)
                    data[method_in_paper] = os.path.join(path, method, file)

        # -------- RULER TABLE (with deltas) --------
        print("================================================")
        print("Ruler Scores (Full-KV absolute, others are deltas to Full-KV):")

        # collect scores
        ruler_scores = {}
        for method in method_list:
            if method in data:
                path_method = data[method]
                # reuse parser logic from print_ruler_scores
                path_obj = Path(path_method)
                with path_obj.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                task_keys = [
                    "niah_single_1",
                    "niah_single_2",
                    "niah_multikey_1",
                    "niah_multiquery",
                    "niah_multivalue",
                    "qa_1",
                    "qa_2",
                    "vt",
                ]
                vals = []
                for k in task_keys:
                    v = d.get(k, None)
                    vals.append(v if isinstance(v, (int, float)) else None)
                avg_score = d.get("avg", None)
                avg_num = d.get("avg_num", None)
                ruler_scores[method] = {
                    "values": vals,
                    "avg": avg_score,
                    "num": avg_num,
                    "path": path_method,
                }

        labels = ["Method", "S1", "S2", "MK1", "MQ", "MV", "QA1", "QA2", "VT", "AVG"]
        col_width = 12
        header_line = "".join(f"{h:>{col_width}}" for h in labels)
        print(header_line)
        print("-" * (col_width * len(labels)))

        baseline = ruler_scores.get("Full-KV")
        for method in method_list:
            if method not in ruler_scores:
                continue
            scores = ruler_scores[method]
            row = [method]
            if method == "Full-KV" or baseline is None:
                # absolute
                for v in scores["values"]:
                    row.append(fmt(v) if isinstance(v, (int, float)) else "N/A")
                row.append(fmt(scores["avg"]) if isinstance(scores["avg"], (int, float)) else "N/A")
            else:
                # deltas
                for v, b in zip(scores["values"], baseline["values"]):
                    if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                        row.append(fmt_delta(v - b))
                    else:
                        row.append("N/A")
                if isinstance(scores["avg"], (int, float)) and isinstance(baseline["avg"], (int, float)):
                    row.append(fmt_delta(scores["avg"] - baseline["avg"]))
                else:
                    row.append("N/A")
            # no NUM in the main table (NUM is available in detail outputs)

            line = "".join(f"{c:>{col_width}}" for c in row)
            print(line)

            # group separators
            if method in ["Full-KV", "KVSwap-eMMC", "KVSwap-t-eMMC"]:
                print("-" * (col_width * len(labels)))

        print("================================================")

    # -------- LONGBENCH TABLE (with deltas) --------
    if print_longbench or print_longbench_mqa:
        # Special pretty table for Table 3 (left): cross-model, MQA only.
        if mode == "table3-left":
            task = "longbench"
            task_key = "MQA"

            # Collect per-model method -> json path.
            per_model_paths = {}
            for model in model_list:
                base_dir = _format_path(sys.argv[1], model_name=model, task=task)
                per_model_paths[model] = _load_longbench_method_paths(base_dir)

            # Collect per-model values and nums.
            per_model_values: dict[str, dict[str, float | None]] = {m: {} for m in model_list}
            per_model_nums: dict[str, dict[str, float | None]] = {m: {} for m in model_list}
            for model in model_list:
                for method in method_list:
                    p = per_model_paths[model].get(method)
                    if p:
                        per_model_values[model][method] = _read_longbench_task_value(p, task_key)
                        per_model_nums[model][method] = _read_longbench_avg_num(p)
                    else:
                        per_model_values[model][method] = None
                        per_model_nums[model][method] = None

            # Print combined table: Full-KV absolute, others are (value - full_kv).
            print("================================================")
            print("Table 3 (left) - LongBench MQA (Full-KV absolute, others are deltas to Full-KV):")

            model_headers = ["Q4B", "Q8B", "Q14B"]
            model_to_col = {
                "Qwen3-4B": "Q4B",
                "Qwen3-8B": "Q8B",
                "Qwen3-14B": "Q14B",
            }
            ordered_models = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B"]

            col_width_method = 14
            col_width = 8
            header = f"{'Methods':<{col_width_method}}" + "".join(
                f"{model_to_col[m]:>{col_width}}" for m in ordered_models
            )
            print(header)
            print("-" * (col_width_method + col_width * len(ordered_models)))

            def cell(method: str, model: str) -> str:
                v = per_model_values[model].get(method)
                if v is None:
                    return "N/A"
                if method == "Full-KV":
                    return fmt(v)
                b = per_model_values[model].get("Full-KV")
                if isinstance(b, (int, float)):
                    return fmt_delta(v - b)
                return "N/A"

            # First print the Full-KV row with absolute values.
            full_kv_row = f"{'Full-KV':<{col_width_method}}" + "".join(
                f"{(fmt(per_model_values[model].get('Full-KV')) if isinstance(per_model_values[model].get('Full-KV'), (int, float)) else 'N/A'):>{col_width}}"
                for model in ordered_models
            )
            print(full_kv_row)
            print("-" * (col_width_method + col_width * len(ordered_models)))

            # Then print other methods as deltas to Full-KV.
            for method in [m for m in method_list if m != "Full-KV"]:
                row = f"{method:<{col_width_method}}" + "".join(
                    f"{cell(method, m):>{col_width}}" for m in ordered_models
                )
                print(row)

                if method in ["KVSwap-eMMC", "KVSwap-t-eMMC"]:
                    print("-" * (col_width_method + col_width * len(ordered_models)))

            # After full table, print raw values (and NUM) per model.
            print()
            for _ in range(20):
                print("----"*80)
            print("Below are detail results:")
            print("Per-model raw MQA values:")
            for model in ordered_models:
                print("-" * (col_width_method + col_width * len(ordered_models)))
                print(f"Model: {model}")
                print(f"{'Methods':<{col_width_method}}{task_key:>{col_width}}{'NUM':>{col_width}}")
                for method in method_list:
                    v = per_model_values[model].get(method)
                    num = per_model_nums[model].get(method)
                    val_str = fmt(v) if isinstance(v, (int, float)) else "N/A"
                    num_str = f"{int(num)}" if isinstance(num, (int, float)) else "N/A"
                    print(f"{method:<{col_width_method}}{val_str:>{col_width}}{num_str:>{col_width}}")

            print("================================================")
            sys.exit(0)

        if print_longbench:
            tasks = ['SQA', 'MQA', 'SUM', 'FSL', 'SYN', 'CODE']
        elif print_longbench_mqa:
            tasks = ['MQA']
        else:
            raise ValueError(f"Unknown mode: {mode}")

        task = 'longbench'
        data = {}
        path = sys.argv[1].format(task=task)
        for method in os.listdir(path):
            for file in os.listdir(os.path.join(path, method)):
                if file.endswith(".json") and file.startswith("tasks_results_"):
                    method_in_paper = replace_method_name(method, file)
                    data[method_in_paper] = os.path.join(path, method, file)

        print("================================================")
        print("LongBench Scores (Full-KV absolute, others are deltas to Full-KV):")

        longbench_scores = {}
        for method in method_list:
            if method in data:
                path_method = data[method]
                path_obj = Path(path_method)
                with path_obj.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                vals = []
                for k in tasks:
                    v = d.get(k, None)
                    vals.append(v if isinstance(v, (int, float)) else None)
                avg_block = d.get("AVG", {})
                avg_score = avg_block.get("score", None)
                avg_num = avg_block.get("num", None)
                longbench_scores[method] = {
                    "values": vals,
                    "avg": avg_score,
                    "num": avg_num,
                    "path": path_method,
                }

        labels = ["Method"] + tasks + ["AVG"]
        header_line = "".join(f"{h:>{col_width}}" for h in labels)
        print(header_line)
        print("-" * (col_width * len(labels)))

        baseline = longbench_scores.get("Full-KV")
        for method in method_list:
            if method not in longbench_scores:
                continue
            scores = longbench_scores[method]
            row = [method]
            if method == "Full-KV" or baseline is None:
                for v in scores["values"]:
                    row.append(fmt(v) if isinstance(v, (int, float)) else "N/A")
                row.append(fmt(scores["avg"]) if isinstance(scores["avg"], (int, float)) else "N/A")
            else:
                for v, b in zip(scores["values"], baseline["values"]):
                    if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                        row.append(fmt_delta(v - b))
                    else:
                        row.append("N/A")
                if isinstance(scores["avg"], (int, float)) and isinstance(baseline["avg"], (int, float)):
                    row.append(fmt_delta(scores["avg"] - baseline["avg"]))
                else:
                    row.append("N/A")
            # no NUM in the main table (NUM is available in detail outputs)

            line = "".join(f"{c:>{col_width}}" for c in row)
            print(line)

            if method in ["Full-KV", "KVSwap-eMMC", "KVSwap-t-eMMC"]:
                print("-" * (col_width * len(labels)))

        print("================================================")

        # Between main tables and detail subtables
        for _ in range(20):
            print("----"*80)
        print("Below are detail results:")

        if print_ruler:
            # -------- RAW PER-METHOD OUTPUTS (absolute) --------
            print("Ruler Scores (raw per method):")
            for method in method_list:
                if method in ruler_scores:
                    print(f"Method: {method}")
                    print_ruler_scores(ruler_scores[method]["path"])
                    print()
            print("================================================")

        if print_longbench or print_longbench_mqa:
            print("LongBench Scores (raw per method):")
            for method in method_list:
                if method in longbench_scores:
                    print(f"Method: {method}")
                    print_longbench_scores(longbench_scores[method]["path"])
                    print()

            print("================================================")