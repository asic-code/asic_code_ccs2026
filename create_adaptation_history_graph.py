import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator


def resolve_existing_csv(path_str: str) -> Path:
    """
    Windows에서 전체 경로가 MAX_PATH(260자) 근처를 넘으면 Path.is_file()이 False가 되는 경우가 있다.
    PowerShell Get-ChildItem은 찾아도 Python이 못 찾는 경우 → \\\\?\\ 확장 경로로 재시도.
    """
    p = Path(path_str)
    if p.is_file():
        return p
    if os.name == "nt":
        abs_path = os.path.abspath(path_str)
        if not abs_path.startswith("\\\\?\\"):
            extended = "\\\\?\\" + abs_path
            if os.path.isfile(extended):
                return Path(extended)
    return p


# 폰트 설정 (Times New Roman)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  # 기본 폰트 크기

# Argument parser 설정
parser = argparse.ArgumentParser(description='Generate performance history graph from CSV file')
DEFAULT_CSV_PATH = r"C:\ASIC_excute\Dataset\Dataset_ISV_turnmap\CICIDS2017\rarm_s0.01_c0.9_cstem_l10_ns0.06_n200_dom0.99_dom(da=0.9)_pul0.7_sepF(turn)_turneval_trtsNA-teNA\history.csv"
parser.add_argument(
    'csv_path',
    type=str,
    nargs='?',
    default=DEFAULT_CSV_PATH,
    help='Path to the history CSV file (default: %(default)s)',
)
parser.add_argument('--start_turn', type=int, default=None, 
                    help='Starting turn number to plot (default: first turn in CSV). Ignored if --ranges is used.')
parser.add_argument('--end_turn', type=int, default=None,
                    help='Ending turn number to plot (default: last turn in CSV). Ignored if --ranges is used.')
parser.add_argument('--ranges', type=str, nargs='+', default=None,
                    help='Multiple turn ranges in format "start-end". Example: --ranges "10-20" "40-50" "90-100"')
parser.add_argument('--fn_range', type=str, default=None,
                    help='FN-prone turn range in format "start-end". Example: --fn_range "13-15"')
parser.add_argument('--fp_range', type=str, default=None,
                    help='FP-prone turn range in format "start-end". Example: --fp_range "21-24"')
parser.add_argument('--no_attack_range', type=str, default=None,
                    help='No-attack turn range in format "start-end". Example: --no_attack_range "18-20"')
parser.add_argument(
    '--mode',
    type=str,
    choices=['full', 'signature_delta'],
    default='full',
    help='Plot mode for segment graphs: full (metrics+counts) or signature_delta (Δ signature only).',
)
parser.add_argument('--output_dir', type=str, default='adaptation',
                    help='Output directory for the graph (default: adaptation)')
parser.add_argument('--gap_width', type=float, default=3.0,
                    help='Width of gap between turn ranges when using --ranges (default: 3.0)')

args = parser.parse_args()

def parse_turn_range(range_str: str) -> tuple[int, int]:
    if range_str is None:
        raise ValueError("range_str is None")
    normalized = range_str.strip().replace('–', '-').replace('—', '-')
    if '-' not in normalized:
        raise ValueError(f"Invalid range format '{range_str}'. Expected 'start-end'.")
    start_s, end_s = normalized.split('-', 1)
    start, end = int(start_s), int(end_s)
    if start > end:
        start, end = end, start
    return start, end


def normalize_range_text(range_str: str) -> str:
    """파일명/파싱용으로 범위 문자열을 ASCII 하이픈으로 정규화."""
    return range_str.strip().replace('–', '-').replace('—', '-').replace(' ', '')


def safe_print_path(prefix: str, path: Path) -> None:
    """
    Windows 콘솔(cp949 등)에서 유니코드 경로 출력 시 UnicodeEncodeError가 날 수 있어서,
    안전하게 출력한다.
    """
    s = str(path)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = s.encode(enc, errors="backslashreplace").decode(enc, errors="ignore")
    print(f"{prefix}{safe}")


def resolve_signature_bar_columns(df: pd.DataFrame) -> tuple[str, str, str, str]:
    """Return (gen_col, rem_col, gen_label, rem_label) for signature count bars / delta."""
    cols = df.columns
    if (
        "plot_generated_survived_actual_only" in cols
        and "plot_removed_not_created_actual_only" in cols
    ):
        return (
            "plot_generated_survived_actual_only",
            "plot_removed_not_created_actual_only",
            "Generated (actual-only survived)",
            "Removed (actual-only not created this turn)",
        )
    if (
        "plot_generated_survived_same_turn" in cols
        and "plot_removed_not_created_this_turn" in cols
    ):
        return (
            "plot_generated_survived_same_turn",
            "plot_removed_not_created_this_turn",
            "Generated (survived same turn)",
            "Removed (not created this turn)",
        )
    if (
        "plot_generated_excl_inactive_reduction" in cols
        and "plot_removed_excl_inactive_reduction" in cols
    ):
        return (
            "plot_generated_excl_inactive_reduction",
            "plot_removed_excl_inactive_reduction",
            "Generated (excl. inactive/reduction)",
            "Removed (excl. inactive/reduction)",
        )
    return "generated", "removed", "Generated", "Removed"


def compute_signature_bar_plot_series(df: pd.DataFrame, gen_col: str, rem_col: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    actual_only plot 컬럼 쌍에 대해, 가능하면 턴 경계 유입/유출(carry_in/carry_out)까지 반영한 2막대로 맞춘다.

    턴 t(행)에서 prev_exit = actual_exit_signature_count[t-1], t가 시간상 첫 행이면 prev_exit = entry_signature_count[t].
      carry_in = max(0, entry[t] - prev_exit)
      carry_out = max(0, prev_exit - entry[t])
      generated_plot_final = plot_generated + carry_in
      removed_plot_final = plot_removed + carry_out
    그러면 prev_exit + generated_plot_final - removed_plot_final = actual_exit[t].

    위에 필요한 컬럼이 없으면 actual_net_change(또는 exit−entry) + removed 로 보조한다.

    그 외 컬럼 조합은 원본(gen_col/rem_col) 값을 그대로 사용한다.
    """
    gen_raw = pd.to_numeric(df[gen_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    rem_raw = pd.to_numeric(df[rem_col], errors="coerce").fillna(0).to_numpy(dtype=float)

    if gen_col == "plot_generated_survived_actual_only" and rem_col == "plot_removed_not_created_actual_only":
        if "entry_signature_count" in df.columns and "actual_exit_signature_count" in df.columns:
            work = df.assign(_orig_pos=np.arange(len(df)))
            if "turn" in work.columns:
                work = work.sort_values("turn", kind="mergesort")
            entry_sig = pd.to_numeric(work["entry_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            actual_exit = pd.to_numeric(work["actual_exit_signature_count"], errors="coerce").fillna(0).to_numpy(
                dtype=float
            )
            gen_w = pd.to_numeric(work[gen_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            rem_w = pd.to_numeric(work[rem_col], errors="coerce").fillna(0).to_numpy(dtype=float)
            if "turn" in work.columns:
                turn_vals = pd.to_numeric(work["turn"], errors="coerce").fillna(0).to_numpy(dtype=int)
            else:
                turn_vals = None
            n = len(work)
            gen_f = np.zeros(n, dtype=float)
            rem_f = np.zeros(n, dtype=float)
            for i in range(n):
                if i > 0 and (turn_vals is None or turn_vals[i] == turn_vals[i - 1] + 1):
                    prev_exit = float(actual_exit[i - 1])
                else:
                    prev_exit = float(entry_sig[i])
                carry_in = max(0.0, float(entry_sig[i] - prev_exit))
                carry_out = max(0.0, float(prev_exit - entry_sig[i]))
                gen_f[i] = gen_w[i] + carry_in
                rem_f[i] = rem_w[i] + carry_out
            gen_out = np.zeros(len(df), dtype=float)
            rem_out = np.zeros(len(df), dtype=float)
            orig = work["_orig_pos"].to_numpy(dtype=int)
            for i in range(n):
                o = orig[i]
                gen_out[o] = gen_f[i]
                rem_out[o] = rem_f[i]
            return gen_out, rem_out, True

        if "actual_net_change" in df.columns:
            net_change = pd.to_numeric(df["actual_net_change"], errors="coerce").fillna(0).to_numpy(dtype=float)
        elif "entry_signature_count" in df.columns and "exit_signature_count" in df.columns:
            entry_sig = pd.to_numeric(df["entry_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            exit_sig = pd.to_numeric(df["exit_signature_count"], errors="coerce").fillna(0).to_numpy(dtype=float)
            net_change = exit_sig - entry_sig
        else:
            net_change = None

        if net_change is not None:
            removed_eff = rem_raw
            generated_eff = net_change + removed_eff
            return generated_eff, removed_eff, True

    return gen_raw, rem_raw, False


def plot_signature_delta_segment(ax: plt.Axes, title: str, df: pd.DataFrame) -> None:
    """구간(연속 turn) 하나를 시그니처 변화 Δ(Generated-Removed)만 ± 색으로 시각화."""
    df = df.reset_index(drop=True)
    x = np.arange(len(df))
    gen_col, rem_col, _, _ = resolve_signature_bar_columns(df)
    gen_series, rem_series, used_eff = compute_signature_bar_plot_series(df, gen_col, rem_col)
    delta = (gen_series - rem_series)

    pos = np.where(delta >= 0, delta, 0)
    neg = np.where(delta < 0, delta, 0)

    ax.bar(
        x[pos > 0],
        pos[pos > 0],
        width=0.6,
        color="lightgreen",
        alpha=0.7,
        edgecolor="darkgreen",
        linewidth=1,
        label="ΔSignature (+)",
        zorder=2,
    )
    ax.bar(
        x[neg < 0],
        neg[neg < 0],
        width=0.6,
        color="lightcoral",
        alpha=0.7,
        edgecolor="darkred",
        linewidth=1,
        label="ΔSignature (−)",
        zorder=2,
    )

    turns = [int(t) for t in df["turn"].values]
    ax.axhline(0, color="black", linewidth=1, alpha=0.6, zorder=1)
    ax.set_title(title, fontsize=40, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in turns], rotation=45, ha="right", fontsize=32)
    # Segment-multi plot에서는 가운데 패널에만 x축 이름을 붙인다.
    if used_eff:
        if "actual_net_change" in df.columns:
            ax.set_ylabel("Δ Signature Count (actual_net_change)", fontsize=40)
        else:
            ax.set_ylabel("Δ Signature Count (exit−entry)", fontsize=40)
    else:
        ax.set_ylabel("Δ Signature Count (Generated - Removed)", fontsize=40)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")


def plot_segment(
    ax_perf: plt.Axes,
    title: str,
    df: pd.DataFrame,
    *,
    show_accuracy: bool,
) -> plt.Axes:
    """구간(연속 turn) 하나를 entry/exit + adaptation 연결 + 시그니처 변화(생성/삭제)로 시각화."""
    base_font = 32
    ax_counts = ax_perf.twinx()
    bar_width = 0.35

    gen_col, rem_col, gen_label, rem_label = resolve_signature_bar_columns(df)

    # 성능 라인 스타일
    metric_specs = [
        ("Recall", "entry_recall", "exit_recall", "o", "blue"),
        ("Precision", "entry_precision", "exit_precision", "x", "purple"),
    ]
    if show_accuracy:
        metric_specs.append(("Accuracy", "entry_accuracy", "exit_accuracy", "d", "green"))

    x_labels: list[str] = []
    x_ticks: list[float] = []

    # Plot each turn
    df = df.reset_index(drop=True)
    bar_gen, bar_rem, used_eff = compute_signature_bar_plot_series(df, gen_col, rem_col)
    if used_eff:
        gen_label = "Generated"
        rem_label = "Removed"
    for i, row in df.iterrows():
        turn = int(row["turn"])
        x_entry = i * 2
        x_exit = i * 2 + 1

        # Learning phase (실선)
        for metric_name, entry_col, exit_col, marker, color in metric_specs:
            # 범례는 Learning만 표시(Adaptation은 범례에서 제외). 괄호 표기도 제거.
            label = f"{metric_name}" if i == 0 else None
            ax_perf.plot(
                [x_entry, x_exit],
                [row[entry_col], row[exit_col]],
                linestyle="-",
                marker=marker,
                color=color,
                linewidth=2,
                markersize=8,
                label=label,
                zorder=3,
            )

        # Adaptation phase (점선): exit(t) -> entry(t+1)
        if i < len(df) - 1:
            next_row = df.iloc[i + 1]
            for metric_name, entry_col, exit_col, marker, color in metric_specs:
                # Adaptation 라인은 그리되 범례에서는 숨김
                label = None
                ax_perf.plot(
                    [x_exit, x_exit + 1],
                    [row[exit_col], next_row[entry_col]],
                    linestyle="--",
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    linewidth=2,
                    markersize=8,
                    dashes=(5, 5),
                    label=label,
                    zorder=2,
                )

        # Count bars (inactive/reduction 제외 기준 컬럼이 있으면 우선 사용)
        bar_center = x_entry + 0.5
        if i == 0:
            ax_counts.bar(
                bar_center - bar_width / 2,
                bar_gen[i],
                bar_width,
                label=gen_label,
                color="lightgreen",
                alpha=0.7,
                edgecolor="darkgreen",
                linewidth=1,
                zorder=1,
            )
            ax_counts.bar(
                bar_center + bar_width / 2,
                bar_rem[i],
                bar_width,
                label=rem_label,
                color="lightcoral",
                alpha=0.7,
                edgecolor="darkred",
                linewidth=1,
                zorder=1,
            )
        else:
            ax_counts.bar(
                bar_center - bar_width / 2,
                bar_gen[i],
                bar_width,
                color="lightgreen",
                alpha=0.7,
                edgecolor="darkgreen",
                linewidth=1,
                zorder=1,
            )
            ax_counts.bar(
                bar_center + bar_width / 2,
                bar_rem[i],
                bar_width,
                color="lightcoral",
                alpha=0.7,
                edgecolor="darkred",
                linewidth=1,
                zorder=1,
            )

        # x축은 entry/exit를 분리 표기하지 않고, 중간 지점에 turn 정수만 표시
        x_ticks.append((x_entry + x_exit) / 2)
        x_labels.append(str(turn))

    ax_perf.set_title(title, fontsize=base_font, pad=10)
    ax_perf.set_xticks(x_ticks)
    ax_perf.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=base_font)
    # Segment-multi plot에서는 가운데 패널에만 x축 이름을 붙인다.
    ax_perf.set_ylabel("Metric Value", fontsize=base_font)
    ax_perf.set_ylim(0, 1.05)
    ax_perf.grid(True, alpha=0.3, linestyle="--")
    ax_perf.tick_params(axis="x", labelsize=base_font)
    ax_perf.tick_params(axis="y", labelsize=base_font)

    ax_counts.set_ylabel("Signature Changes", fontsize=base_font)
    ax_counts.tick_params(axis="y", labelsize=base_font)
    # 오른쪽 y축(시그니처 개수)은 정수로만 표시
    ax_counts.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_counts.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v)}"))
    return ax_counts


# CSV 파일 경로 확인 (긴 경로는 Windows 확장 경로로 해석)
csv_path = resolve_existing_csv(args.csv_path)
if not csv_path.is_file():
    print(f"Error: CSV file not found: {csv_path}")
    parent = csv_path.parent
    if parent.is_dir():
        matches = sorted(parent.glob("*performance_history*.csv"))
        if matches:
            print("Same folder has these performance_history CSV files (copy the exact name):")
            for p in matches:
                print(f"  {p.name}")
        else:
            print(f"No *performance_history*.csv in: {parent}")
    print("Tip: Explorer truncates long names — check the full filename (e.g. ..._ex.csv vs ..._eex.csv).")
    exit(1)

csv_files = [csv_path]

# CSV 파일 처리
for csv_file in csv_files:
    print(f"Processing: {csv_file}")
    
    try:
        history_df = pd.read_csv(csv_file)
        # 컬럼명 정리 (공백 제거)
        history_df.columns = history_df.columns.str.strip()
        
        # 필수 컬럼 확인 (카운트 컬럼은 아래에서 자동 선택)
        required_cols = [
            'turn',
            'entry_recall',
            'exit_recall',
            'entry_precision',
            'exit_precision',
            'entry_f1',
            'exit_f1',
            'entry_accuracy',
            'exit_accuracy',
        ]
        
        missing_cols = [col for col in required_cols if col not in history_df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {history_df.columns.tolist()}")
            continue

        gen_col_main, rem_col_main, gen_label_main, rem_label_main = resolve_signature_bar_columns(history_df)

        if gen_col_main not in history_df.columns or rem_col_main not in history_df.columns:
            print(
                "Error: Count columns not found. Expected one of:\n"
                "  ('plot_generated_survived_actual_only','plot_removed_not_created_actual_only')\n"
                "  ('plot_generated_survived_same_turn','plot_removed_not_created_this_turn')\n"
                "  ('plot_generated_excl_inactive_reduction','plot_removed_excl_inactive_reduction')\n"
                "  ('generated','removed')"
            )
            print(f"Available columns: {history_df.columns.tolist()}")
            continue

        # --- Segment mode: FN / FP / No-attack ranges ---
        if args.fn_range or args.fp_range or args.no_attack_range:
            # Parse provided ranges
            seg_specs: list[tuple[str, str, bool]] = []
            if args.fn_range:
                seg_specs.append(("False Negative segment", args.fn_range, False))
            if args.fp_range:
                seg_specs.append(("False Positive segment", args.fp_range, False))
            if args.no_attack_range:
                seg_specs.append(("Non-attack segment", args.no_attack_range, True))

            segments: list[tuple[str, pd.DataFrame, bool, tuple[int, int]]] = []
            for seg_name, range_str, show_acc in seg_specs:
                start, end = parse_turn_range(range_str)
                seg_df = history_df[(history_df["turn"] >= start) & (history_df["turn"] <= end)].copy()
                if seg_df.empty:
                    print(f"Warning: No data found for {seg_name} range {start}-{end}. Skipping.")
                    continue
                segments.append((seg_name, seg_df, show_acc, (start, end)))

            if not segments:
                print("Error: No valid segment data to plot. Please check --fn_range/--fp_range/--no_attack_range.")
                continue

            # Create a multi-column figure (one column per provided segment)
            fig_width = max(22, 11 * len(segments))
            fig_height = 8.4 if args.mode == "signature_delta" else 8.6
            fig, axes = plt.subplots(1, len(segments), figsize=(fig_width, fig_height), sharey=False)
            if len(segments) == 1:
                axes = [axes]

            seg_count_axes: list[plt.Axes] = []
            for ax, (seg_name, seg_df, show_acc, (start, end)) in zip(axes, segments, strict=False):
                title = seg_name
                if args.mode == "signature_delta":
                    plot_signature_delta_segment(ax, title, seg_df)
                else:
                    ax_counts = plot_segment(ax, title, seg_df, show_accuracy=show_acc)
                    seg_count_axes.append(ax_counts)

            # y축 라벨/틱은 좌측(첫 그래프) 1개, 우측(마지막 그래프의 counts) 1개만 보이게 정리
            if len(axes) > 1:
                for i_ax, ax in enumerate(axes):
                    if i_ax == 0:
                        continue
                    # 이름(라벨)만 숨기고 숫자(틱)는 유지
                    ax.set_ylabel("")
                if seg_count_axes:
                    for i_c, axc in enumerate(seg_count_axes):
                        if i_c == len(seg_count_axes) - 1:
                            continue
                        axc.set_ylabel("")

                # x축 이름은 가운데 패널에만 표시
                mid_idx = (len(axes) - 1) // 2
                axes[mid_idx].set_xlabel("Period", fontsize=32)

            # Build a single legend from ALL segment axes (fix: Accuracy legend might be only in no-attack segment)
            legend_handles: list[object] = []
            legend_labels: list[str] = []

            def add_handles(ax_obj: plt.Axes) -> None:
                h_list, l_list = ax_obj.get_legend_handles_labels()
                for h, l in zip(h_list, l_list, strict=False):
                    if not l or h is None:
                        continue
                    if l in legend_labels:
                        continue
                    legend_handles.append(h)
                    legend_labels.append(l)

            for ax in axes:
                add_handles(ax)
            for axc in seg_count_axes:
                add_handles(axc)

            if legend_handles:
                # 그래프(각 segment) 사이 좌우 간격을 조금 더 띄우고,
                # 그래프와 하단 범례 간격은 더 붙인다.
                # 그래프들 사이 가로 간격을 조금 더 붙임
                fig.subplots_adjust(top=0.93, bottom=0.185, wspace=0.23)
                fig.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    loc="upper center",
                    # 범례를 조금 더 아래로 + 1줄(가로로 길게)
                    bbox_to_anchor=(0.5, 0.092),
                    ncol=len(legend_labels),
                    frameon=True,
                    fontsize=32,
                )
            else:
                fig.subplots_adjust(top=0.93, bottom=0.15, wspace=0.23)

            # Output path
            file_name = csv_file.stem
            parts = file_name.split('_')
            file_type = parts[0] if len(parts) > 0 else 'Unknown'

            base_output_dir = Path(args.output_dir)
            output_dir = base_output_dir / file_type
            output_dir.mkdir(parents=True, exist_ok=True)

            seg_suffix_parts = []
            if args.fn_range:
                seg_suffix_parts.append(f"fn{normalize_range_text(args.fn_range)}")
            if args.fp_range:
                seg_suffix_parts.append(f"fp{normalize_range_text(args.fp_range)}")
            if args.no_attack_range:
                seg_suffix_parts.append(f"noatk{normalize_range_text(args.no_attack_range)}")
            seg_suffix = "_".join(seg_suffix_parts)

            graph_filename = f"{file_name}_segments_{seg_suffix}_{args.mode}.png"
            graph_path = output_dir / graph_filename
            try:
                plt.savefig(graph_path, format="png", dpi=300, bbox_inches="tight")
                safe_print_path("Segment graph saved to: ", graph_path)
            except Exception as e:
                print(f"Failed to save segment graph: {e}")
            plt.close()
            continue
        
        # Parse multiple ranges if provided
        turn_ranges = []
        if args.ranges:
            # Multiple ranges specified
            for range_str in args.ranges:
                try:
                    if '-' in range_str:
                        start, end = range_str.split('-', 1)
                        turn_ranges.append((int(start), int(end)))
                    else:
                        print(f"Warning: Invalid range format '{range_str}'. Expected 'start-end'. Skipping.")
                except ValueError as e:
                    print(f"Warning: Could not parse range '{range_str}': {e}. Skipping.")
            
            if not turn_ranges:
                print("Error: No valid ranges provided. Using all data.")
                turn_ranges = None
        elif args.start_turn is not None or args.end_turn is not None:
            # Single range specified using start_turn/end_turn
            start = args.start_turn if args.start_turn is not None else history_df['turn'].min()
            end = args.end_turn if args.end_turn is not None else history_df['turn'].max()
            turn_ranges = [(start, end)]
        else:
            # No range specified - use all data
            turn_ranges = None
        
        # Filter data based on ranges
        if turn_ranges:
            # Filter to specified ranges
            filtered_dfs = []
            for start, end in turn_ranges:
                range_df = history_df[(history_df['turn'] >= start) & (history_df['turn'] <= end)].copy()
                if len(range_df) > 0:
                    filtered_dfs.append(range_df)
                    print(f"Range {start}-{end}: {len(range_df)} turns")
                else:
                    print(f"Warning: No data found for range {start}-{end}")
            
            if not filtered_dfs:
                print("Error: No data remaining after filtering. Please check your turn ranges.")
                continue
            
            # If multiple ranges, we'll combine them with gaps
            if len(filtered_dfs) > 1:
                # Will be handled in plotting section
                all_range_data = filtered_dfs
            else:
                # Single range - use as normal
                history_df = filtered_dfs[0].reset_index(drop=True)
                all_range_data = None
        else:
            # No filtering - use all data
            all_range_data = None
            print(f"Using all data: {len(history_df)} turns")
        
        if all_range_data is None:
            # Single range or all data - original plotting logic
            print(f"Plotting {len(history_df)} turns: {history_df['turn'].min()} to {history_df['turn'].max()}")
            # 인덱스 리셋 (필터링 후 인덱스가 연속되지 않을 수 있음)
            history_df = history_df.reset_index(drop=True)
        else:
            # Multiple ranges - will be handled separately
            print(f"Plotting {len(all_range_data)} ranges with gaps")

        # Signature bar heights (reactivation-safe) — 반드시 필터/리셋 이후 df 기준으로 계산
        bar_gen_series = None
        bar_rem_series = None
        range_bar_series = None
        if all_range_data is None:
            bar_gen_series, bar_rem_series, used_eff = compute_signature_bar_plot_series(
                history_df, gen_col_main, rem_col_main
            )
            if used_eff:
                gen_label_main = "Generated"
                rem_label_main = "Removed"
        else:
            range_bar_series = []
            for range_df in all_range_data:
                range_df2 = range_df.reset_index(drop=True)
                bg, br, ue = compute_signature_bar_plot_series(range_df2, gen_col_main, rem_col_main)
                range_bar_series.append((bg, br, ue))
            if any(ue for _, _, ue in range_bar_series):
                gen_label_main = "Generated"
                rem_label_main = "Removed"
        
        # 파일명에서 파라미터 추출 (선택사항)
        file_name = csv_file.stem
        parts = file_name.split('_')
        file_type = parts[0] if len(parts) > 0 else 'Unknown'
        param_str = '_'.join(parts[1:-2]) if len(parts) > 2 else 'default'
        
        # y축 범위를 데이터에 맞게 동적으로 설정 (데이터를 그리기 전에 분석)
        # 모든 성능 지표 값 수집
        all_perf_values_pre = []
        if all_range_data is not None and len(all_range_data) > 0:
            # Multiple ranges or single range in list
            for range_df in all_range_data:
                all_perf_values_pre.extend(range_df['entry_recall'].values)
                all_perf_values_pre.extend(range_df['exit_recall'].values)
                all_perf_values_pre.extend(range_df['entry_precision'].values)
                all_perf_values_pre.extend(range_df['exit_precision'].values)
                all_perf_values_pre.extend(range_df['entry_f1'].values)
                all_perf_values_pre.extend(range_df['exit_f1'].values)
                all_perf_values_pre.extend(range_df['entry_accuracy'].values)
                all_perf_values_pre.extend(range_df['exit_accuracy'].values)
        else:
            # Single range or all data - use history_df
            all_perf_values_pre.extend(history_df['entry_recall'].values)
            all_perf_values_pre.extend(history_df['exit_recall'].values)
            all_perf_values_pre.extend(history_df['entry_precision'].values)
            all_perf_values_pre.extend(history_df['exit_precision'].values)
            all_perf_values_pre.extend(history_df['entry_f1'].values)
            all_perf_values_pre.extend(history_df['exit_f1'].values)
            all_perf_values_pre.extend(history_df['entry_accuracy'].values)
            all_perf_values_pre.extend(history_df['exit_accuracy'].values)
        
        # Broken axis를 위한 데이터 분석 (플롯 전에)
        # 실제 데이터 분포를 확인해서 빈 구간을 찾기
        use_broken_axis_pre = False
        lower_max_pre = None
        upper_min_pre = None
        break_start_pre = None
        break_end_pre = None
        
        if all_perf_values_pre:
            valid_values_pre = [v for v in all_perf_values_pre if pd.notna(v) and v > 0]
            if valid_values_pre:
                min_val = min(valid_values_pre)
                max_val = max(valid_values_pre)
                
                # 데이터를 구간별로 나누어서 분포 확인
                # 0.0~1.0을 20개 구간으로 나누기
                num_bins = 20
                bin_width = (max_val - min_val) / num_bins if max_val > min_val else 0.05
                bins = np.linspace(min_val, max_val, num_bins + 1)
                
                # 각 구간에 데이터가 얼마나 있는지 확인
                hist, bin_edges = np.histogram(valid_values_pre, bins=bins)
                
                # 데이터가 거의 없는 빈 구간 찾기 (데이터가 전체의 5% 미만인 구간)
                threshold = len(valid_values_pre) * 0.05
                empty_bins = []
                for i in range(len(hist)):
                    if hist[i] < threshold:
                        empty_bins.append((bin_edges[i], bin_edges[i+1]))
                
                # 가장 큰 빈 구간 찾기
                if empty_bins:
                    largest_gap = max(empty_bins, key=lambda x: x[1] - x[0])
                    gap_size = largest_gap[1] - largest_gap[0]
                    
                    # 빈 구간이 충분히 크면 (전체 범위의 10% 이상) broken axis 사용
                    if gap_size > (max_val - min_val) * 0.1:
                        # 빈 구간 아래와 위에 실제 데이터가 있는지 확인
                        lower_data = [v for v in valid_values_pre if v < largest_gap[0]]
                        upper_data = [v for v in valid_values_pre if v > largest_gap[1]]
                        
                        if len(lower_data) > 0 and len(upper_data) > 0:
                            use_broken_axis_pre = True
                            lower_max_pre = max(lower_data)
                            upper_min_pre = min(upper_data)
                            break_start_pre = lower_max_pre + 0.01
                            break_end_pre = upper_min_pre - 0.01
                            print(f"Broken axis detected: lower max={lower_max_pre:.3f}, upper min={upper_min_pre:.3f}, gap={largest_gap[0]:.3f}-{largest_gap[1]:.3f}, break={break_start_pre:.3f}-{break_end_pre:.3f}")
        
        # 그래프 생성
        fig, ax_perf = plt.subplots(figsize=(20, 11))  # 좌우 너비 증가
        
        # 제목 제거됨
        
        # Broken axis를 위한 상단 subplot 미리 생성 (필요한 경우)
        ax_perf_upper = None
        if use_broken_axis_pre and break_start_pre is not None and break_end_pre is not None:
            # 상단 subplot 생성 (같은 위치에 겹치기)
            ax_perf_upper = fig.add_subplot(111, frameon=False)
            ax_perf_upper.set_ylim(break_end_pre, 1.05)
            ax_perf_upper.spines['top'].set_visible(False)
            ax_perf_upper.spines['bottom'].set_visible(False)
            ax_perf_upper.spines['right'].set_visible(False)
            ax_perf_upper.spines['left'].set_position(ax_perf.spines['left'].get_position())
            ax_perf_upper.tick_params(left=True, labelleft=True, labelsize=40, bottom=False, labelbottom=False)
            ax_perf_upper.set_ylabel('Metric Value', fontsize=40)
            # 하단 subplot 설정
            ax_perf.set_ylim(0, break_start_pre)
            ax_perf.spines['top'].set_visible(False)
        
        # Signature counts용 보조 축 생성
        ax_counts = ax_perf.twinx()
        
        # Helper function to plot on appropriate axis based on value
        def plot_on_axis(x_data, y_data, *args, **kwargs):
            """Plot on appropriate axis based on y values for broken axis"""
            if use_broken_axis_pre and ax_perf_upper is not None and break_start_pre is not None and break_end_pre is not None:
                # Check y values
                y_min = min(y_data) if len(y_data) > 0 else 0
                y_max = max(y_data) if len(y_data) > 0 else 0
                
                # If any value is in upper range, plot on upper axis
                if y_max >= break_end_pre:
                    # Transform: values >= break_end_pre stay as is, values < break_end_pre are set to break_end_pre
                    y_upper = [y if y >= break_end_pre else break_end_pre for y in y_data]
                    ax_perf_upper.plot(x_data, y_upper, *args, **kwargs)
                
                # If any value is in lower range, plot on lower axis
                if y_min <= break_start_pre:
                    # Transform: values <= break_start_pre stay as is, values > break_start_pre are set to break_start_pre
                    y_lower = [y if y <= break_start_pre else break_start_pre for y in y_data]
                    ax_perf.plot(x_data, y_lower, *args, **kwargs)
            else:
                # Normal plotting
                ax_perf.plot(x_data, y_data, *args, **kwargs)
        
        x_labels = []
        x_ticks = []
        bar_width = 0.35
        
        # Handle multiple ranges or single range
        if all_range_data and len(all_range_data) > 1:
            # Multiple ranges with gaps
            current_x = 0
            first_range = True
            
            for range_idx, range_df in enumerate(all_range_data):
                range_df = range_df.reset_index(drop=True)
                
                # Add gap with ellipsis between ranges
                if not first_range:
                    gap_start = current_x + 0.1
                    gap_end = gap_start + args.gap_width
                    
                    # Draw ellipsis
                    gap_center = (gap_start + gap_end) / 2
                    gap_y = 0.5
                    for dot_offset in [-0.6, 0, 0.6]:
                        dot_x = gap_center + dot_offset
                        ax_perf.plot([dot_x, dot_x], [gap_y - 0.05, gap_y + 0.05], 
                                    'k-', linewidth=3, alpha=0.6)
                    
                    current_x = gap_end
                
                first_range = False
                
                # Plot each turn in this range
                for i, row in range_df.iterrows():
                    turn = row['turn']
                    x_entry = current_x
                    x_exit = current_x + 0.9
                    
                    # --- 1. Performance Lines (Learning phase) ---
                    if range_idx == 0 and i == 0:
                        plot_on_axis([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 
                                    'o-', color='blue', label='Recall (Learning)', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 
                                    'x-', color='purple', label='Precision (Learning)', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 
                                    's-', color='orange', label='F1-Score (Learning)', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 
                                    'd-', color='green', label='Accuracy (Learning)', linewidth=2, markersize=8)
                    else:
                        plot_on_axis([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 
                                    'o-', color='blue', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 
                                    'x-', color='purple', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 
                                    's-', color='orange', linewidth=2, markersize=8)
                        plot_on_axis([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 
                                    'd-', color='green', linewidth=2, markersize=8)
                    
                    # --- 2. Adaptation phase ---
                    if i < len(range_df) - 1:
                        next_row = range_df.iloc[i+1]
                        if range_idx == 0 and i == 0:
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_recall'], next_row['entry_recall']], 
                                        'o--', color='blue', alpha=0.7, label='Recall (Adaptation)', 
                                        linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_precision'], next_row['entry_precision']], 
                                        'x--', color='purple', alpha=0.7, label='Precision (Adaptation)', 
                                        linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_f1'], next_row['entry_f1']], 
                                        's--', color='orange', alpha=0.7, label='F1-Score (Adaptation)', 
                                        linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_accuracy'], next_row['entry_accuracy']], 
                                        'd--', color='green', alpha=0.7, label='Accuracy (Adaptation)', 
                                        linewidth=2, markersize=8, dashes=(5, 5))
                        else:
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_recall'], next_row['entry_recall']], 
                                        'o--', color='blue', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_precision'], next_row['entry_precision']], 
                                        'x--', color='purple', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_f1'], next_row['entry_f1']], 
                                        's--', color='orange', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                            plot_on_axis([x_exit, x_exit + 0.9], [row['exit_accuracy'], next_row['entry_accuracy']], 
                                        'd--', color='green', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                    
                    # --- 3. Count Bars ---
                    bar_center = x_entry + 0.45
                    if range_idx == 0 and i == 0:
                        ax_counts.bar(bar_center - bar_width/2, range_bar_series[range_idx][0][i], bar_width,
                                     label=gen_label_main, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                        ax_counts.bar(bar_center + bar_width/2, range_bar_series[range_idx][1][i], bar_width,
                                     label=rem_label_main, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                    else:
                        ax_counts.bar(bar_center - bar_width/2, range_bar_series[range_idx][0][i], bar_width,
                                     color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                        ax_counts.bar(bar_center + bar_width/2, range_bar_series[range_idx][1][i], bar_width,
                                     color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                    
                    x_ticks.append(x_entry)
                    x_labels.append(f"{turn}-entry")
                    x_ticks.append(x_exit)
                    x_labels.append(f"{turn}-exit")
                    
                    current_x = x_exit + 0.1
        else:
            # Single range or all data - original plotting logic
            # 각 turn에 대해 플롯
            for i, row in history_df.iterrows():
                turn = row['turn']
                x_entry = i * 2
                x_exit = i * 2 + 1
                
                # --- 1. Performance Lines (Learning phase - 실선) ---
                if i == 0:
                    # 첫 번째 iteration에서만 범례 라벨 추가
                    plot_on_axis([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 
                                'o-', color='blue', label='Recall (Learning)', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 
                                'x-', color='purple', label='Precision (Learning)', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 
                                's-', color='orange', label='F1-Score (Learning)', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 
                                'd-', color='green', label='Accuracy (Learning)', linewidth=2, markersize=8)
                else:
                    # 나머지 iteration에서는 라벨 없이 그리기
                    plot_on_axis([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 
                                'o-', color='blue', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 
                                'x-', color='purple', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 
                                's-', color='orange', linewidth=2, markersize=8)
                    plot_on_axis([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 
                                'd-', color='green', linewidth=2, markersize=8)
                
                # --- 2. Adaptation phase (점선) ---
                if i < len(history_df) - 1:
                    next_row = history_df.iloc[i+1]
                    if i == 0:
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_recall'], next_row['entry_recall']], 
                                    'o--', color='blue', alpha=0.7, label='Recall (Adaptation)', 
                                    linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_precision'], next_row['entry_precision']], 
                                    'x--', color='purple', alpha=0.7, label='Precision (Adaptation)', 
                                    linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_f1'], next_row['entry_f1']], 
                                    's--', color='orange', alpha=0.7, label='F1-Score (Adaptation)', 
                                    linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_accuracy'], next_row['entry_accuracy']], 
                                    'd--', color='green', alpha=0.7, label='Accuracy (Adaptation)', 
                                    linewidth=2, markersize=8, dashes=(5, 5))
                    else:
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_recall'], next_row['entry_recall']], 
                                    'o--', color='blue', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_precision'], next_row['entry_precision']], 
                                    'x--', color='purple', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_f1'], next_row['entry_f1']], 
                                    's--', color='orange', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                        plot_on_axis([x_exit, x_exit + 1], [row['exit_accuracy'], next_row['entry_accuracy']], 
                                    'd--', color='green', alpha=0.7, linewidth=2, markersize=8, dashes=(5, 5))
                
                # --- 3. Count Bars (막대 그래프) ---
                bar_center = x_entry + 0.5
                if i == 0:
                    ax_counts.bar(bar_center - bar_width/2, bar_gen_series[i], bar_width,
                                 label=gen_label_main, color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                    ax_counts.bar(bar_center + bar_width/2, bar_rem_series[i], bar_width,
                                 label=rem_label_main, color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                else:
                    ax_counts.bar(bar_center - bar_width/2, bar_gen_series[i], bar_width,
                                 color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                    ax_counts.bar(bar_center + bar_width/2, bar_rem_series[i], bar_width,
                                 color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                
                x_ticks.extend([x_entry, x_exit])
                x_labels.extend([f"{turn}-entry", f"{turn}-exit"])
        
        # 축 설정
        ax_perf.set_xticks(x_ticks)
        ax_perf.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=40)
        ax_perf.set_xlabel('Turn (Entry/Exit)', fontsize=40)
        ax_perf.set_ylabel('Metric Value', fontsize=40)
        ax_perf.tick_params(axis='x', labelsize=40)
        ax_perf.tick_params(axis='y', labelsize=40)
        
        # y축 범위를 데이터에 맞게 동적으로 설정
        # 모든 성능 지표 값 수집
        all_perf_values = []
        if all_range_data is not None and len(all_range_data) > 0:
            # Multiple ranges or single range in list
            for range_df in all_range_data:
                all_perf_values.extend(range_df['entry_recall'].values)
                all_perf_values.extend(range_df['exit_recall'].values)
                all_perf_values.extend(range_df['entry_precision'].values)
                all_perf_values.extend(range_df['exit_precision'].values)
                all_perf_values.extend(range_df['entry_f1'].values)
                all_perf_values.extend(range_df['exit_f1'].values)
                all_perf_values.extend(range_df['entry_accuracy'].values)
                all_perf_values.extend(range_df['exit_accuracy'].values)
        else:
            # Single range or all data - use history_df
            all_perf_values.extend(history_df['entry_recall'].values)
            all_perf_values.extend(history_df['exit_recall'].values)
            all_perf_values.extend(history_df['entry_precision'].values)
            all_perf_values.extend(history_df['exit_precision'].values)
            all_perf_values.extend(history_df['entry_f1'].values)
            all_perf_values.extend(history_df['exit_f1'].values)
            all_perf_values.extend(history_df['entry_accuracy'].values)
            all_perf_values.extend(history_df['exit_accuracy'].values)
        
        # y축 범위 설정 (broken axis 분석은 이미 위에서 완료됨)
        if all_perf_values:
            # NaN과 0 값을 제거하고 유효한 값만 사용
            valid_values = [v for v in all_perf_values if pd.notna(v) and v > 0]
            
            if valid_values:
                min_val = min(valid_values)
                max_val = max(valid_values)
                
                # Broken axis를 사용하지 않는 경우에만 일반 y축 범위 설정
                if not use_broken_axis_pre:
                    # 기존 로직: 최소값이 0.8 이상이면 0.8부터 시작
                    if min_val >= 0.8:
                        y_min = 0.8
                    else:
                        # 데이터의 대부분이 0.8 이상인지 확인 (75% 이상이면 0.8부터 시작)
                        high_values = [v for v in valid_values if v >= 0.8]
                        if len(high_values) / len(valid_values) >= 0.75:
                            y_min = 0.8
                        else:
                            # 최소값에서 약간 여유를 두고
                            y_min = max(0, min_val - (max_val - min_val) * 0.1)  # 최소값의 10% 여유
                    
                    ax_perf.set_ylim(y_min, 1.05)
                    print(f"Y-axis range set to: {y_min:.3f} to 1.05 (data range: {min_val:.3f} to {max_val:.3f}, valid values: {len(valid_values)})")
                else:
                    # Broken axis를 사용하는 경우: 하단과 상단 모두 표시
                    # 하단은 0부터 lower_max_pre까지, 상단은 upper_min_pre부터 1.05까지
                    ax_perf.set_ylim(0, break_start_pre)
                    print(f"Broken axis y-axis: lower 0-{break_start_pre:.3f}, upper {break_end_pre:.3f}-1.05")
            else:
                ax_perf.set_ylim(0, 1.05)
                print("No valid performance data found (all NaN or 0), using default y-axis range: 0 to 1.05")
        else:
            ax_perf.set_ylim(0, 1.05)
            print("No performance data found, using default y-axis range: 0 to 1.05")
        
        # Broken axis 물결선 그리기 (데이터를 그린 후, 축 설정 후)
        if use_broken_axis_pre and break_start_pre is not None and break_end_pre is not None and ax_perf_upper is not None:
            # x축 위치 (y축 왼쪽, 약간 왼쪽으로)
            x_lim = ax_perf.get_xlim()
            x_wave = x_lim[0] - (x_lim[1] - x_lim[0]) * 0.03  # 더 왼쪽으로 이동
            
            # 물결선 그리기 (사인파 형태)
            wave_points = 100
            wave_width = (x_lim[1] - x_lim[0]) * 0.01
            wave_x = np.linspace(x_wave - wave_width, x_wave + wave_width, wave_points)
            
            # 하단 물결선 (break_start_pre 근처)
            wave_amp_bottom = (break_start_pre - 0) * 0.015
            wave_y_bottom = break_start_pre + wave_amp_bottom * np.sin(np.linspace(0, 4*np.pi, wave_points))
            
            # 상단 물결선 (break_end_pre 근처)
            wave_amp_top = (1.05 - break_end_pre) * 0.015
            wave_y_top = break_end_pre - wave_amp_top * np.sin(np.linspace(0, 4*np.pi, wave_points))
            
            # 하단 물결선 (흰색으로 두껍게, 배경에)
            # 세로선
            ax_perf.plot([x_wave, x_wave], 
                        [break_start_pre - wave_amp_bottom * 1.5, break_start_pre + wave_amp_bottom * 1.5], 
                        'w-', linewidth=12, zorder=1000, clip_on=False)
            # 물결선
            ax_perf.plot(wave_x, wave_y_bottom, 'w-', linewidth=10, zorder=1000, clip_on=False)
            
            # 상단 물결선 (상단 subplot에)
            ax_perf_upper.plot([x_wave, x_wave], 
                              [break_end_pre - wave_amp_top * 1.5, break_end_pre + wave_amp_top * 1.5], 
                              'w-', linewidth=12, zorder=1000, clip_on=False)
            ax_perf_upper.plot(wave_x, wave_y_top, 'w-', linewidth=10, zorder=1000, clip_on=False)
            
            # 상단 subplot의 x축 범위 설정
            ax_perf_upper.set_xlim(ax_perf.get_xlim())
            
            print(f"Broken axis applied: lower 0-{break_start_pre:.3f}, upper {break_end_pre:.3f}-1.05")
        
        ax_perf.grid(True, linestyle='--', alpha=0.3)
        
        ax_counts.set_ylabel('Signature Count', fontsize=40, color='black')
        ax_counts.tick_params(axis='y', labelcolor='black', labelsize=40)
        ax_counts.set_ylim(bottom=0)
        
        # 범례 통합 및 순서 지정
        handles_perf, labels_perf = ax_perf.get_legend_handles_labels()
        handles_counts, labels_counts = ax_counts.get_legend_handles_labels()
        
        # 범례 항목을 딕셔너리로 저장
        legend_dict = {}
        for handle, label in zip(handles_perf + handles_counts, labels_perf + labels_counts):
            if label not in legend_dict:
                legend_dict[label] = handle
        
        # 범례 순서 명시적으로 지정 (3열 배치)
        # 행1: Recall (Learning), Precision (Learning), F1-Score (Learning)
        # 행2: Accuracy (Learning), Recall (Adaptation), Precision (Adaptation)
        # 행3: F1-Score (Adaptation), Accuracy (Adaptation), Generated
        # 행4: Removed, (빈), (빈)
        from matplotlib.patches import Rectangle as Rect
        
        ordered_labels = [
            'Recall (Learning)', 'Precision (Learning)', 'F1-Score (Learning)',
            'Accuracy (Learning)', 'Recall (Adaptation)', 'Precision (Adaptation)',
            'F1-Score (Adaptation)', 'Accuracy (Adaptation)',
            gen_label_main,
            rem_label_main,
        ]
        
        ordered_handles = []
        ordered_labels_final = []
        for label in ordered_labels:
            if label in legend_dict:
                ordered_handles.append(legend_dict[label])
                ordered_labels_final.append(label)
        
        # 빈 항목 생성 (보이지 않지만 공간 차지)
        empty_patch = plt.Rectangle((0,0), 0.001, 0.001, 
                                    facecolor='white', edgecolor='white', 
                                    alpha=0.0, linewidth=0, label=' ')
        
        # 3열로 배치하기 위해 빈 항목 추가
        # 총 10개 항목이므로 3열로 하면 4행이 필요 (마지막 행에 빈 항목 2개)
        while len(ordered_handles) % 3 != 0:
            ordered_handles.append(empty_patch)
            ordered_labels_final.append(' ')
        
        # 범례를 아래쪽에 배치 (create_conditions_graph.py처럼)
        # 위쪽 여백 확보 및 아래쪽 여백 조정 (x축 제목과 범례 공간 확보)
        plt.subplots_adjust(top=0.94, bottom=0.48)  # 아래쪽 여백 더 늘림
        
        # 범례 생성: 그래프 아래쪽으로 분리
        ax_perf.legend(ordered_handles, ordered_labels_final, loc='upper center', 
                       bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=True, fontsize=40)
        
        plt.tight_layout(rect=[0, 0, 1, 1], pad=0.1)  # 범례가 잘리지 않도록 조정
        
        # 출력 디렉토리 설정 (file_type별 하위 폴더 생성)
        base_output_dir = Path(args.output_dir)
        output_dir = base_output_dir / file_type
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 그래프 파일명 생성
        turn_range_str = ""
        if args.ranges and turn_ranges:
            # Multiple ranges
            range_parts = [f"{tr[0]}-{tr[1]}" for tr in turn_ranges]
            turn_range_str = f"_turns{'_'.join(range_parts)}"
        elif args.start_turn is not None or args.end_turn is not None:
            # Single range
            if all_range_data is None and 'history_df' in locals():
                start = args.start_turn if args.start_turn is not None else history_df['turn'].min()
                end = args.end_turn if args.end_turn is not None else history_df['turn'].max()
                turn_range_str = f"_turns{start}-{end}"
            elif turn_ranges:
                start = turn_ranges[0][0]
                end = turn_ranges[0][1]
                turn_range_str = f"_turns{start}-{end}"
        
        graph_filename = f"{file_name}{turn_range_str}_metrics_eex.png"
        graph_path = output_dir / graph_filename
        
        try:
            plt.savefig(graph_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Performance graph saved to: {graph_path}")
        except Exception as e:
            print(f"Failed to save graph: {e}")
        
        plt.close()
        
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        import traceback
        traceback.print_exc()

print("\nAll graphs generated successfully!")

