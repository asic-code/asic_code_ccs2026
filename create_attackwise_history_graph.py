import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import re
from pathlib import Path

from matplotlib.patches import Patch


def strip_parenthetical(text: str) -> str:
    """범례/라벨에서 ' (…)' 형태의 괄호 구간을 모두 제거한다."""
    s = str(text)
    while True:
        t = re.sub(r"\s*\([^)]*\)", "", s)
        if t == s:
            break
        s = t
    return s.strip()


def is_recall_entry_legend_label(label: str) -> bool:
    """attack-wise 범례에서 entry 재현 선만 숨길 때 사용 (exit 막대/라벨은 유지)."""
    s = str(label).strip().lower()
    if s == "entry":
        return True
    if s.endswith(" entry"):
        return True
    return False


def max_attack_recall_in_scope(
    attack: str,
    scope_dfs: list[pd.DataFrame],
    entry_cols: dict[str, str],
    exit_cols: dict[str, str],
    include_entry: bool,
) -> float:
    """
    선택된 turn 구간(scope_dfs)에서, 실제로 그릴 recall 열의 최댓값.
    판단 불가(열 없음·전부 NaN)이면 nan을 반환해 필터에서 제외하지 않는다.
    """
    chunks: list[np.ndarray] = []
    ec = entry_cols.get(attack)
    xc = exit_cols.get(attack)
    if include_entry and ec:
        for df in scope_dfs:
            if ec in df.columns:
                chunks.append(pd.to_numeric(df[ec], errors="coerce").to_numpy(dtype=float))
    if xc:
        for df in scope_dfs:
            if xc in df.columns:
                chunks.append(pd.to_numeric(df[xc], errors="coerce").to_numpy(dtype=float))
    elif not include_entry:
        return float("nan")
    if not chunks:
        return float("nan")
    vals = np.concatenate(chunks)
    if not np.any(np.isfinite(vals)):
        return float("nan")
    return float(np.nanmax(vals))


def pick_attackwise_legend_nrows(total_slots: int, max_nrows: int = 6) -> int:
    """
    범례 칸 수(total_slots)에 맞춰 행 수를 정한다.
    열은 최대 5열까지 허용하고, 그보다 많은 열이 필요하면 행을 늘린다(최대 max_nrows).
    """
    if total_slots <= 9:
        return 3
    target_ncol = 5
    for nr in range(3, max_nrows + 1):
        ncol = max(2, int(np.ceil(total_slots / nr)))
        if ncol <= target_ncol:
            return nr
    return max_nrows


def pick_attackwise_legend_ncol(n_attacks: int, ncol_max: int = 10) -> int:
    """공격 라벨은 대략 2행, Generated/Removed는 마지막 1행(좌측부터)으로 내려가도록 열 개수 선택."""
    if n_attacks <= 0:
        return 4
    # 공격 라벨을 2행 정도로 맞추고(+ 마지막 1행은 bars) 전체 3행 내외를 목표로 한다.
    ncol = int(np.ceil(n_attacks / 2))
    return max(4, min(ncol_max, ncol))


def build_attackwise_combined_legend_grid(
    line_pairs: list[tuple],
    bar_pairs: list[tuple],
    ncol_max: int = 10,
) -> tuple[list, list, int, int]:
    """
    attack-wise 합친 그래프 범례: 그림 가로에 맞게 행 수를 늘려 ncol을 줄인다.

    - 공격 수가 적으면 3 rows.
    - 열은 최대 5열까지; 그 이상 필요하면 최대 6 rows까지 늘린다.
    - Generated / Removed는 마지막 행에서 공격 라벨 뒤에 붙인다.

    matplotlib legend(ncol)은 column-major 배치이므로, grid를 채운 뒤 column-major로 flatten한다.
    """

    def blank_item():
        return (Patch(facecolor="none", edgecolor="none", linewidth=0, label=" "), " ")

    # attack line items
    line_items = [(h, strip_parenthetical(lb)) for h, lb in line_pairs]

    # Generated / Removed handle 찾기
    gen_item = None
    rem_item = None

    for h, lb in bar_pairs:
        sl = strip_parenthetical(lb).strip().lower()
        if sl == "generated":
            gen_item = (h, "Generated")
        elif sl == "removed":
            rem_item = (h, "Removed")

    # Generated / Removed가 없으면 일반 multi-row legend로 fallback
    if gen_item is None or rem_item is None:
        items = line_items + [(h, strip_parenthetical(lb)) for h, lb in bar_pairs]
        if not items:
            return [], [], 1, 1

        nrows = pick_attackwise_legend_nrows(len(items))
        ncol = max(1, int(np.ceil(len(items) / nrows)))
        if ncol_max is not None:
            ncol = min(ncol, ncol_max)
        if ncol * nrows < len(items):
            ncol = int(np.ceil(len(items) / nrows))
            ncol = max(1, ncol)

        grid = [[blank_item() for _ in range(ncol)] for _ in range(nrows)]

        for idx, item in enumerate(items):
            r = idx // ncol
            c = idx % ncol
            if r < nrows:
                grid[r][c] = item

        handles: list = []
        labels: list[str] = []
        for c in range(ncol):
            for r in range(nrows):
                h, lb = grid[r][c]
                handles.append(h)
                labels.append(lb)

        return handles, labels, ncol, nrows

    # Generated / Removed가 있는 경우
    n_attacks = len(line_items)
    total_slots = n_attacks + 2

    nrows = pick_attackwise_legend_nrows(total_slots)

    ncol = max(2, int(np.ceil(total_slots / nrows)))
    if ncol_max is not None:
        ncol = min(ncol, ncol_max)
    if ncol * nrows < total_slots:
        ncol = int(np.ceil(total_slots / nrows))
        ncol = max(2, ncol)

    # 빈 grid 생성: grid[row][col]
    grid = [[blank_item() for _ in range(ncol)] for _ in range(nrows)]

    # 마지막 row에서 Generated / Removed를 attack 뒤에 붙인다.
    attacks_in_last_row = max(0, n_attacks - (nrows - 1) * ncol)
    bar_start_col = min(attacks_in_last_row, ncol - 2)
    grid[nrows - 1][bar_start_col] = gen_item
    grid[nrows - 1][bar_start_col + 1] = rem_item

    reserved = {
        (nrows - 1, bar_start_col),
        (nrows - 1, bar_start_col + 1),
    }

    # attack labels를 row-major로 채움. Generated / Removed 위치는 건너뜀.
    cells = []
    for r in range(nrows):
        for c in range(ncol):
            if (r, c) not in reserved:
                cells.append((r, c))
    for item, (r, c) in zip(line_items, cells):
        grid[r][c] = item

    # matplotlib legend는 column-major로 읽히므로 column-major로 flatten
    handles: list = []
    labels: list[str] = []
    for c in range(ncol):
        for r in range(nrows):
            h, lb = grid[r][c]
            handles.append(h)
            labels.append(lb)

    return handles, labels, ncol, nrows


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

    def _is_actual_only_pair(g: str, r: str) -> bool:
        # plot_generated_survived_actual_only[_suffix] / plot_removed_not_created_actual_only[_suffix]
        return g.startswith("plot_generated_survived_actual_only") and r.startswith(
            "plot_removed_not_created_actual_only"
        )

    if _is_actual_only_pair(gen_col, rem_col):
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


# 폰트 설정 (Times New Roman)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  # 기본 폰트 크기
# adaptation 세그먼트 그래프(plot_segment)와 동일한 축/틱/범례 글자 크기
BASE_FONT_SEG = 32

# Argument parser 설정
parser = argparse.ArgumentParser(description='Generate performance history graph from CSV file')
DEFAULT_CSV_PATH = r"C:\ASIC_excute\Dataset\Dataset_ISV_turnmap\CICIDS2017\attackwise\CICIDS2017_1_rarm_s0.01_c0.9_cstem_l10_ns0.06_n200_dom0.99_dom(da=0.9)_pul0.7_sepF(turn)_turneval_trtsNA-teNA_attackwise_performance_history_eex.csv"
parser.add_argument(
    'csv_path',
    type=str,
    nargs='?',
    default=DEFAULT_CSV_PATH,
    help='Path to the attack-wise performance history CSV file (default: %(default)s)',
)
parser.add_argument('--start_turn', type=int, default=None, 
                    help='Starting turn number to plot (default: first turn in CSV). Ignored if --ranges is used.')
parser.add_argument('--end_turn', type=int, default=None,
                    help='Ending turn number to plot (default: last turn in CSV). Ignored if --ranges is used.')
parser.add_argument('--ranges', type=str, nargs='+', default=None,
                    help='Multiple turn ranges in format "start-end". Example: --ranges "10-20" "40-50" "90-100"')
parser.add_argument(
    "--include_entry",
    action="store_true",
    help="Also plot entry recall lines (default: exit only).",
)
parser.add_argument(
    '--per_attack',
    action='store_true',
    help='Also generate one PNG per attack type (default: only a single combined graph).',
)
parser.add_argument('--output_dir', type=str, default='attackwise',
                    help='Output directory for attack-wise graphs (default: attackwise)')
parser.add_argument('--gap_width', type=float, default=3.0,
                    help='Width of gap between turn ranges when using --ranges (default: 3.0)')
parser.add_argument(
    "--keep_no_alert",
    type=int,
    default=0,
    choices=[0, 1],
    help="If 1, prefer *_no_alert_excluded actual_only columns for Generated/Removed bars (default: 0).",
)
parser.add_argument(
    "--include_all_zero_recall_attacks",
    action="store_true",
    help=(
        "Plot attacks whose recall is 0 on every plotted turn in the selected scope "
        "(default: omit them from combined and per-attack outputs)."
    ),
)

args = parser.parse_args()

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
        
        # 필수 컬럼 확인 (attack-wise 그래프는 전체 성능/카운트 컬럼에 의존하지 않음)
        required_cols = ['turn']
        
        missing_cols = [col for col in required_cols if col not in history_df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {history_df.columns.tolist()}")
            continue

        # --- Attack-wise recall columns discovery ---
        # Attack-wise recall columns are expected after exit_accuracy, but we detect them by name
        # because file_type마다 attack type 종류/개수가 다를 수 있음.
        col_re = re.compile(r"^(entry|exit)_attack_recall_(.+)$")
        entry_cols: dict[str, str] = {}
        exit_cols: dict[str, str] = {}
        for col in history_df.columns:
            m = col_re.match(col)
            if not m:
                continue
            which, attack = m.group(1), m.group(2)
            if which == "entry":
                entry_cols[attack] = col
            else:
                exit_cols[attack] = col

        attack_types = sorted(set(entry_cols.keys()) | set(exit_cols.keys()))
        if not attack_types:
            print("Error: No attack-wise recall columns found. Expected columns like 'entry_attack_recall_*' / 'exit_attack_recall_*'.")
            print("Available columns after exit_accuracy include:")
            try:
                idx = list(history_df.columns).index("exit_accuracy")
                print(history_df.columns[idx + 1 :].tolist())
            except ValueError:
                print(history_df.columns.tolist())
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

        # 선택 turn 구간에서 전부 0인 공격은 기본 제외 (--include_all_zero_recall_attacks 시 유지)
        if all_range_data is None:
            scope_dfs = [history_df]
        else:
            scope_dfs = [d.reset_index(drop=True) for d in all_range_data]

        if not args.include_all_zero_recall_attacks:
            kept_attacks: list[str] = []
            dropped_zero: list[str] = []
            for atk in attack_types:
                mx = max_attack_recall_in_scope(
                    atk, scope_dfs, entry_cols, exit_cols, args.include_entry
                )
                if np.isnan(mx) or mx > 1e-12:
                    kept_attacks.append(atk)
                else:
                    dropped_zero.append(atk)
            if dropped_zero:
                print(
                    f"Omitting {len(dropped_zero)} attack(s) with zero recall on all plotted turns: "
                    f"{', '.join(dropped_zero)}"
                )
            if not kept_attacks:
                print(
                    "Error: All attacks would be omitted (all-zero recall in scope). "
                    "Use --include_all_zero_recall_attacks to plot them anyway."
                )
                continue
            attack_types = kept_attacks

        # --- Attack-wise plotting ---
        # 각 attack type별로 entry/exit recall을 turn 축으로 그린 그래프를 개별 저장한다.
        def iter_plot_df():
            if all_range_data is None:
                yield history_df, None
            else:
                for df_part in all_range_data:
                    yield df_part.reset_index(drop=True), None

        # Output directory: attackwise/<file_type>/
        file_name = csv_file.stem
        parts = file_name.split('_')
        file_type = parts[0] if len(parts) > 0 else 'Unknown'
        base_output_dir = Path(args.output_dir)
        output_dir = base_output_dir / file_type
        output_dir.mkdir(parents=True, exist_ok=True)

        # Range suffix (optional)
        turn_range_str = ""
        if args.ranges and turn_ranges:
            range_parts = [f"{tr[0]}-{tr[1]}" for tr in turn_ranges]
            turn_range_str = f"_turns{'_'.join(range_parts)}"
        elif args.start_turn is not None or args.end_turn is not None:
            start = args.start_turn if args.start_turn is not None else int(history_df['turn'].min())
            end = args.end_turn if args.end_turn is not None else int(history_df['turn'].max())
            turn_range_str = f"_turns{start}-{end}"

        # 1) Combined graph (default): all attacks in one figure
        turns = history_df["turn"].astype(int).values
        cmap = plt.get_cmap("tab20")

        # 세로를 조금 넓혀 데이터 영역(막대+라인) 비율 개선
        plt.figure(figsize=(29, 9.5))
        fig = plt.gcf()
        ax_all = plt.gca()

        # Optional: counts bars on secondary axis (prefer newest plot_* columns if present)
        cols = history_df.columns
        if args.keep_no_alert == 1 and (
            "plot_generated_survived_actual_only_no_alert_excluded" in cols
            and "plot_removed_not_created_actual_only_no_alert_excluded" in cols
        ):
            gen_col = "plot_generated_survived_actual_only_no_alert_excluded"
            rem_col = "plot_removed_not_created_actual_only_no_alert_excluded"
            gen_label = "Generated"
            rem_label = "Removed"
        elif (
            "plot_generated_survived_actual_only" in cols
            and "plot_removed_not_created_actual_only" in cols
        ):
            gen_col = "plot_generated_survived_actual_only"
            rem_col = "plot_removed_not_created_actual_only"
            gen_label = "Generated"
            rem_label = "Removed"
        elif (
            "plot_generated_survived_same_turn" in cols
            and "plot_removed_not_created_this_turn" in cols
        ):
            gen_col = "plot_generated_survived_same_turn"
            rem_col = "plot_removed_not_created_this_turn"
            gen_label = "Generated"
            rem_label = "Removed"
        elif (
            "plot_generated_excl_inactive_reduction" in cols
            and "plot_removed_excl_inactive_reduction" in cols
        ):
            gen_col = "plot_generated_excl_inactive_reduction"
            rem_col = "plot_removed_excl_inactive_reduction"
            gen_label = "Generated"
            rem_label = "Removed"
        elif "generated" in cols and "removed" in cols:
            gen_col = "generated"
            rem_col = "removed"
            gen_label = "Generated"
            rem_label = "Removed"
        else:
            gen_col = None
            rem_col = None
            gen_label = None
            rem_label = None

        ax_counts = None
        if gen_col and rem_col:
            ax_counts = ax_all.twinx()
            bar_width = 0.35
            bar_gen, bar_rem, used_eff = compute_signature_bar_plot_series(history_df, gen_col, rem_col)
            if used_eff:
                gen_label = "Generated"
                rem_label = "Removed"
            ax_counts.bar(
                turns - bar_width / 2,
                bar_gen,
                width=bar_width,
                color="lightgreen",
                alpha=0.35,
                edgecolor="darkgreen",
                linewidth=1,
                label=strip_parenthetical(gen_label),
                zorder=1,
            )
            ax_counts.bar(
                turns + bar_width / 2,
                bar_rem,
                width=bar_width,
                color="lightcoral",
                alpha=0.35,
                edgecolor="darkred",
                linewidth=1,
                label=strip_parenthetical(rem_label),
                zorder=1,
            )
            ax_counts.set_ylabel("Signature Changes", fontsize=BASE_FONT_SEG)
            ax_counts.tick_params(axis="y", labelsize=BASE_FONT_SEG)
            ax_counts.set_ylim(bottom=0)

        for i, attack in enumerate(attack_types):
            entry_col = entry_cols.get(attack)
            exit_col = exit_cols.get(attack)
            entry_vals = history_df[entry_col].values if entry_col else np.full(len(history_df), np.nan)
            exit_vals = history_df[exit_col].values if exit_col else np.full(len(history_df), np.nan)
            color = cmap(i % 20)

            # Default: exit only. Optional: include entry (dashed) via --include_entry
            ax_all.plot(turns, exit_vals, "-", color=color, linewidth=3, label=f"{attack}")
            if args.include_entry:
                ax_all.plot(turns, entry_vals, "--", color=color, alpha=0.6, linewidth=2, label=f"{attack} entry")

        ax_all.set_xlabel("Period", fontsize=BASE_FONT_SEG)
        ax_all.set_ylabel("Attack Coverage", fontsize=BASE_FONT_SEG)
        ax_all.set_ylim(0, 1.05)
        # 모든 turn(period) 값을 x축에 표시
        ax_all.set_xticks(turns)
        ax_all.set_xticklabels([str(int(t)) for t in turns], rotation=0, ha="center", fontsize=BASE_FONT_SEG)
        ax_all.set_xlim(float(np.min(turns)) - 0.5, float(np.max(turns)) + 0.5)
        ax_all.grid(True, alpha=0.3, linestyle="--")
        ax_all.tick_params(axis="both", labelsize=BASE_FONT_SEG)
        ax_all.set_title("")

        # Merge legends (lines + optional bars); entry 재현 선은 범례에서 제외, 괄호 설명 제거
        handles_all, labels_all = ax_all.get_legend_handles_labels()
        if ax_counts is not None:
            handles_c, labels_c = ax_counts.get_legend_handles_labels()
            handles_all += handles_c
            labels_all += labels_c
        filt = [
            (h, strip_parenthetical(lb))
            for h, lb in zip(handles_all, labels_all)
            if not is_recall_entry_legend_label(lb)
        ]
        line_pairs: list[tuple] = []
        bar_pairs: list[tuple] = []
        for h, lb in filt:
            sl = lb.strip().lower()
            if sl in ("generated", "removed"):
                bar_pairs.append((h, lb))
            else:
                line_pairs.append((h, lb))

        if len(line_pairs) <= 3:
            # MiraiBotnet처럼 공격 종류가 적은 경우: 범례를 한 줄로 단순 표시
            leg_handles = [h for h, _ in line_pairs] + [h for h, _ in bar_pairs]
            leg_labels = [lb for _, lb in line_pairs] + [lb for _, lb in bar_pairs]
            leg_ncol = max(1, len(leg_handles))
            leg_nrows = 1
        else:
            leg_handles, leg_labels, leg_ncol, leg_nrows = build_attackwise_combined_legend_grid(
                line_pairs, bar_pairs, ncol_max=10
            )
        # 범례 행 수가 많을수록 하단 여백 확대
        if leg_nrows <= 3:
            bottom_margin = 0.32
        elif leg_nrows == 4:
            bottom_margin = 0.40
        elif leg_nrows == 5:
            bottom_margin = 0.46
        else:
            bottom_margin = 0.52
        # twinx: 패드 가로를 조금 넓히고 좌우를 맞춰 가운데에 가깝게
        if ax_counts is not None:
            fig.subplots_adjust(left=0.078, right=0.905, top=0.97, bottom=bottom_margin)
        else:
            fig.subplots_adjust(left=0.09, right=0.93, top=0.97, bottom=bottom_margin)
        fig.legend(
            leg_handles,
            leg_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, bottom_margin - 0.085),
            ncol=leg_ncol,
            frameon=True,
            fontsize=BASE_FONT_SEG,
            columnspacing=0.65,
            handlelength=1.4,
            handletextpad=0.40,
            borderpad=0.35,
        )

        combined_filename = f"{file_name}{turn_range_str}_attack_recall_all.png"
        combined_path = output_dir / combined_filename
        try:
            plt.savefig(combined_path, format="png", dpi=300, bbox_inches="tight")
            print(f"Attack-wise combined graph saved to: {combined_path}")
        except Exception as e:
            print(f"Failed to save combined attack-wise graph: {e}")
        plt.close()

        # 2) Optional: one graph per attack type
        if args.per_attack:
            for attack in attack_types:
                entry_col = entry_cols.get(attack)
                exit_col = exit_cols.get(attack)

                entry_vals = history_df[entry_col].values if entry_col else np.full(len(history_df), np.nan)
                exit_vals = history_df[exit_col].values if exit_col else np.full(len(history_df), np.nan)

                plt.figure(figsize=(18, 11))
                ax = plt.gca()

                # Default: exit only. Optional: include entry via --include_entry
                ax.plot(turns, exit_vals, "-", color="purple", linewidth=4, label="Exit")
                if args.include_entry:
                    ax.plot(turns, entry_vals, "--", color="blue", linewidth=3, alpha=0.7, label="Entry")

                ax.set_xlabel("Period", fontsize=BASE_FONT_SEG)
                ax.set_ylabel("Attack Coverage", fontsize=BASE_FONT_SEG)
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.tick_params(axis="both", labelsize=BASE_FONT_SEG)
                ax.set_title("")

                plt.subplots_adjust(top=0.92, bottom=0.22)
                h1, l1 = ax.get_legend_handles_labels()
                filt = [
                    (h, strip_parenthetical(lb))
                    for h, lb in zip(h1, l1)
                    if not is_recall_entry_legend_label(lb)
                ]
                h1 = [p[0] for p in filt]
                l1 = [p[1] for p in filt]
                ax.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=BASE_FONT_SEG)
                plt.tight_layout()

                graph_filename = f"{file_name}{turn_range_str}_attack_recall_{attack}.png"
                graph_path = output_dir / graph_filename
                try:
                    plt.savefig(graph_path, format="png", dpi=300, bbox_inches="tight")
                    print(f"Attack-wise graph saved to: {graph_path}")
                except Exception as e:
                    print(f"Failed to save attack-wise graph for {attack}: {e}")
                plt.close()

            print(f"Generated {len(attack_types)} per-attack graphs in: {output_dir}")
        continue
        
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
                        ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, 
                                     label='Generated', color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                        ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, 
                                     label='Removed', color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                    else:
                        ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, 
                                     color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                        ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, 
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
                    ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, 
                                 label='Generated', color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                    ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, 
                                 label='Removed', color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
                else:
                    ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, 
                                 color='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1)
                    ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, 
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
            'F1-Score (Adaptation)', 'Accuracy (Adaptation)', 'Generated',
            'Removed'
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

