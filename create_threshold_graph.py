import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path

# 폰트 설정 (Times New Roman)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  # 기본 폰트 크기 (2배)

# Argument parser
parser = argparse.ArgumentParser(description='Generate threshold impact graphs from Kmeans_threshold*.csv')
parser.add_argument(
    '--add_netml',
    action='store_true',
    help='Include netML dataset folder and save as *_addnetML.png',
)
parser.add_argument(
    '--exclude_darpa98',
    action='store_true',
    help='Exclude DARPA98 dataset and save as *_nodarpa98.png',
)
args = parser.parse_args()

# 데이터셋 목록 - 범례/플롯 순서 고정
# 요청 순서: DARPA98, NSL-KDD, CICIDS2017, Kitsune, Mirai, CICIoT2023 (+ netML 옵션)
# (코드 내부 명칭: MiraiBotnet -> 표시명 Mirai)
datasets = ['DARPA98', 'NSL-KDD', 'CICIDS2017', 'Kitsune', 'MiraiBotnet', 'CICIoT2023']
if args.add_netml:
    datasets.append('netML')
if args.exclude_darpa98:
    datasets = [d for d in datasets if d != 'DARPA98']

# 색상 매핑 (그래프와 유사하게)
colors = {
    'CICIDS2017': '#1f77b4',      # Blue
    'CICIoT2023': '#2ca02c',      # Green
    'DARPA98': '#8c564b',         # Brown
    'NSL-KDD': '#d62728',         # Red
    'MiraiBotnet': '#9467bd',     # Purple
    'Kitsune': '#e377c2',         # Pink
    'netML': '#17becf'            # Cyan
}

# 마커 스타일
markers = {
    'CICIDS2017': 'o',
    'CICIoT2023': 'o',
    'DARPA98': 'o',
    'NSL-KDD': 'o',
    'MiraiBotnet': 'o',
    'Kitsune': 'o',
    'netML': 'o'
}

def pick_threshold_csv(csv_files: list[Path]) -> Path:
    """
    폴더에 다음처럼 2개가 있을 때 기본 파일만 선택:
      - Kmeans_thresholds.csv
      - Kmeans_thresholds_before260423.csv
    우선순위:
      1) 정확히 'Kmeans_thresholds.csv'
      2) 'before'가 포함되지 않은 파일(정렬 후 첫 번째)
      3) 정렬 후 첫 번째
    """
    by_name = {p.name.lower(): p for p in csv_files}
    preferred = by_name.get('kmeans_thresholds.csv')
    if preferred:
        return preferred

    non_before = [p for p in csv_files if 'before' not in p.name.lower()]
    if non_before:
        return sorted(non_before, key=lambda p: p.name.lower())[0]
    return sorted(csv_files, key=lambda p: p.name.lower())[0]

# 데이터 로드
base_path = Path(r'C:\ASIC_excute\Dataset\threshold_evaluation')
data_dict = {}

for dataset in datasets:
    dataset_path = base_path / dataset
    # CSV 파일 찾기 (Kmeans_threshold로 시작하는 파일)
    csv_files = list(dataset_path.glob('Kmeans_threshold*.csv'))
    
    if csv_files:
        csv_file = pick_threshold_csv(csv_files)
        try:
            df = pd.read_csv(csv_file)
            # 컬럼명 정리 (공백 제거)
            df.columns = df.columns.str.strip()
            data_dict[dataset] = df
            print(f"Loaded {dataset}: {csv_file}")
        except Exception as e:
            print(f"Error loading {dataset}: {e}")
    else:
        print(f"CSV file not found for {dataset} in {dataset_path}")

# 그래프 생성
# 요구사항:
# - netML로 범례 1줄이 늘어나는 만큼 "그림 자체" 높이만 늘림
# - 차트(축) 영역의 세로 길이(인치)는 기존(16x11, top=0.95, bottom=0.14)과 동일하게 유지
base_fig_h = 11
top_margin = 0.95
base_bottom = 0.14
base_axes_height_in = base_fig_h * (top_margin - base_bottom)

fig_h = 12.5 if args.add_netml else base_fig_h
bottom_margin = base_bottom
if args.add_netml:
    # axes_height_in = fig_h * (top - bottom)  => bottom = top - axes_height_in/fig_h
    bottom_margin = top_margin - (base_axes_height_in / fig_h)

plt.figure(figsize=(16, fig_h))
ax = plt.gca()

# 각 데이터셋에 대해 플롯
for dataset in datasets:
    if dataset not in data_dict:
        continue
    
    df = data_dict[dataset]
    
    # 필요한 컬럼 확인
    if 'threshold' not in df.columns or 'jaccard' not in df.columns:
        print(f"Missing columns in {dataset}. Available: {df.columns.tolist()}")
        continue
    
    # 데이터 정렬
    df_sorted = df.sort_values('threshold')
    thresholds = df_sorted['threshold'].values
    jaccard = df_sorted['jaccard'].values
    
    # fp, fn 데이터 가져오기
    fp = df_sorted['fp'].values if 'fp' in df_sorted.columns else None
    fn = df_sorted['fn'].values if 'fn' in df_sorted.columns else None
    
    # 기본 라인 플롯
    color = colors[dataset]
    marker = markers[dataset]
    
    # 라인 플롯
    # 범례에 표시할 이름 (MiraiBotnet -> Mirai)
    display_name = 'Mirai' if dataset == 'MiraiBotnet' else dataset
    line = ax.plot(thresholds, jaccard, color=color, marker=marker, 
                   markersize=8, linewidth=2, label=display_name, zorder=2)

# 배경색 영역 계산 (양쪽 각각 3번째 점까지)
# 모든 데이터셋의 threshold 값 수집
all_thresholds = []
for dataset in datasets:
    if dataset in data_dict:
        df = data_dict[dataset]
        if 'threshold' in df.columns:
            df_sorted = df.sort_values('threshold')
            all_thresholds.extend(df_sorted['threshold'].values[:3])  # 왼쪽 3개
            all_thresholds.extend(df_sorted['threshold'].values[-3:])  # 오른쪽 3개

if all_thresholds:
    # 왼쪽 영역: 0부터 0.2까지 (파란색 배경)
    left_max = 0.2
    # 오른쪽 영역: 오른쪽 3번째 점부터 1.0까지의 최소값
    right_min = min([t for t in all_thresholds if t > 0.5]) if any(t > 0.5 for t in all_thresholds) else 1.0
    
    # 왼쪽 배경 (fn dominant - 파란색 계열)
    ax.axvspan(0.0, left_max, alpha=0.15, color='#4444ff', zorder=0)
    # 오른쪽 배경 (fp dominant - 빨간색 계열)
    ax.axvspan(right_min, 1.0, alpha=0.15, color='#ff4444', zorder=0)

# 그래프 설정
ax.set_xlabel('Normality Threshold', fontsize=40)
ax.set_ylabel('Jaccard Index', fontsize=40)

# 범위 설정
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.5, 1.0)

# 눈금 설정
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(axis='both', labelsize=40)
# y축 0.3 값이 x축과 겹치지 않도록 조정 (두 번째 그래프에서만 필요)

# 그리드
ax.grid(True, alpha=0.3, linestyle='--')

# 위쪽 여백 제거, 아래쪽 여백 추가
plt.subplots_adjust(top=top_margin, bottom=bottom_margin)

# 범례 (하단에 배치, 2줄로, 그래프와 겹치지 않게)
handles, labels = ax.get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))

# 범례 가로(좌→우) 표시 순서 고정
# 원하는 표시(가로) 순서(2행, 3열):
#   1행: DARPA98, NSL-KDD, CICIDS2017
#   2행: Kitsune, Mirai, CICIoT2023
# Matplotlib legend는 기본적으로 column-major로 채우므로,
# row-major(가로) 순서가 되도록 handles 배열을 재배치한다.
empty_patch = plt.Rectangle(
    (0, 0),
    0.001,
    0.001,
    facecolor='white',
    edgecolor='white',
    alpha=0.0,
    linewidth=0,
    label=' ',
)

if args.add_netml and 'netML' in label_to_handle:
    # 3열(ncol=3) 유지 + 3번째 줄에 netML:
    # 표시(가로):
    #   1행: DARPA98, NSL-KDD, CICIDS2017
    #   2행: Kitsune, Mirai, CICIoT2023
    #   3행: (blank), netML, (blank)  <-- 요청사항 (가운데)
    # column-major 핸들 배열:
    #   col1=[DARPA98, Kitsune, blank]
    #   col2=[NSL-KDD, Mirai, netML]
    #   col3=[CICIDS2017, CICIoT2023, blank]
    legend_handles = [
        label_to_handle.get('DARPA98'),
        label_to_handle.get('Kitsune'),
        empty_patch,
        label_to_handle.get('NSL-KDD'),
        label_to_handle.get('Mirai'),
        label_to_handle.get('netML'),
        label_to_handle.get('CICIDS2017'),
        label_to_handle.get('CICIoT2023'),
        empty_patch,
    ]
else:
    legend_handles = [
        label_to_handle.get('DARPA98'),
        label_to_handle.get('Kitsune'),
        label_to_handle.get('NSL-KDD'),
        label_to_handle.get('Mirai'),
        label_to_handle.get('CICIDS2017'),
        label_to_handle.get('CICIoT2023'),
    ]
legend_handles = [h for h in legend_handles if h is not None]
ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    fontsize=40,
)
plt.tight_layout()
suffix = '_addnetML' if args.add_netml else ''
if args.exclude_darpa98:
    suffix += '_nodarpa98'
plt.savefig(f"threshold_impact_graph{suffix}.png", dpi=300, bbox_inches='tight')
print(f"\nGraph saved as 'threshold_impact_graph{suffix}.png'")
plt.close()

# 두 번째 그래프: y축 범위 확장 버전 (_2)
plt.figure(figsize=(16, fig_h))
ax2 = plt.gca()

# 각 데이터셋에 대해 플롯
for dataset in datasets:
    if dataset not in data_dict:
        continue
    
    df = data_dict[dataset]
    
    # 필요한 컬럼 확인
    if 'threshold' not in df.columns or 'jaccard' not in df.columns:
        continue
    
    # 데이터 정렬
    df_sorted = df.sort_values('threshold')
    thresholds = df_sorted['threshold'].values
    jaccard = df_sorted['jaccard'].values
    
    # fp, fn 데이터 가져오기
    fp = df_sorted['fp'].values if 'fp' in df_sorted.columns else None
    fn = df_sorted['fn'].values if 'fn' in df_sorted.columns else None
    
    # 기본 라인 플롯
    color = colors[dataset]
    marker = markers[dataset]
    
    # 라인 플롯
    # 범례에 표시할 이름 (MiraiBotnet -> Mirai)
    display_name = 'Mirai' if dataset == 'MiraiBotnet' else dataset
    ax2.plot(thresholds, jaccard, color=color, marker=marker, 
             markersize=8, linewidth=2, label=display_name, zorder=2)

# 배경색 영역 계산 (양쪽 각각 3번째 점까지)
# 모든 데이터셋의 threshold 값 수집
all_thresholds_2 = []
for dataset in datasets:
    if dataset in data_dict:
        df = data_dict[dataset]
        if 'threshold' in df.columns:
            df_sorted = df.sort_values('threshold')
            all_thresholds_2.extend(df_sorted['threshold'].values[:3])  # 왼쪽 3개
            all_thresholds_2.extend(df_sorted['threshold'].values[-3:])  # 오른쪽 3개

if all_thresholds_2:
    # 왼쪽 영역: 0부터 0.2까지 (파란색 배경)
    left_max_2 = 0.2
    # 오른쪽 영역: 오른쪽 3번째 점부터 1.0까지의 최소값
    right_min_2 = min([t for t in all_thresholds_2 if t > 0.5]) if any(t > 0.5 for t in all_thresholds_2) else 1.0
    
    # 왼쪽 배경 (fn dominant - 파란색 계열)
    ax2.axvspan(0.0, left_max_2, alpha=0.15, color='#4444ff', zorder=0)
    # 오른쪽 배경 (fp dominant - 빨간색 계열)
    ax2.axvspan(right_min_2, 1.0, alpha=0.15, color='#ff4444', zorder=0)

# 그래프 설정
ax2.set_xlabel('Normality Threshold', fontsize=40)
ax2.set_ylabel('Jaccard Index', fontsize=40)

# 범위 설정 (y축 확장)
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.3, 1.0)  # 0.5에서 0.3으로 확장

# 눈금 설정
ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax2.tick_params(axis='both', labelsize=40)
# y축 0.3 값이 x축과 겹치지 않도록 조정 (0.3 라벨만 위로 올리기)
for label in ax2.get_yticklabels():
    if label.get_text() == '0.3':
        # 라벨 위치를 위로 조정
        label.set_verticalalignment('bottom')
        label.set_y(label.get_position()[1] + 0.015)

# 그리드
ax2.grid(True, alpha=0.3, linestyle='--')

# 위쪽 여백 제거, 아래쪽 여백 추가
plt.subplots_adjust(top=top_margin, bottom=bottom_margin)

# 범례 (하단에 배치, 2줄로, 그래프와 겹치지 않게)
handles2, labels2 = ax2.get_legend_handles_labels()
label_to_handle2 = dict(zip(labels2, handles2))

legend_handles2 = [
    label_to_handle2.get('DARPA98'),
    label_to_handle2.get('Kitsune'),
    label_to_handle2.get('NSL-KDD'),
    label_to_handle2.get('Mirai'),
    label_to_handle2.get('CICIDS2017'),
    label_to_handle2.get('CICIoT2023'),
]

empty_patch2 = plt.Rectangle(
    (0, 0),
    0.001,
    0.001,
    facecolor='white',
    edgecolor='white',
    alpha=0.0,
    linewidth=0,
    label=' ',
)

if args.add_netml and 'netML' in label_to_handle2:
    legend_handles2 = [
        label_to_handle2.get('DARPA98'),
        label_to_handle2.get('Kitsune'),
        empty_patch2,
        label_to_handle2.get('NSL-KDD'),
        label_to_handle2.get('Mirai'),
        label_to_handle2.get('netML'),
        label_to_handle2.get('CICIDS2017'),
        label_to_handle2.get('CICIoT2023'),
        empty_patch2,
    ]
legend_handles2 = [h for h in legend_handles2 if h is not None]
ax2.legend(
    handles=legend_handles2,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    fontsize=40,
)
plt.tight_layout()
plt.savefig(f"threshold_impact_graph_2{suffix}.png", dpi=300, bbox_inches='tight')
print(f"Graph saved as 'threshold_impact_graph_2{suffix}.png'")
plt.close()

