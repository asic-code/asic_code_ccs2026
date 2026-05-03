import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# 폰트 설정 (Times New Roman)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 40  # 기본 폰트 크기 (2배)

# Argument parser
parser = argparse.ArgumentParser(description='Generate conditions impact graph from mapping evaluation CSVs')
parser.add_argument(
    '--add_netml',
    action='store_true',
    help='Include netML dataset folder if present',
)
parser.add_argument(
    '--exclude_darpa98',
    action='store_true',
    help='Exclude DARPA98 dataset from plotting and save as *_nodarpa98.png',
)
args = parser.parse_args()

# 허용된 데이터셋 목록 (backup 폴더 제외)
# DARPA98, MiraiBotnet, Kitsune, NSL-KDD, CICIDS2017, CICIoT2023
allowed_datasets = ['DARPA98', 'MiraiBotnet', 'Kitsune', 'NSL-KDD', 'CICIDS2017', 'CICIoT2023']
if args.add_netml:
    allowed_datasets.append('netML')
if args.exclude_darpa98 and 'DARPA98' in allowed_datasets:
    allowed_datasets.remove('DARPA98')
# 현재 사용 가능한 데이터셋 (자동으로 로드됨)
# CICIDS2017, CICIoT2023은 나중에 추가될 예정 (주석 처리할 필요 없음, 자동으로 제외됨)
datasets = []

# 색상 매핑 (Figure 4와 유사하게)
colors = {
    'CICIDS2017': '#1f77b4',      # Blue
    'CICIoT2023': '#ff7f0e',      # Orange
    'DARPA98': '#2ca02c',         # Green
    'MiraiBotnet': '#d62728',     # Red (Mirai)
    'Kitsune': '#9467bd',         # Purple
    'NSL-KDD': '#8c564b',         # Brown
    'netML': '#17becf'            # Cyan
}

# 데이터 로드
base_path = Path(r'C:\ASIC_excute\Dataset\conditions_evaluation')
data_dict = {}

# 사용 가능한 모든 데이터셋 폴더 찾기
for dataset_folder in base_path.iterdir():
    dataset_name = dataset_folder.name
    # backup 폴더 제외, 허용된 데이터셋만, netML과 Time_Logs 제외
    if (dataset_folder.is_dir() and 
        '_backup' not in dataset_name and  # backup 폴더 제외
        dataset_name in allowed_datasets and  # 허용된 데이터셋만
        dataset_name not in (['Time_Logs'] if args.add_netml else ['netML', 'Time_Logs'])):
        # CSV 파일 찾기 (mapping_evaluation으로 끝나는 파일)
        csv_files = list(dataset_folder.glob('*mapping_evaluation*.csv'))
        
        if csv_files:
            # 첫 번째 매칭 파일 사용
            csv_file = csv_files[0]
            try:
                df = pd.read_csv(csv_file)
                # 컬럼명 정리 (공백 제거)
                df.columns = df.columns.str.strip()
                data_dict[dataset_name] = df
                datasets.append(dataset_name)
                print(f"Loaded {dataset_name}: {csv_file}")
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
        else:
            print(f"CSV file not found for {dataset_name} in {dataset_folder}")

# 데이터셋 이름 정렬 (일관된 순서) - 범례/플롯 순서 고정
# 요청 순서: DARPA98, NSL-KDD, CICIDS2017, Kitsune, Mirai, CICIoT2023 (+ netML 옵션)
dataset_order = ['DARPA98', 'NSL-KDD', 'CICIDS2017', 'Kitsune', 'MiraiBotnet', 'CICIoT2023']
if args.add_netml:
    dataset_order.append('netML')
if args.exclude_darpa98:
    dataset_order = [d for d in dataset_order if d != 'DARPA98']
datasets = [d for d in dataset_order if d in datasets] + [d for d in datasets if d not in dataset_order]

print(f"\nDatasets to plot: {datasets}")

# 그래프 생성 (이중 축)
fig, ax1 = plt.subplots(figsize=(18, 11))  # 폭을 16에서 20으로 증가

# 막대 그래프용 축 (왼쪽) - signature_count (비율로 표시)
ax1.set_xlabel('Condition Count', fontsize=40)
ax1.set_ylabel('Signature Count (Normalized)', fontsize=40, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=40)
# y축 제목을 y- 방향(아래)으로 이동
ax1.yaxis.set_label_coords(-0.03, 0.42)  # 두 번째 값을 줄이면 아래로 이동 (0.5 -> 0.3)

# 선 그래프용 축 (오른쪽) - precision
ax2 = ax1.twinx()
ax2.set_ylabel('Total Precision', fontsize=40, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=40)

# 모든 데이터셋의 condition_count 값 수집
all_conditions = set()
for dataset in datasets:
    if dataset in data_dict:
        all_conditions.update(data_dict[dataset]['condition_count'].values)
x_positions = np.array(sorted(all_conditions))

# x축을 동간격으로 설정 (인덱스 기반)
x_indices = np.arange(len(x_positions))  # 0, 1, 2, 3, ...

bar_width = 0.4  # 데이터셋 간 간격 (크기 비율에 맞게 조정)
n_datasets = len(datasets)

# 각 데이터셋에 대해 플롯
for idx, dataset in enumerate(datasets):
    if dataset not in data_dict:
        continue
    
    df = data_dict[dataset]
    
    # 필요한 컬럼 확인
    required_cols = ['condition_count', 'total_precision', 'signature_count']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing columns in {dataset}. Available: {df.columns.tolist()}")
        continue
    
    # 데이터 정렬
    df_sorted = df.sort_values('condition_count')
    conditions = df_sorted['condition_count'].values
    precision = df_sorted['total_precision'].values
    signature_count = df_sorted['signature_count'].values
    
    # 시그니처 개수를 비율로 정규화 (첫 번째 값 기준)
    if len(signature_count) > 0 and signature_count[0] > 0:
        signature_normalized = signature_count / signature_count[0]
    else:
        signature_normalized = signature_count
    
    # x축 위치 계산 (막대 그래프를 위해) - 동간격 인덱스 사용 (간격 확대)
    x_spacing = 2.8  # 포인트 간 간격 배율 (막대 그래프 묶음 간격 증가)
    x_pos = []
    for cond in conditions:
        cond_idx = np.where(x_positions == cond)[0]
        if len(cond_idx) > 0:
            # 동간격 인덱스를 사용하고 offset 추가 (간격 확대 반영)
            offset = (idx - n_datasets / 2 + 0.5) * bar_width
            x_pos.append(x_indices[cond_idx[0]] * x_spacing + offset)
        else:
            # 조건이 없으면 인덱스 기반으로 계산
            offset = (idx - n_datasets / 2 + 0.5) * bar_width
            x_pos.append(len(x_indices) * x_spacing + offset)
    x_pos = np.array(x_pos)
    
    color = colors.get(dataset, f'C{idx}')
    
    # 막대 그래프: signature_count 정규화된 값 (왼쪽 축)
    ax1.bar(x_pos, signature_normalized, bar_width, 
            color=color, alpha=0.6, zorder=1, edgecolor='white', linewidth=1)
    
    # condition_count를 동간격 인덱스로 변환 (간격 확대 반영)
    x_spacing = 2.8  # 포인트 간 간격 배율 (막대 그래프 묶음 간격 증가)
    condition_indices = []
    for cond in conditions:
        cond_idx = np.where(x_positions == cond)[0]
        if len(cond_idx) > 0:
            condition_indices.append(x_indices[cond_idx[0]] * x_spacing)
        else:
            condition_indices.append(len(x_indices) * x_spacing)
    condition_indices = np.array(condition_indices)
    
    # 선 그래프: total_precision (오른쪽 축) - 흰색 테두리 효과로 더 잘 보이게
    # 먼저 흰색 배경 선 그리기 (크기 비율 유지)
    ax2.plot(condition_indices, precision, color='white', marker='o', markersize=18, 
             linewidth=10, linestyle='-', zorder=2, alpha=0.9)
    # 그 위에 원래 색상의 선 그리기
    ax2.plot(condition_indices, precision, color=color, marker='o', markersize=14, 
             linewidth=6, linestyle='-', zorder=3)

# x축 설정 (동간격 인덱스 사용, 간격을 넓게)
# 포인트 간 간격을 늘리기 위해 인덱스를 2.8배로 확장
x_spacing = 2.8  # 포인트 간 간격 배율 (막대 그래프 묶음 간격 증가)
x_indices_spaced = x_indices * x_spacing

# 마지막 막대그래프 묶음의 오른쪽 끝 계산
# 마지막 condition count의 인덱스
last_idx = len(x_indices) - 1
# 최대 offset (가장 오른쪽 막대)
max_offset = (n_datasets / 2 - 0.5) * bar_width
# 마지막 막대그래프 묶음의 오른쪽 끝 (오른쪽 y축에 딱 붙이기)
right_end = last_idx * x_spacing + max_offset + bar_width + 0.01  # 오른쪽 여백

# 첫 번째 막대그래프 묶음의 왼쪽 끝 계산
# 첫 번째 condition count의 인덱스는 0
min_offset = (-n_datasets / 2 + 0.5) * bar_width
# 첫 번째 막대그래프 묶음의 왼쪽 끝 = 첫 번째 막대의 왼쪽 끝
# 첫 번째 막대의 중심은 0 * x_spacing + min_offset = min_offset
# 첫 번째 막대의 왼쪽 끝은 min_offset - bar_width/2
left_end_of_first_bar = min_offset - bar_width/2
# 왼쪽 시작점: 첫 번째 막대의 왼쪽 끝에서 약간 왼쪽으로 (오른쪽 여백과 비슷하게)
left_start = left_end_of_first_bar + 0.01  # 오른쪽 여백과 동일한 0.01

ax1.set_xlim(left_start, right_end)
ax1.set_xticks(x_indices_spaced)
ax1.set_xticklabels(x_positions, fontsize=40)  # 라벨은 실제 condition_count 값
ax1.tick_params(axis='x', labelsize=40)

# y축 범위 설정
# signature_normalized 최대값 확인하여 위쪽 여백 추가
all_signature_max = []
for dataset in datasets:
    if dataset in data_dict:
        df = data_dict[dataset]
        if 'signature_count' in df.columns:
            signature_count = df['signature_count'].values
            if len(signature_count) > 0 and signature_count[0] > 0:
                signature_normalized = signature_count / signature_count[0]
                all_signature_max.append(max(signature_normalized))

if all_signature_max:
    max_sig = max(all_signature_max)
    ax1.set_ylim(0, max_sig * 1.1)  # 최대값의 10% 여유 공간 추가
else:
    ax1.set_ylim(0, None)  # 자동으로 최대값에 맞춤

# 꺾은선 그래프가 잘 보이도록 y축 범위 조정
# precision의 최소값과 최대값 확인
all_precision = []
for dataset in datasets:
    if dataset in data_dict:
        df = data_dict[dataset]
        if 'total_precision' in df.columns:
            all_precision.extend(df['total_precision'].values)

if all_precision:
    min_val = min(all_precision)
    max_val = max(all_precision)
    range_val = max_val - min_val
    y_min = max(0, min_val - range_val * 0.05)
    y_max = min(1.0, max_val + range_val * 0.05)
    ax2.set_ylim(y_min, y_max)
else:
    ax2.set_ylim(0.5, 1.0)  # 기본값

# 그리드
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

# 범례 생성
from matplotlib.lines import Line2D

# 데이터셋 이름 변환 (MiraiBotnet -> Mirai)
def format_dataset_name(dataset):
    if dataset == 'MiraiBotnet':
        return 'Mirai'
    return dataset

dataset_legend_elements = []
ordered_datasets = ['DARPA98', 'NSL-KDD', 'CICIDS2017', 'Kitsune', 'MiraiBotnet', 'CICIoT2023']
if args.add_netml:
    ordered_datasets.append('netML')
for dataset in ordered_datasets:
    if dataset in data_dict and dataset in datasets:
        color = colors.get(dataset, 'gray')
        display_name = format_dataset_name(dataset)
        dataset_legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                alpha=0.6,
                label=display_name,
                edgecolor='white',
                linewidth=1,
            )
        )

# Total Precision 범례 요소 (데이터셋 뒤에 배치)
precision_line = Line2D(
    [0],
    [0],
    color='black',
    marker='o',
    markersize=7,
    linewidth=3,
    linestyle='-',
    label='Total Precision',
)

# 범례 가로(좌→우) 표시 순서 고정
# Matplotlib legend는 기본적으로 column-major로 채우므로,
# 원하는 row-major(가로) 순서가 되도록 handles 배열을 재배치한다.
label_to_elem = {elem.get_label(): elem for elem in dataset_legend_elements}
darpa = label_to_elem.get('DARPA98')
nslkdd = label_to_elem.get('NSL-KDD')
cicids = label_to_elem.get('CICIDS2017')
kitsune = label_to_elem.get('Kitsune')
mirai = label_to_elem.get('Mirai')
ciciot = label_to_elem.get('CICIoT2023')
netml = label_to_elem.get('netML')

# 빈 항목(마지막 칸 채우기용)
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

# 범례를 아래쪽에 배치 (x축 제목과 범례 공간 확보)
plt.subplots_adjust(top=0.94, bottom=0.22)

legend_handles = [h for h in dataset_legend_elements if h is not None]
legend_handles.append(precision_line)

legend = ax1.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.18),
    ncol=min(3, max(1, len(legend_handles))),
    frameon=True,
    fontsize=40,
)

plt.tight_layout()
suffix = '_addnetML' if args.add_netml else ''
if args.exclude_darpa98:
    suffix += '_nodarpa98'
out_name = f'conditions_impact_graph{suffix}.png'
plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"\nGraph saved as '{out_name}'")
plt.close()

