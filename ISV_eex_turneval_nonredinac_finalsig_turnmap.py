import pandas as pd
import argparse
import time
import os
import sys
from datetime import datetime
#import copy # --- copy module for deepcopy ---
import numpy as np # --- numpy for direct mapping ---
import re # --- re for robust interval parsing ---
import signal # --- Import signal to handle BrokenPipeError gracefully ---
import shutil # --- Import for directory deletion ---
import gc  # --- Import gc to allow explicit garbage collection ---

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
# Suppress Numba bytecode/interpreter DEBUG flood when turn_bin_conditions loads @njit code
logging.getLogger("numba").setLevel(logging.WARNING)


LEVEL_LIMITS_BY_FILE_TYPE = {
    'MiraiBotnet': 5,
    'NSL-KDD': 7,
    'NSL_KDD': 7,
    'DARPA98': 10,
    'DARPA': 10,
    'CICIDS2017': 7,
    'CICIDS': 7,
    'CICModbus23': None,
    'CICModbus': None,
    'IoTID20': None,
    'IoTID': None,
    'CICIoT': 3,
    'CICIoT2023': 3,
    'netML': 3,
    'Kitsune': 3,
    'default': None
}

# === START: Project Root Path Correction ===
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if '..' not in sys.path:
        sys.path.insert(0, '..')
# === END: Project Root Path Correction ===

from Dataset_Choose_Rule.dtype_optimize import load_csv_safely
from utils.remove_rare_columns import remove_rare_columns # Import the function
from Heterogeneous_Method.Feature_Encoding import Heterogeneous_Feature_named_featrues # --- Import for protection list ---
from Dataset_Choose_Rule.isv_save_log import get_params_str # --- Import for logging (itemset, rule saving) ---
from Dataset_Choose_Rule.isv_checkpoint_handler import save_checkpoint, load_checkpoint # --- Import for checkpointing ---
from Dataset_Choose_Rule.dataset_window_preset import get_temporal_chunk_size
from Modules.Signature_Organize import organize_signatures # --- Import for signature pruning ---
from utils.chunk_cache import ChunkCache, _yield_batches_from_source
from utils.rule_spooler import RuleSpooler
from utils.isv_filtering import (
    resolve_rule_spool_settings,
    calculate_and_log_support_stats,
    calculate_support_for_itemset,
)
from utils.support_dealing import get_dominant_columns, build_rules_from_dominant
from utils.class_row import get_label_columns_to_exclude
from utils.auto_support import AutoSupportController
from utils.separability import compute_feature_separability, log_separability_summary
from utils.isv_rule_filtering_runtime import filter_rule_batch_runtime
from utils.signature_reporting import extract_attack_types_for_signature_from_cache
from Rebuild_Method.fp_validation_runtime_ISV import validate_fp_subset_runtime
from utils.fp_contribution_reporting import build_fp_contribution_rows_by_creation_turn
from Rebuild_Method.fp_rule_priority_reporting import build_rule_phase_stats_by_signature
from Rebuild_Method.fp_rule_priority_chunked import generate_fp_priority_reports_from_csv
from utils.separation_feature_filter import (
    load_separation_feature_pool,
    load_turn_separation_feature_pools,
    add_separation_filter_args,
    apply_separation_cli_postprocess,
)
from Signature_tool.signature_reduction import reduce_signatures_by_subsets
from utils.metrics_by_turn import compute_metrics_by_turn_range, calculate_metrics_from_alerts
from utils.precision_underlimit_runtime import (
    init_precision_underlimit_state,
    build_cluster_based_sig_metrics,
    select_signatures_by_precision_underlimit,
    cleanup_precision_underlimit_state,
    run_precision_underlimit_auto_adjust,
)
from utils.attackwise_turn_recall import (
    _sanitize_attack_name_for_column,
    build_attack_index_map_full_dataset,
    build_attack_index_map_by_turn_full_dataset,
    compute_attackwise_recall_turn_locked_cumulative,
)
try:
    from Evaluation.calculate_signature import calculate_signature
except ImportError:
    logger.warning("Could not import calculate_signature. Individual signature metrics will not be available.")
    def calculate_signature(data, signatures):
        return []

try:
    from Dataset_Choose_Rule.association_data_choose import get_clustered_data_path # MODIFIED
    from Dataset_Choose_Rule.choose_amount_dataset import file_cut
    from definition.Anomal_Judgment import anomal_judgment_label, anomal_judgment_nonlabel
    from utils.time_transfer import time_scalar_transfer
    from Modules.Heterogeneous_module import choose_heterogeneous_method
    from Heterogeneous_Method.separate_group_mapping import map_intervals_to_groups, flush_interval_mapping_debug
    from Heterogeneous_Method.turn_bin_conditions import (
        apply_range_signatures_to_dataset,
        preprocess_and_map_chunk_turn,
        _normalize_rule_conditions,
        _rule_looks_range,
        _patch_range_matchers,
        set_range_match_engine,
        set_range_match_jit,
        set_use_rule_eval_c,
        try_extract_attack_types_c_ext,
    )
    from Heterogeneous_Method.reverse_mapping import build_reverse_mapping, reverse_map_rule, build_original_value_range_mapping, reverse_map_rule_with_fallback
    from Modules.Association_module import association_module
    from Modules.Signature_underlimit import under_limit
    from Modules.Difference_sets import dict_list_difference
    from Rebuild_Method.FalsePositive_Check import apply_signatures_to_dataset, evaluate_false_positives, summarize_fp_results
    from Rebuild_Method.fp_diagnostics import run_fp_diagnostics
except ImportError as e:
    print(f"Warning: Could not import all project modules: {e}. Some functionalities might be limited.")
    def association_module(df, *args, **kwargs):
        print("WARNING: Using dummy 'association_module'.")
        if not df.empty:
            rule = {col: val for col, val in df.iloc[0].items() if pd.notna(val)}
            return [rule] if rule else []
        return []
    def apply_signatures_to_dataset(df, sigs):
        print("WARNING: Using dummy 'apply_signatures_to_dataset'.")
        return pd.DataFrame()
    def under_limit(signature_dict, count, precision_lim):
        print("WARNING: Using dummy 'under_limit'. No filtering will be applied.")
        return signature_dict
    def load_csv_safely(file_type, path):
        print("WARNING: Using dummy 'load_csv_safely'. Attempting pd.read_csv directly.")
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            return None
    def set_use_rule_eval_c(use):
        pass

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.set_loglevel("warning")
except ImportError:
    print("Warning: matplotlib is not installed. Plotting functionality will be disabled.")
    plt = None


# --- Logger Setup ---
logger = logging.getLogger(__name__)
# MODIFIED: Set level to DEBUG to see detailed logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def _rule_has_nan(rule: dict) -> bool:
    """Return True if any value in rule is NaN/NA (np.nan, pd.NA, None treated as NaN by pd.isna)."""
    try:
        return any(pd.isna(v) for v in rule.values())
    except Exception:
        # Be conservative: if rule values are odd types, don't crash the pipeline.
        return False

def _suppress_broken_pipe_tracebacks():
    """Prevent BrokenPipeError tracebacks from noisy worker exits."""
    def _handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, BrokenPipeError):
            return
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.excepthook = _handler

def log_dataframe_debug_info(df, name="DataFrame"):
    """Logs detailed debug information about a DataFrame."""
    if df is None or df.empty:
        logger.debug(f"--- {name} Info: DataFrame is empty or None. ---")
        return
        
    logger.debug(f"--- START: {name} Info ---")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Columns: {df.columns.tolist()}")
    logger.debug(f"Data Types:\n{df.dtypes.to_string()}")
    logger.debug(f"Head:\n{df.head().to_string()}")
    logger.debug(f"--- END: {name} Info ---")

def get_attack_columns_for_report(file_type: str):
    """
    Return attack/label columns used ONLY for reporting attack types.
    These columns must NOT be used for rule generation.
    """
    if file_type in ['CICIDS2017']:
        return ['Label']
    if file_type in ['CICIoT2023', 'CICIoT']:
        return ['attack_name']
    if file_type in ['DARPA98', 'DARPA']:
        return ['Class']
    if file_type in ['MiraiBotnet']:
        return ['reconnaissance', 'infection', 'action']
    if file_type in ['netML']:
        return ['Label']
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        return ['class']
    return []


def calculate_performance_in_batches(
    data_source,
    signatures,
    batch_size,
    label_col="label",
    show_apply_progress=True,
    show_apply_info=True,
    show_batch_progress=False,
):
    """
    Calculates performance metrics by streaming batches from the provided data source.
    Returns alert results to identify which signatures triggered alerts.
    """
    if not signatures or data_source is None:
        return 0, 0, 0, 0, 0, 0, pd.DataFrame() # tp, fp, recall, precision, f1, accuracy, alerts_df

    all_alerted_indices = set()
    actual_positives_indices = set()
    actual_negatives_indices = set()
    all_alerts = []  # Collect all alerts to track which signatures triggered them

    total_rows = 0
    batch_counter = 0

    #for batch_df in _yield_batches_from_source(data_source, batch_size):
    def _apply_signatures(batch_df):
        try:
            return apply_signatures_to_dataset(
                batch_df,
                signatures,
                show_progress=show_apply_progress,
                log_info=show_apply_info,
            )
        except TypeError:
            return apply_signatures_to_dataset(batch_df, signatures)

    batch_iter = _yield_batches_from_source(data_source, batch_size)
    if show_batch_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(batch_iter, desc="  [Final Eval] Applying signatures (batches)")
        except Exception:
            pass

    for batch_df in batch_iter:
        batch_counter += 1
        total_rows += len(batch_df)

        #alerts_batch = apply_signatures_to_dataset(batch_df, signatures)
        alerts_batch = _apply_signatures(batch_df)
        if not alerts_batch.empty:
            all_alerted_indices.update(alerts_batch['alert_index'].unique())
            all_alerts.append(alerts_batch)

        if label_col in batch_df.columns:
            actual_positives_indices.update(batch_df[batch_df[label_col] == 1].index)
            actual_negatives_indices.update(batch_df[batch_df[label_col] == 0].index)

    if total_rows == 0:
        return 0, 0, 0, 0, 0, 0, pd.DataFrame()

    logger.info(f"  [Perf Eval] Processed {total_rows} rows across {batch_counter} streamed batch(es).")

    # Combine all alerts
    alerts_df = pd.concat(all_alerts, ignore_index=True) if all_alerts else pd.DataFrame()

    tp = len(all_alerted_indices.intersection(actual_positives_indices))
    fp = len(all_alerted_indices.intersection(actual_negatives_indices))
    total_negatives = len(actual_negatives_indices)
    tn = total_negatives - fp
    
    total_anomalies = len(actual_positives_indices)
    total_alerts = len(all_alerted_indices)

    recall = tp / total_anomalies if total_anomalies > 0 else 0
    precision = tp / total_alerts if total_alerts > 0 else 0
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / total_rows if total_rows > 0 else 0
        
    return tp, fp, recall, precision, f1, accuracy, alerts_df


# Use range-aware matcher for this script
apply_signatures_to_dataset = apply_range_signatures_to_dataset


# --- START: New Helper function for parallel filtering ---
# This global tuple will hold the data needed by the worker processes.
# It's a workaround to avoid passing large dataframes to each process,
# which can be slow due to serialization (pickling).

def preprocess_and_map_chunk(chunk_df, file_type, category_mapping, data_list, n_splits=None, debug=False, debug_context=None, value_rank_map=None):
    """
    Applies robust labeling and then maps a data chunk using map_intervals_to_groups.
    NOTE: Time-based feature conversion should be done BEFORE calling this function.
    
    IMPORTANT: This function uses existing_mapping to ensure that intervals are created
    consistently across all chunks using the same category_mapping from the full dataset scan.
    """
    # MODIFIED: The labeling logic is removed. The loaded data already has 'label' and 'cluster'.
    # This function now uses map_intervals_to_groups for proper interval mapping.
    
    # Use existing_mapping to ensure consistent interval creation across chunks
    # This prevents each chunk from creating different intervals based on its own data distribution
    chunk_embedded, _, _, chunk_data_list = choose_heterogeneous_method(
        chunk_df,
        file_type, 'Interval_inverse', 'N', 
        n_splits_override=n_splits,
        existing_mapping=category_mapping  # Use the category_mapping from full dataset scan
    )
    
    # For interval (numeric) columns, ensure we map using raw numeric values
    # to align with value_rank_map and avoid re-mapping already binned values.
    if isinstance(category_mapping, dict) and isinstance(category_mapping.get('interval'), pd.DataFrame):
        interval_cols = [c for c in category_mapping['interval'].columns if c in chunk_df.columns]
    else:
        interval_cols = []
    chunk_for_mapping = chunk_embedded.copy()
    if interval_cols:
        for c in interval_cols:
            if c in chunk_df.columns:
                chunk_for_mapping[c] = chunk_df[c].values

    # Use map_intervals_to_groups to properly map intervals to group numbers
    # This uses the category_mapping from the initial full-dataset scan to ensure consistency
    mapped_chunk, _ = map_intervals_to_groups(
        chunk_for_mapping,
        category_mapping,
        chunk_data_list,
        'N',
        debug=debug,
        debug_context=debug_context,
        value_rank_map=value_rank_map
    )
    
    # Preserve label and cluster columns from original chunk
    if 'label' in chunk_df.columns:
        mapped_chunk['label'] = chunk_df['label'].values
    if 'cluster' in chunk_df.columns:
        mapped_chunk['cluster'] = chunk_df['cluster'].values
    if 'adjusted_cluster' in chunk_df.columns:
        mapped_chunk['adjusted_cluster'] = chunk_df['adjusted_cluster'].values

    # Preserve attack-type columns for reporting only (not used in rule generation)
    attack_cols = get_attack_columns_for_report(file_type)
    for col in attack_cols:
        if col in chunk_df.columns:
            mapped_chunk[col] = chunk_df[col].values
    
    return mapped_chunk

def main(args):
    # --- NEW: Top-level try-except block to catch the root cause of process failures ---
    try:
        set_range_match_engine(getattr(args, "range_match_engine", "bool"))
        set_range_match_jit(getattr(args, "range_match_jit", "none"))
        set_use_rule_eval_c(not getattr(args, "no_rule_eval_c", False))
        _patch_range_matchers()
        start_time = time.time()
        logger.info("--- Initial Setup: Generating Mapping On-the-fly ---")

        rule_spool_chunk_size_runtime, rule_spool_force_flush_threshold = resolve_rule_spool_settings(args.file_type, args.rule_spool_chunk_size)
        logger.info(f"[RuleSpool] chunk_size={rule_spool_chunk_size_runtime}, flush_threshold={rule_spool_force_flush_threshold}")

        negative_filtering_enabled = bool(args.negative_filtering)
        strict_normal_zero = bool(getattr(args, "strict_normal_zero", False))
        negative_filter_threshold = max(0.0, args.negative_filter_threshold)
        normal_min_support = args.normal_min_support if args.normal_min_support is not None else args.min_support
        auto_support_controller = None
        if args.auto_support:
            auto_support_controller = AutoSupportController(
                base_support=args.min_support,
                target_min=args.auto_support_target_min,
                target_max=args.auto_support_target_max,
                step=args.auto_support_step,
                min_support=args.auto_support_min,
                max_support=args.auto_support_max,
            )
            logger.info(
                f"[AutoSupport] Enabled: target={args.auto_support_target_min}-{args.auto_support_target_max}, "
                f"step={args.auto_support_step}, bounds=({args.auto_support_min}, {args.auto_support_max})."
            )
            if args.dynamic_support:
                logger.info("[AutoSupport] dynamic_support is enabled; internal adjustments may override base support.")
        if strict_normal_zero:
            logger.info("[NormalFilter] Strict mode enabled: rules must have ZERO matches in normal data.")
        if negative_filtering_enabled:
            if negative_filter_threshold >= normal_min_support:
                logger.warning(f"[NegativeFilter] Threshold ({negative_filter_threshold:.4f}) >= normal_min_support ({normal_min_support:.4f}). Consider lowering for stricter filtering.")
            logger.info(f"[NegativeFilter] Enabled with threshold {negative_filter_threshold:.4f}.")

        # --- Temporal window preset for chunk_size ---
        if args.cstemporal:
            preset_cs = get_temporal_chunk_size(args.file_type)
            if preset_cs:
                logger.info(f"[TemporalWindow] Using preset chunk_size={preset_cs} for {args.file_type}")
                args.chunk_size = preset_cs
                args.chunk_size_label = "tem"
            else:
                logger.warning(f"[TemporalWindow] No preset found for {args.file_type}; keeping chunk_size={args.chunk_size}")
                args.chunk_size_label = None
        
        # --- NEW: Get max_level for filename ---
        max_level = LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
        if getattr(args, "max_level", None) is not None:
            max_level = args.max_level

        # --- Build turn range tag for filename ---
        start_label = args.run_turn_start if args.run_turn_start is not None else "NA"
        end_label = args.run_turn_end if args.run_turn_end is not None else "NA"
        turn_range_tag = f"ts{start_label}-te{end_label}"

        # --- Parameter and Path Setup ---
        params_str_base = get_params_str(args, max_level)
        rulemake_suffix = "_rulemakelabel" if getattr(args, "rule_make_label", False) else ""
        eval_suffix = "_testcluster" if getattr(args, "eval_target", "label") == "cluster" else ""
        dom_parts = []
        if args.dominant_min_support is not None:
            dom_parts.append(f"ms={args.dominant_min_support}")
        if args.dominant_min_confidence is not None:
            dom_parts.append(f"mc={args.dominant_min_confidence}")
        if args.dominant_normal_min_support is not None:
            dom_parts.append(f"nms={args.dominant_normal_min_support}")
        if args.dominant_level is not None:
            dom_parts.append(f"lvl={args.dominant_level}")
        if getattr(args, "dominant_attach_filter", False):
            dom_parts.append(f"da={args.dominant_attach_threshold}")
        dom_suffix = ""
        if dom_parts:
            dom_suffix = "_dom(" + "_".join(dom_parts) + ")"
        pul_suffix = ""
        if args.precision_underlimit is not None:
            if bool(int(getattr(args, "precision_underlimit_auto_adjust", 0))):
                pul_suffix = f"_pulauto{args.precision_underlimit}"
            else:
                pul_suffix = f"_pul{args.precision_underlimit}"
        sep_suffix = ""
        if getattr(args, "use_separation_feature_filter", False):
            sep_mode = "turn" if getattr(args, "use_separation_turn_dynamic", False) else "global"
            sep_suffix = f"_sepF({sep_mode})"
        params_str = f"{params_str_base}_n{args.n_splits}_dom{args.dominant_freq_threshold}{dom_suffix}{pul_suffix}{rulemake_suffix}{eval_suffix}{sep_suffix}_turneval_tr{turn_range_tag}"
        if getattr(args, "save_attackwise_turn_recall", False):
            params_str = f"{params_str}_attackwise"
        run_dir = os.path.join("../Dataset_ISV_turnmap", args.file_type, params_str) # Used for artifacts and checkpoints
        chunk_cache_dir = os.path.join(run_dir, "chunk_cache")
        chunk_cache = ChunkCache(chunk_cache_dir)
        turn_separability_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_turn_separability.csv")
        turn_fp_events_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_turn_fp_events.csv")
        turn_diag_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_turn_diagnostics.csv")
        turn_fp_contrib_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_turn_fp_contribution_by_creation_turn.csv")
        turn_fp_rule_phase_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_turn_fp_rule_phase_stats.csv")
        fp_top_cohort_rule_details_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_fp_top_creation_turn_rule_details.csv")
        fp_rule_delete_priority_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_fp_rule_delete_priority.csv")
        fp_rule_delete_turn_impact_csv = os.path.join(run_dir, f"{args.file_type}_{args.file_number}_fp_rule_delete_priority_turn_impact.csv")

        # Ensure run_dir exists (checkpoint load can be a no-op if missing)
        os.makedirs(run_dir, exist_ok=True)

        # --- Optional: feature separation based mining filter (default OFF) ---
        separation_feature_pool = load_separation_feature_pool(args)
        separation_turn_feature_pools = load_turn_separation_feature_pools(args)
        if getattr(args, "use_separation_feature_filter", False):
            logger.info(
                f"[FeatFilter] enabled mode={'turn' if getattr(args, 'use_separation_turn_dynamic', True) else 'global'} "
                f"global={0 if separation_feature_pool is None else len(separation_feature_pool)} "
                f"turns={len(separation_turn_feature_pools)}"
            )
        
        # --- Reset Logic ---
        if args.reset:
            if os.path.isdir(run_dir):
                logger.info(f"--reset flag is set. Deleting directory: {run_dir}")
                try:
                    shutil.rmtree(run_dir)
                    logger.info(f"Successfully deleted directory.")
                except OSError as e:
                    logger.error(f"Error deleting directory {run_dir}: {e}")
            else:
                logger.info(f"--reset flag is set, but the target directory does not exist: {run_dir}")
            
            # Exit the script gracefully after handling the reset operation.
            sys.exit(0)
        
        # --- Load Checkpoint ---
        resume_from_turn = 0
        all_valid_signatures = {}
        sleeping_signatures = {}
        signatures_to_remove = set()
        history = []
        signature_turn_created = {}
        signature_turn_removed = {}
        fp_event_records = []
        # Turn-by-turn NaN-in-signature summary (for debugging/reporting only)
        nan_sig_turn_stats = []
        # Precision-underlimit temporal state (in-memory for this run)
        precision_underlimit_state = init_precision_underlimit_state()
        
        checkpoint_data = load_checkpoint(run_dir)
        if checkpoint_data:
            resume_from_turn = checkpoint_data['resume_from_turn']
            all_valid_signatures = checkpoint_data['all_valid_signatures']
            sleeping_signatures = checkpoint_data.get('sleeping_signatures', {})
            signatures_to_remove = checkpoint_data['signatures_to_remove']
            history = checkpoint_data['history']
            signature_turn_created = checkpoint_data.get('signature_turn_created', {})
            signature_turn_removed = checkpoint_data.get('signature_turn_removed', {})

        # --- Data Loading (TURNMAP) ---
        file_path, total_rows = get_clustered_data_path(args.file_type, args.file_number)
        logger.info("Turnmap mode: skipping full-dataset scan/mapping; per-turn mapping is used.")
        category_mapping = None
        mapping_features = None
        mapping_features_original = None
        interval_rules_df_full = None
        value_rank_map = {}
        # Legacy commented block (old full-dataset mapping path) moved to:
        # docs/comment_archives/ISV_eex_turneval_nonredinac_finalsig_turnmap_L490-L963.txt
        # This data_list is for the old hunter/prey logic, which is now simplified.
        # We keep it for preprocess_and_map_chunk's signature but it's not used for state.
        data_list = [pd.DataFrame(), pd.DataFrame()]

        # --- State Re-creation and Fast-Forward ---
        #processed_data_so_far = pd.DataFrame()
        full_eval_cache = None
        attack_to_indices_full = {}
        attack_to_indices_by_turn_full = {}
        attack_totals_full = {}
        attackwise_entry_captured_state = {}
        attackwise_exit_captured_state = {}
        if getattr(args, "save_attackwise_turn_recall", False):
            full_eval_cache_dir = os.path.join(run_dir, "full_eval_cache")
            full_eval_cache = ChunkCache(full_eval_cache_dir)
            logger.info("[Attackwise] Building full-dataset cache for attack-wise recall evaluation...")
            full_iter = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
            full_turn = 0
            for full_chunk in full_iter:
                full_turn += 1
                full_processed = time_scalar_transfer(full_chunk, args.file_type)
                full_eval_cache.register_chunk(full_processed, full_turn)
            logger.info(f"[Attackwise] Full cache ready: rows={full_eval_cache.total_rows}")
            attack_to_indices_full, attack_totals_full = build_attack_index_map_full_dataset(
                full_eval_cache,
                args.file_type,
                batch_size=args.evaluation_batch_size,
            )
            attack_to_indices_by_turn_full = build_attack_index_map_by_turn_full_dataset(
                full_eval_cache,
                args.file_type,
                batch_size=args.evaluation_batch_size,
            )
            logger.info(
                f"[Attackwise] Attack types={len(attack_totals_full)} -> "
                f"{ {k: attack_totals_full[k] for k in list(sorted(attack_totals_full.keys()))[:10]} }"
            )

        main_chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
        
        if resume_from_turn > 0:
            logger.info(f"Fast-forwarding to turn {resume_from_turn + 1}. Rebuilding cumulative data...")
            ff_turn_counter = 0
            for chunk in main_chunk_iterator:
                ff_turn_counter += 1
                if ff_turn_counter > resume_from_turn:
                    # We've reached the chunk where we need to resume processing.
                    # Break the loop so the main loop can start with this chunk.
                    # To do this, we need to restructure the loop.
                    break 

                # Apply the necessary preprocessing to rebuild the streamed cache state
                processed_chunk = time_scalar_transfer(chunk, args.file_type)
                mapped_chunk, _turn_mapping, _turn_data_list, _turn_mapping_features, _turn_mapping_original = preprocess_and_map_chunk_turn(
                    processed_chunk,
                    args.file_type,
                    args.n_splits,
                    debug=True,
                    debug_context={"phase": "fast_forward", "turn": ff_turn_counter},
                )
                # Store raw (processed) chunk for range-based evaluation
                chunk_cache.register_chunk(processed_chunk, ff_turn_counter)
            
            logger.info(f"Fast-forward complete. {chunk_cache.total_rows} rows of prior data rebuilt.")
        
        # --- NEW: Initial Pruning after Checkpoint Load ---
        if args.signature_organize and all_valid_signatures and not chunk_cache.is_empty():
            logger.info("--- Performing Initial Signature Organization after loading checkpoint ---")
            
            all_valid_signatures = organize_signatures(
                all_signatures=all_valid_signatures,
                data_provider=chunk_cache,
                data_batch_size=args.evaluation_batch_size,
                num_processes=args.num_processes,
                run_dir=run_dir,
                turn_counter=0, # Use 0 for the initial organization step
                coverage_threshold=args.prune_coverage_threshold,
                enable_merging=args.merge_signatures,
                merge_infrequent_threshold=args.merge_infrequent_threshold
            )
            
            # Update the blacklist to remove any pruned signatures that were also blacklisted
            blacklisted_and_pruned = signatures_to_remove - set(all_valid_signatures.keys())
            if blacklisted_and_pruned:
                signatures_to_remove = signatures_to_remove.intersection(set(all_valid_signatures.keys()))
                logger.info(f"Removed {len(blacklisted_and_pruned)} signatures from the blacklist as they were pruned.")

        # --- Main Loop ---
        # Re-initialize iterator to start from the beginning for the main loop logic
        main_chunk_iterator = pd.read_csv(file_path, chunksize=args.chunk_size, low_memory=False)
        turn_counter = 0

        for chunk in main_chunk_iterator:
            turn_counter += 1
            logger.info(f"[TurnLoop] Enter turn {turn_counter} with {len(chunk)} rows.")

            # --- TURN RANGE CONTROL (for stopping) ---
            if args.run_turn_end is not None and turn_counter > args.run_turn_end:
                logger.info(f"Reached run_turn_end={args.run_turn_end}. Stop experiment.")
                break
            
            # --- Preprocessing for the Current Chunk (ALWAYS for state) ---
            processed_chunk = time_scalar_transfer(chunk, args.file_type)
            mapped_chunk, turn_category_mapping, turn_data_list, turn_mapping_features, turn_mapping_original = preprocess_and_map_chunk_turn(
                processed_chunk,
                args.file_type,
                args.n_splits,
                debug=True,
                debug_context={"phase": "main", "turn": turn_counter},
            )

            # Persist the raw (processed) chunk for range-based evaluations
            chunk_cache.register_chunk(processed_chunk, turn_counter)
            
            # --- TURN RANGE CONTROL (for processing) ---
            # Skip actual processing if outside turn range, but keep data in cache for state
            if args.run_turn_start is not None and turn_counter < args.run_turn_start:
                logger.info(f"[TurnLoop] Skip processing for turn {turn_counter} (before run_turn_start={args.run_turn_start}).")
                continue

            # --- 1. Skip or Process (checkpoint resume) ---
            if turn_counter <= resume_from_turn:
                # Rebuild state (already done in fast-forward, but need to concat the first chunk if turn 1 is skipped)
                logger.info(f"[TurnLoop] Skip processing for turn {turn_counter} (<= resume_from_turn={resume_from_turn}).")
                continue # Skip processing for turns already completed.

            # The loop starts from the next turn to process
            logger.info(f"--- Processing Turn {turn_counter} (Rows {(turn_counter-1)*args.chunk_size + 1} - {turn_counter*args.chunk_size}) ---")
            # NEW: keep an in-memory view for per-turn entry/exit evaluation (raw values)
            turn_eval_data = processed_chunk.copy()
            eval_label_col = "cluster" if getattr(args, "eval_target", "label") == "cluster" else "label"
            if eval_label_col not in turn_eval_data.columns:
                logger.warning(f"[EvalTarget] '{eval_label_col}' not found; falling back to 'label'.")
                eval_label_col = "label"

            # Per-turn reverse mappings for converting rules to original value ranges
            turn_reverse_mapping = build_reverse_mapping(turn_category_mapping)
            turn_original_value_range_mapping = build_original_value_range_mapping(
                turn_category_mapping,
                turn_mapping_features,
                turn_mapping_original,
                prefer_interval_bounds=True,
                mapped_df=mapped_chunk
            )
            # Compact reverse-mapping check: confirm value-range mapping built and sample
            n_rev = len(turn_original_value_range_mapping)
            rev_sample = ""
            if n_rev:
                fc = next(iter(turn_original_value_range_mapping))
                gr = turn_original_value_range_mapping[fc]
                one = next(iter(gr.items()), (None, "")) if gr else (None, "")
                rev_sample = f", sample {fc} group{one[0]}->{str(one[1])[:40]}"
            logger.info(f"[RevMap] Turn {turn_counter}: value-range mapping for {n_rev} features{rev_sample}")

            if turn_counter == 1:
                log_dataframe_debug_info(mapped_chunk, "First Chunk (After Mapping)")

            #processed_data_so_far = pd.concat([processed_data_so_far, mapped_chunk], ignore_index=True)
            
            # ... (The rest of the main loop logic for processing `mapped_chunk` remains the same) ...
            # [This includes blacklist reset, performance evaluation, validation, hunter, generation, etc.]
            # --- Blacklist Reset Logic ---
            if turn_counter > 1 and (turn_counter - 1) % 2 == 0:
                if signatures_to_remove:
                    logger.warning(f"*** Resetting blacklist at the start of Turn {turn_counter}. "
                                   f"Removing {len(signatures_to_remove)} signatures from the blacklist. ***")
                    signatures_to_remove.clear()
                else:
                    logger.info(f"*** Blacklist reset point at Turn {turn_counter}, but it was already empty. ***")

            # --- 0. (NEW) Entry Performance Evaluation ---
            entry_alerts = pd.DataFrame() # FIX: Initialize to prevent UnboundLocalError
            entry_recall, entry_precision, entry_f1, entry_accuracy = 0, 0, 0, 0
            entry_cluster_recall, entry_cluster_precision, entry_cluster_f1, entry_cluster_accuracy = 0, 0, 0, 0
            rules_at_turn_start = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
            entry_sig_count = len(rules_at_turn_start)
            # Actual active set at entry (recording-only removals like inactive/reduction excluded).
            active_actual_ids_at_entry = set(rules_at_turn_start.keys())
            # Quick diagnostics: rule keys missing in current chunk, and NaN-in-rule ratio
            if rules_at_turn_start:
                chunk_cols = set(turn_eval_data.columns) if turn_eval_data is not None else set()
                missing_key_rules = 0
                nan_rules = 0
                nan_rule_samples = []
                nan_keys_counter = {}
                for _sid, _rule in rules_at_turn_start.items():
                    if isinstance(_rule, dict):
                        if any(k not in chunk_cols for k in _rule.keys()):
                            missing_key_rules += 1
                        if _rule_has_nan(_rule):
                            nan_rules += 1
                            nan_keys = [k for k, v in _rule.items() if pd.isna(v)]
                            for k in nan_keys:
                                nan_keys_counter[k] = nan_keys_counter.get(k, 0) + 1
                            if len(nan_rule_samples) < 5:
                                nan_rule_samples.append({"id": _sid, "rule": _rule, "nan_keys": nan_keys})
                total_rules = len(rules_at_turn_start)
                missing_ratio = (missing_key_rules / total_rules) if total_rules > 0 else 0.0
                nan_ratio = (nan_rules / total_rules) if total_rules > 0 else 0.0
                logger.info(f"[RuleCheck] Turn {turn_counter}: missing-key rules {missing_key_rules}/{total_rules} ({missing_ratio:.2%})")
                logger.info(f"[RuleCheck] Turn {turn_counter}: NaN-in-rule {nan_rules}/{total_rules} ({nan_ratio:.2%})")
                if nan_rule_samples:
                    logger.info(f"[RuleCheck] Turn {turn_counter}: NaN rule samples (up to 5):")
                    for rec in nan_rule_samples:
                        logger.info(f"  - sig_id={rec['id']}, nan_keys={rec['nan_keys']}, rule={rec['rule']}")
                if nan_keys_counter:
                    top_nan_keys = sorted(nan_keys_counter.items(), key=lambda x: x[1], reverse=True)[:10]
                    logger.info(f"[RuleCheck] Turn {turn_counter}: NaN keys top10={top_nan_keys}")
            else:
                logger.info(f"[RuleCheck] Turn {turn_counter}: no rules to check.")
            if rules_at_turn_start and turn_eval_data is not None and not turn_eval_data.empty:
                # Sort by signature ID to ensure consistent ordering (for "first match wins" logic)
                sorted_entry_sigs = sorted(rules_at_turn_start.items(), key=lambda x: x[0])
                formatted_entry_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in sorted_entry_sigs]
                
                # --- MODIFIED: Use batched performance calculation ---
                _, _, entry_recall, entry_precision, entry_f1, entry_accuracy, entry_alerts = calculate_performance_in_batches(
                    turn_eval_data,
                    formatted_entry_sigs,
                    args.evaluation_batch_size,
                    label_col=eval_label_col,
                )
                # Reference-only cluster evaluation from existing alerts (no extra matching, no leakage into generation).
                if getattr(args, "eval_target", "label") != "cluster" and "cluster" in turn_eval_data.columns:
                    entry_cluster_recall, entry_cluster_precision, entry_cluster_f1, entry_cluster_accuracy = calculate_metrics_from_alerts(
                        entry_alerts, turn_eval_data, "cluster"
                    )
                
                logger.info(f"Turn {turn_counter} ENTRY Performance - Recall: {entry_recall:.4f}, Precision: {entry_precision:.4f}, F1: {entry_f1:.4f}, Accuracy: {entry_accuracy:.4f}")
            else:
                formatted_entry_sigs = []

            entry_attack_recall_full = {}
            if getattr(args, "save_attackwise_turn_recall", False) and full_eval_cache is not None:
                if formatted_entry_sigs:
                    _, _, _, _, _, _, entry_alerts_full = calculate_performance_in_batches(
                        full_eval_cache,
                        formatted_entry_sigs,
                        args.evaluation_batch_size,
                        label_col=eval_label_col,
                        show_apply_progress=False,
                        show_apply_info=False,
                        show_batch_progress=False,
                    )
                else:
                    entry_alerts_full = pd.DataFrame()
                entry_attack_recall_full, attackwise_entry_captured_state = compute_attackwise_recall_turn_locked_cumulative(
                    entry_alerts_full,
                    attack_to_indices_by_turn_full,
                    attack_totals_full,
                    turn_counter=turn_counter,
                    captured_state=attackwise_entry_captured_state,
                )

            # Split chunk data based on chosen rule-making target (label or cluster)
            label_cols_to_exclude = get_label_columns_to_exclude(args.file_type)
            rule_target_col = "label" if getattr(args, "rule_make_label", False) else "cluster"
            if rule_target_col not in mapped_chunk.columns:
                logger.warning(f"[RuleMake] '{rule_target_col}' missing. Falling back to 'cluster' split.")
                rule_target_col = "cluster"
            # Mapped data for mining/filtering
            normal_data_in_chunk = mapped_chunk[mapped_chunk[rule_target_col] == 0].copy().drop(columns=label_cols_to_exclude, errors='ignore')
            anomalous_data_in_chunk = mapped_chunk[mapped_chunk[rule_target_col] == 1].copy().drop(columns=label_cols_to_exclude, errors='ignore')
            # Raw data for evaluation/FP (range matching)
            normal_data_raw = processed_chunk[processed_chunk[rule_target_col] == 0].copy()
            anomalous_data_raw = processed_chunk[processed_chunk[rule_target_col] == 1].copy()

            # Separability diagnostics (cluster-based, no label leakage)
            if args.separability:
                try:
                    attack_cols = get_attack_columns_for_report(args.file_type)
                    exclude_cols = list(set(label_cols_to_exclude + attack_cols))
                    sep_results = compute_feature_separability(
                        mapped_chunk,
                        label_col="cluster",
                        exclude_cols=exclude_cols,
                        sample_size=args.separability_sample_size,
                        top_n=args.separability_top_n,
                        max_unique=args.separability_max_unique,
                    )
                    log_separability_summary(sep_results, turn_counter=turn_counter)
                    if sep_results:
                        sep_df = pd.DataFrame(
                            [
                                {
                                    "turn": int(turn_counter),
                                    **rec,
                                }
                                for rec in sep_results
                            ]
                        )
                        sep_df.to_csv(
                            turn_separability_csv,
                            mode="a",
                            index=False,
                            header=not os.path.exists(turn_separability_csv),
                            encoding='utf-8-sig',
                        )
                except Exception as e:
                    logger.warning(f"[Separability] Turn {turn_counter}: failed to compute. Error: {e}")

            # Debug: per-turn NaN distribution in mapped data (focus on anomaly subset)
            try:
                if not anomalous_data_in_chunk.empty:
                    nan_counts = anomalous_data_in_chunk.isna().sum()
                    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
                    if not nan_counts.empty:
                        top_nan_cols = nan_counts.head(10).to_dict()
                        logger.info(f"[MapNaN] Turn {turn_counter}: anomaly NaN cols top10={top_nan_cols}")
                        # For top NaN columns, compare original values vs value_rank_map coverage
                        if value_rank_map and processed_chunk is not None and isinstance(interval_rules_df_full, pd.DataFrame):
                            for col in list(top_nan_cols.keys())[:5]:
                                if col in processed_chunk.columns and col in mapped_chunk.columns:
                                    try:
                                        orig_vals = pd.to_numeric(processed_chunk[col], errors='coerce').fillna(0)
                                        nan_mask = mapped_chunk[col].isna()
                                        if nan_mask.any():
                                            nan_vals = orig_vals[nan_mask]
                                            sample_vals = nan_vals.dropna().unique()[:5].tolist()
                                            col_rank_map = value_rank_map.get(col, {})
                                            missing_in_map = [v for v in sample_vals if v not in col_rank_map]
                                            # Rank unmapped count for this column
                                            rank_unmapped_count = int(orig_vals.map(col_rank_map).isna().sum()) if col_rank_map else 0
                                            logger.info(
                                                f"[MapNaN] Turn {turn_counter}: col={col}, "
                                                f"nan_rows={int(nan_mask.sum())}, sample_vals={sample_vals}, "
                                                f"missing_in_rank_map={missing_in_map}, "
                                                f"rank_unmapped_count={rank_unmapped_count}"
                                            )
                                            # Check if ranks are covered by interval rules
                                            try:
                                                if col not in interval_rules_df_full.columns:
                                                    logger.info(f"[MapNaN] Turn {turn_counter}: col={col} no interval rules in full mapping.")
                                                    continue
                                                rules = interval_rules_df_full[col].dropna().astype(str).tolist()
                                                intervals = []
                                                for rule_str in rules:
                                                    interval_part = rule_str.split('=')[0].strip()
                                                    left_bracket = interval_part[0]
                                                    right_bracket = interval_part[-1]
                                                    nums = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', interval_part)]
                                                    if len(nums) != 2:
                                                        continue
                                                    closed = 'neither'
                                                    if left_bracket == '[' and right_bracket == ']':
                                                        closed = 'both'
                                                    elif left_bracket == '[':
                                                        closed = 'left'
                                                    elif right_bracket == ']':
                                                        closed = 'right'
                                                    intervals.append(pd.Interval(nums[0], nums[1], closed=closed))
                                                if intervals:
                                                    interval_index = pd.IntervalIndex(intervals)
                                                    overlapping = bool(interval_index.is_overlapping)
                                                    sample_ranks = [col_rank_map.get(v) for v in sample_vals if v in col_rank_map]
                                                    uncovered = [r for r in sample_ranks if r is not None and interval_index.get_indexer([r])[0] == -1]
                                                    logger.info(
                                                        f"[MapNaN] Turn {turn_counter}: col={col}, "
                                                        f"sample_ranks={sample_ranks}, uncovered_ranks={uncovered}, "
                                                        f"overlapping_intervals={overlapping}"
                                                    )
                                            except Exception as e:
                                                logger.warning(f"[MapNaN] Turn {turn_counter}: col={col} rank coverage check failed: {e}")
                                    except Exception as e:
                                        logger.warning(f"[MapNaN] Turn {turn_counter}: col={col} detail failed: {e}")
            except Exception as e:
                logger.warning(f"[MapNaN] Turn {turn_counter}: failed to summarize NaN columns. Error: {e}")
            # Release the mapped chunk reference early to reduce peak memory per turn
            del mapped_chunk
            gc.collect()

            # --- 1. Validation Step (FP Detection) ---
            newly_removed_count = 0
            underlimit_removed_count = 0
            removed_ids_this_turn = set()
            no_alert_signature_ids_this_turn = set()
            newly_flagged_for_removal = set() # Store IDs of signatures removed THIS turn
            if not args.disable_fp_removal and all_valid_signatures and not normal_data_raw.empty:
                pre_removed, pre_flagged = validate_fp_subset_runtime(
                    signatures_subset=rules_at_turn_start,
                    turn_counter=turn_counter,
                    phase_tag="pre",
                    rules_at_turn_start_ref=rules_at_turn_start,
                    normal_data_raw=normal_data_raw,
                    anomalous_data_raw=anomalous_data_raw,
                    turn_eval_data=turn_eval_data,
                    args=args,
                    run_dir=run_dir,
                    signatures_to_remove=signatures_to_remove,
                    fp_event_records=fp_event_records,
                    turn_fp_events_csv=turn_fp_events_csv,
                    apply_signatures_to_dataset_fn=apply_signatures_to_dataset,
                    evaluate_false_positives_fn=evaluate_false_positives,
                    summarize_fp_results_fn=summarize_fp_results,
                    run_fp_diagnostics_fn=run_fp_diagnostics,
                )
                newly_removed_count += pre_removed
                newly_flagged_for_removal.update(pre_flagged)
                removed_ids_this_turn.update(set(pre_flagged))
            elif args.disable_fp_removal:
                logger.info(f"[FP Removal] FP removal is disabled. Skipping validation step.")

            rule_spooler = RuleSpooler(run_dir, turn_counter, chunk_size=rule_spool_chunk_size_runtime)
            new_signatures_found = 0
            newly_added_sig_ids_this_turn = set()
            try:
                # --- 2. Generation Step (Spooling) ---
                if not anomalous_data_in_chunk.empty:
                    logger.info(f"Generating new candidate rules from {len(anomalous_data_in_chunk)} anomalous data rows...")
                    
                    # Identify dominant (near-constant) columns and mask them for support counting
                    # NOTE: anomalous_data_in_chunk already has label columns excluded at line 526
                    dominant_cols = {}
                    mask_skipped = False
                    if args.mask_dominant_cols:
                        dominant_cols = get_dominant_columns(anomalous_data_in_chunk, freq_threshold=args.dominant_freq_threshold)
                        if dominant_cols:
                            logger.info(f"[DominantCols] Masking {len(dominant_cols)} near-constant columns for support counting: {list(dominant_cols.keys())}")
                            # Log dominant column frequency in anomaly vs normal for diagnostics
                            try:
                                dom_stats = []
                                for d_col, d_val in list(dominant_cols.items())[:10]:
                                    anom_freq = float((anomalous_data_in_chunk[d_col] == d_val).mean()) if d_col in anomalous_data_in_chunk.columns else None
                                    norm_freq = float((normal_data_in_chunk[d_col] == d_val).mean()) if d_col in normal_data_in_chunk.columns else None
                                    dom_stats.append((d_col, d_val, anom_freq, norm_freq))
                                logger.info(f"[DominantCols] Top10 freq (col, val, anom, norm): {dom_stats}")
                            except Exception as e:
                                logger.warning(f"[DominantCols] Failed to compute freq stats: {e}")

                    anomalous_for_mining = anomalous_data_in_chunk.drop(columns=dominant_cols.keys(), errors='ignore') if dominant_cols else anomalous_data_in_chunk
                    # label_cols_to_exclude already defined above (line 523)
                    
                    # Guard: if masking removed everything (or nearly everything), skip masking
                    # NOTE: anomalous_data_in_chunk and anomalous_for_mining already have label columns excluded
                    if args.mask_dominant_cols and dominant_cols and anomalous_for_mining.shape[1] == 0:
                        # Check if all columns are truly constant (support=1). If yes, keep masking.
                        cols_for_check = list(anomalous_data_in_chunk.columns)
                        all_constant = True
                        for c in cols_for_check:
                            if anomalous_data_in_chunk[c].nunique(dropna=False) > 1:
                                all_constant = False
                                break
                        if all_constant:
                            logger.warning("[DominantCols] All features are constant (support=1); keeping masking despite empty remainder.")
                        else:
                            logger.warning("[DominantCols] Masking removed all feature columns; skipping masking for this turn.")
                            dominant_cols = {}
                            anomalous_for_mining = anomalous_data_in_chunk
                            mask_skipped = True
                        '''
                        else:
                            logger.warning("[DominantCols] All features masked for Kitsune; keeping masking (no skip).")
                        '''

                    # Optional feature filter for better benign<->attack separability
                    # (no label leakage: uses precomputed CSV scores only)
                    active_feature_pool = separation_feature_pool
                    pool_mode = "global"
                    if separation_turn_feature_pools:
                        turn_pool = separation_turn_feature_pools.get(int(turn_counter))
                        if turn_pool is not None:
                            active_feature_pool = turn_pool
                            pool_mode = "turn"
                        else:
                            logger.warning(
                                f"[FeatFilter] Turn {turn_counter}: no turn-dynamic pool found; "
                                f"fallback to global pool."
                            )

                    if active_feature_pool is not None:
                        keep_cols = [c for c in anomalous_for_mining.columns if c in active_feature_pool]
                        pool_size = len(active_feature_pool)
                        keep_sample = ",".join(keep_cols[:5]) if keep_cols else "-"
                        if len(keep_cols) >= args.separation_min_cols_per_turn:
                            anomalous_for_mining = anomalous_for_mining[keep_cols].copy()
                            logger.info(
                                f"[FeatFilter] T{turn_counter} mode={pool_mode} apply=1 "
                                f"pool={pool_size} kept={len(keep_cols)}/{len(anomalous_data_in_chunk.columns)} "
                                f"min_cols={args.separation_min_cols_per_turn} sample={keep_sample}"
                            )
                        else:
                            logger.warning(
                                f"[FeatFilter] T{turn_counter} mode={pool_mode} apply=0 "
                                f"pool={pool_size} kept={len(keep_cols)}/{len(anomalous_data_in_chunk.columns)} "
                                f"min_cols={args.separation_min_cols_per_turn} sample={keep_sample}"
                            )

                    # Adjust thresholds when masking is skipped
                    min_support_eff = auto_support_controller.get_support() if auto_support_controller else args.min_support
                    min_conf_eff = args.min_confidence
                    normal_min_support_effective = normal_min_support
                    if mask_skipped:
                        if args.dominant_min_support is not None:
                            min_support_eff = args.dominant_min_support
                        if args.dominant_min_confidence is not None:
                            min_conf_eff = args.dominant_min_confidence
                        if args.dominant_normal_min_support is not None:
                            normal_min_support_effective = args.dominant_normal_min_support
                    
                    # NOTE: anomalous_for_mining already has label columns excluded
                    calculate_and_log_support_stats(
                        anomalous_for_mining,
                        min_support_eff,
                        turn_counter
                    )

                    logger.debug(f"  [Association Params] Turn: {turn_counter}, "
                                 f"Anomalous Rows: {len(anomalous_data_in_chunk)}, "
                                 f"min_support: {min_support_eff}, "
                                 f"min_confidence: {min_conf_eff}")

                    max_level = args.max_level if getattr(args, "max_level", None) is not None else LEVEL_LIMITS_BY_FILE_TYPE.get(args.file_type, LEVEL_LIMITS_BY_FILE_TYPE['default'])
                    if mask_skipped and args.dominant_level is not None:
                        max_level = args.dominant_level
                    
                    assoc_args = {
                        'association_rule_choose': args.association_method,
                        'min_support': min_support_eff,
                        'min_confidence': min_conf_eff,
                        'association_metric': 'confidence',
                        'num_processes': args.num_processes,
                        'file_type_for_limit': args.file_type,
                        'max_level_limit': max_level,
                        'itemset_limit': args.itemset_limit
                    }
                    if args.save_artifacts:
                        assoc_args['turn_counter'] = turn_counter
                        assoc_args['params_str'] = params_str
                    
                    if args.dynamic_support:
                        assoc_args['enable_dynamic_support'] = True
                        assoc_args['dynamic_support_threshold'] = args.itemset_count_threshold
                        assoc_args['support_increment_factor'] = args.support_increment_factor

                    # NOTE: anomalous_for_mining already has label columns excluded at line 526
                    '''
                    standard_rules, _ = association_module(
                        anomalous_for_mining,
                        **assoc_args
                    )
                    '''
                    retry_enabled = bool(getattr(args, "adaptive_support_retry", False))
                    retry_max = max(0, int(getattr(args, "adaptive_support_retry_max_retries", 0)))
                    retry_factor = max(1.0, float(getattr(args, "adaptive_support_retry_factor", 1.0)))
                    retry_rule_threshold = max(1, int(getattr(args, "adaptive_support_retry_rule_threshold", 50000)))
                    retry_itemset_l1_threshold = max(0, int(getattr(args, "adaptive_support_retry_itemset_l1_threshold", 0)))
                    retry_candidate_l1_threshold = max(0, int(getattr(args, "adaptive_support_retry_candidate_l1_threshold", 0)))
                    retry_bidirectional = bool(getattr(args, "adaptive_support_retry_bidirectional", False))
                    retry_low_itemset_l1_threshold = max(0, int(getattr(args, "adaptive_support_retry_low_itemset_l1_threshold", 0)))
                    retry_low_candidate_l1_threshold = max(0, int(getattr(args, "adaptive_support_retry_low_candidate_l1_threshold", 0)))
                    retry_min_support = max(0.0, float(getattr(args, "adaptive_support_retry_min_support", 1e-6)))
                    retry_max_support = max(0.0, float(getattr(args, "adaptive_support_retry_max_support", 0.9)))
                    retry_on_level_shortfall = bool(getattr(args, "adaptive_support_retry_on_level_shortfall", True))
                    target_level = getattr(args, "adaptive_support_retry_target_level", None)
                    if target_level is None:
                        target_level = int(max_level) if max_level is not None else 1

                    support_try = float(min_support_eff)
                    best_rules = []
                    best_level = -1
                    best_rule_count = -1
                    best_support = support_try
                    l1_itemset_count_cache = {}

                    def _estimate_l1_itemset_count(df_for_mining, support_value):
                        cache_key = round(float(support_value), 12)
                        if cache_key in l1_itemset_count_cache:
                            return l1_itemset_count_cache[cache_key]
                        total_rows = len(df_for_mining)
                        if total_rows <= 0:
                            l1_itemset_count_cache[cache_key] = 0
                            return 0
                        min_count = int(np.ceil(float(support_value) * total_rows))
                        if min_count < 1:
                            min_count = 1
                        l1_count = 0
                        for col_name in df_for_mining.columns:
                            try:
                                vc = df_for_mining[col_name].value_counts(dropna=False)
                                l1_count += int((vc >= min_count).sum())
                            except Exception:
                                # Skip problematic columns; keep retry logic robust.
                                continue
                        l1_itemset_count_cache[cache_key] = l1_count
                        return l1_count

                    for attempt in range(retry_max + 1):
                        assoc_args['min_support'] = support_try
                        rules_try, level_try = association_module(
                            anomalous_for_mining,
                            **assoc_args
                        )
                        level_try = int(level_try) if level_try is not None else 0
                        rule_count_try = len(rules_try) if rules_try else 0
                        l1_itemset_count_try = _estimate_l1_itemset_count(anomalous_for_mining, support_try)
                        l1_candidate_count_try = int((l1_itemset_count_try * (l1_itemset_count_try - 1)) // 2) if l1_itemset_count_try >= 2 else 0
                        logger.info(
                            f"[AdaptiveSupport] Turn {turn_counter} attempt {attempt+1}/{retry_max+1}: "
                            f"min_support={support_try:.6f}, max_level_reached={level_try}, rules={rule_count_try}, "
                            f"l1_itemsets={l1_itemset_count_try}, l1_candidates={l1_candidate_count_try}, target_level={target_level}, "
                            f"rule_threshold={retry_rule_threshold}, l1_threshold={retry_itemset_l1_threshold}, "
                            f"l1_candidate_threshold={retry_candidate_l1_threshold}, "
                            f"low_l1_threshold={retry_low_itemset_l1_threshold}, "
                            f"low_l1_candidate_threshold={retry_low_candidate_l1_threshold}, "
                            f"bidirectional={int(retry_bidirectional)}, "
                            f"retry_on_level_shortfall={int(retry_on_level_shortfall)}"
                        )

                        if (level_try > best_level) or (level_try == best_level and (best_rule_count < 0 or rule_count_try < best_rule_count)):
                            best_rules = rules_try
                            best_level = level_try
                            best_rule_count = rule_count_try
                            best_support = support_try

                        if not retry_enabled or attempt >= retry_max:
                            break

                        # Retry condition (composite): level is shallow and one of triggers is active.
                        rule_trigger = rule_count_try >= retry_rule_threshold
                        l1_trigger = (retry_itemset_l1_threshold > 0) and (l1_itemset_count_try >= retry_itemset_l1_threshold)
                        l1_candidate_trigger = (retry_candidate_l1_threshold > 0) and (l1_candidate_count_try >= retry_candidate_l1_threshold)
                        l1_low_trigger = (retry_low_itemset_l1_threshold > 0) and (l1_itemset_count_try <= retry_low_itemset_l1_threshold)
                        l1_candidate_low_trigger = (retry_low_candidate_l1_threshold > 0) and (l1_candidate_count_try <= retry_low_candidate_l1_threshold)
                        level_shortfall = level_try < target_level
                        high_trigger_any = (rule_trigger or l1_trigger or l1_candidate_trigger)
                        low_trigger_any = (l1_low_trigger or l1_candidate_low_trigger)

                        retry_direction = None
                        if level_shortfall:
                            # Default behavior is unchanged: retry upward.
                            # When bidirectional mode is enabled, allow downward retry
                            # only for low-L1 signals when high-side triggers are absent.
                            if retry_bidirectional and low_trigger_any and (not high_trigger_any):
                                retry_direction = "down"
                            elif retry_on_level_shortfall or high_trigger_any:
                                retry_direction = "up"

                        if retry_direction is None:
                            break

                        if retry_direction == "up":
                            next_support = min(support_try * retry_factor, retry_max_support)
                        else:
                            next_support = max(support_try / max(retry_factor, 1.0), retry_min_support)

                        if abs(next_support - support_try) <= 1e-12:
                            break

                        logger.warning(
                            f"[AdaptiveSupport] Turn {turn_counter}: retry with {retry_direction} min_support "
                            f"{support_try:.6f} -> {next_support:.6f} "
                            f"(level_shortfall={level_shortfall} [{level_try}<{target_level}], "
                            f"shortfall_trigger={int(retry_on_level_shortfall)}, "
                            f"rule_trigger={rule_trigger} [{rule_count_try}>={retry_rule_threshold}], "
                            f"l1_trigger={l1_trigger} [{l1_itemset_count_try}>={retry_itemset_l1_threshold}], "
                            f"l1_candidate_trigger={l1_candidate_trigger} [{l1_candidate_count_try}>={retry_candidate_l1_threshold}], "
                            f"l1_low_trigger={l1_low_trigger} [{l1_itemset_count_try}<={retry_low_itemset_l1_threshold}], "
                            f"l1_candidate_low_trigger={l1_candidate_low_trigger} [{l1_candidate_count_try}<={retry_low_candidate_l1_threshold}])."
                        )
                        support_try = next_support

                    standard_rules = best_rules
                    if best_support != float(min_support_eff):
                        logger.info(
                            f"[AdaptiveSupport] Turn {turn_counter}: selected min_support={best_support:.6f} "
                            f"(initial={float(min_support_eff):.6f}, best_level={best_level}, best_rules={best_rule_count})."
                        )
                    min_support_eff = best_support
                    # Diagnostics: rule length stats before dominant append
                    if standard_rules:
                        try:
                            lengths_pre = [len(r) for r in standard_rules if isinstance(r, dict)]
                            if lengths_pre:
                                logger.info(f"[RuleLen] Turn {turn_counter}: pre-dominant min/avg/max="
                                            f"{min(lengths_pre)}/{(sum(lengths_pre)/len(lengths_pre)):.2f}/{max(lengths_pre)} "
                                            f"(max_level={max_level})")
                        except Exception as e:
                            logger.warning(f"[RuleLen] Turn {turn_counter}: pre-dominant stats failed: {e}")
                    # --- NaN-in-signature summary (generated rules) ---
                    gen_total = len(standard_rules) if standard_rules else 0
                    gen_with_nan = sum(1 for r in standard_rules if isinstance(r, dict) and _rule_has_nan(r)) if standard_rules else 0
                    # Re-attach dominant columns (near-constant) so rules include them
                    if standard_rules and dominant_cols:
                        dominant_to_attach = dominant_cols
                        # Optionally exclude dominant cols that are nearly always true in BOTH anomaly and normal
                        if getattr(args, "dominant_attach_filter", False):
                            filtered = {}
                            for d_col, d_val in dominant_cols.items():
                                anom_freq = float((anomalous_data_in_chunk[d_col] == d_val).mean()) if d_col in anomalous_data_in_chunk.columns else 0.0
                                norm_freq = float((normal_data_in_chunk[d_col] == d_val).mean()) if d_col in normal_data_in_chunk.columns else 0.0
                                if anom_freq >= args.dominant_attach_threshold and norm_freq >= args.dominant_attach_threshold:
                                    continue
                                filtered[d_col] = d_val
                            dominant_to_attach = filtered
                        for rule in standard_rules:
                            for d_col, d_val in dominant_to_attach.items():
                                rule.setdefault(d_col, d_val)
                        # Diagnostics: rule length stats after dominant append
                        try:
                            lengths_post = [len(r) for r in standard_rules if isinstance(r, dict)]
                            if lengths_post:
                                over_max = sum(1 for l in lengths_post if l > max_level)
                                logger.info(f"[RuleLen] Turn {turn_counter}: post-dominant min/avg/max="
                                            f"{min(lengths_post)}/{(sum(lengths_post)/len(lengths_post)):.2f}/{max(lengths_post)} "
                                            f"(over_max={over_max}/{len(lengths_post)})")
                        except Exception as e:
                            logger.warning(f"[RuleLen] Turn {turn_counter}: post-dominant stats failed: {e}")
                    # Fallback: if no rules were generated, create minimal rules from dominant cols
                    is_fallback_rule = False
                    if (not standard_rules or len(standard_rules) == 0) and dominant_cols:
                        standard_rules = build_rules_from_dominant(dominant_cols)
                        is_fallback_rule = True
                        logger.info(f"[DominantCols] Created {len(standard_rules)} fallback rule(s) from dominant columns. These will bypass normal_min_support filtering.")
                        # Recompute generated stats for fallback rules
                        gen_total = len(standard_rules) if standard_rules else 0
                        gen_with_nan = sum(1 for r in standard_rules if isinstance(r, dict) and _rule_has_nan(r)) if standard_rules else 0
                    if standard_rules:
                        rule_spooler.add_rules(standard_rules)
                        logger.info(f"[RuleSpool] Queued {len(standard_rules)} rules for filtering.")
                        if rule_spool_force_flush_threshold and rule_spooler.buffer_length() >= rule_spool_force_flush_threshold:
                            logger.debug(f"[RuleSpool] Buffer reached {rule_spooler.buffer_length()} rules. Forcing flush to disk.")
                            rule_spooler.force_flush()
                    if auto_support_controller:
                        auto_support_controller.update(gen_total, turn_counter=turn_counter)
                else:
                    logger.info(f"No anomalous data found in the current chunk. Skipping new rule generation for Turn {turn_counter}.")

                # --- Filtering and Adding new rules from spool ---
                total_spooled = rule_spooler.rule_count()
                if rule_spooler.has_rules():
                    logger.info(f"Filtering {total_spooled} spooled rules in disk-backed batches...")
                    normal_data_empty = normal_data_in_chunk.empty
                    # Prepare memory-only batching, but compute support globally across all chunks
                    if args.normal_data_batch_size and len(normal_data_in_chunk) > args.normal_data_batch_size:
                        logger.info(f"Splitting normal data of size {len(normal_data_in_chunk)} into batches of {args.normal_data_batch_size} (memory only, global support).")
                        normal_data_chunks = [
                            normal_data_in_chunk.iloc[i:i + args.normal_data_batch_size]
                            for i in range(0, len(normal_data_in_chunk), args.normal_data_batch_size)
                        ]
                    else:
                        normal_data_chunks = [normal_data_in_chunk] if not normal_data_empty else []
                    
                    # Rule filtering logic extracted to utils.isv_rule_filtering_runtime

                    total_filtered = 0
                    filtered_with_nan_total = 0
                    added_new_with_nan = 0
                    # Check for fallback rules only if we created fallback rules this turn
                    check_for_fallback = is_fallback_rule
                    for rules_batch in rule_spooler.consume_chunks():
                        #filtered_rules = filter_rule_batch(rules_batch, check_fallback=check_for_fallback)
                        filtered_rules = filter_rule_batch_runtime(
                            rules_batch,
                            check_fallback=check_for_fallback,
                            dominant_cols=dominant_cols,
                            normal_data_empty=normal_data_empty,
                            num_processes=args.num_processes,
                            normal_data_chunks=normal_data_chunks,
                            normal_min_support_effective=normal_min_support_effective,
                            negative_filtering_enabled=negative_filtering_enabled,
                            negative_filter_threshold=negative_filter_threshold,
                            strict_normal_zero=strict_normal_zero,
                            anomalous_data_in_chunk=anomalous_data_in_chunk,
                        )
                        total_filtered += len(filtered_rules)
                        filtered_with_nan_total += sum(1 for r in filtered_rules if isinstance(r, dict) and _rule_has_nan(r))
                        for rule in filtered_rules:
                            # Convert rule to original value ranges (per-turn mapping)
                            reversed_rule = reverse_map_rule_with_fallback(
                                rule,
                                turn_original_value_range_mapping,
                                turn_reverse_mapping,
                            )
                            normalized_rule = _normalize_rule_conditions(reversed_rule)
                            rule_id = hash(frozenset(normalized_rule.items()))
                            if (
                                rule_id not in all_valid_signatures
                                and rule_id not in sleeping_signatures
                                and rule_id not in signatures_to_remove
                            ):
                                all_valid_signatures[rule_id] = normalized_rule
                                newly_added_sig_ids_this_turn.add(rule_id)
                                signature_turn_created[rule_id] = turn_counter  # Record when this signature was created
                                new_signatures_found += 1
                                if isinstance(normalized_rule, dict) and _rule_has_nan(normalized_rule):
                                    added_new_with_nan += 1
                    logger.info(f"{total_filtered} rules passed filter.")
                    logger.info(f"Added {new_signatures_found} new unique signatures.")
                    # Record per-turn NaN stats (generated vs filtered vs added-new)
                    nan_sig_turn_stats.append({
                        "turn": int(turn_counter),
                        "generated_total": int(gen_total),
                        "generated_with_nan": int(gen_with_nan),
                        "filtered_total": int(total_filtered),
                        "filtered_with_nan": int(filtered_with_nan_total),
                        "added_new_total": int(new_signatures_found),
                        "added_new_with_nan": int(added_new_with_nan),
                        "is_fallback_rule": bool(is_fallback_rule),
                    })
                else:
                    logger.info("No new anomalous rules were generated in this turn.")
                    # Still record the turn for visibility
                    nan_sig_turn_stats.append({
                        "turn": int(turn_counter),
                        "generated_total": 0,
                        "generated_with_nan": 0,
                        "filtered_total": 0,
                        "filtered_with_nan": 0,
                        "added_new_total": 0,
                        "added_new_with_nan": 0,
                        "is_fallback_rule": False,
                    })

            finally:
                rule_spooler.cleanup()
                gc.collect()

            # --- 2.5 Validation Step (FP Detection for newly added signatures this turn) ---
            if not args.disable_fp_removal and newly_added_sig_ids_this_turn and not normal_data_raw.empty:
                new_sig_subset = {
                    sid: all_valid_signatures[sid]
                    for sid in newly_added_sig_ids_this_turn
                    if sid in all_valid_signatures and sid not in signatures_to_remove
                }
                post_removed, post_flagged = validate_fp_subset_runtime(
                    signatures_subset=new_sig_subset,
                    turn_counter=turn_counter,
                    phase_tag="post_new",
                    rules_at_turn_start_ref=rules_at_turn_start,
                    normal_data_raw=normal_data_raw,
                    anomalous_data_raw=anomalous_data_raw,
                    turn_eval_data=turn_eval_data,
                    args=args,
                    run_dir=run_dir,
                    signatures_to_remove=signatures_to_remove,
                    fp_event_records=fp_event_records,
                    turn_fp_events_csv=turn_fp_events_csv,
                    apply_signatures_to_dataset_fn=apply_signatures_to_dataset,
                    evaluate_false_positives_fn=evaluate_false_positives,
                    summarize_fp_results_fn=summarize_fp_results,
                    run_fp_diagnostics_fn=run_fp_diagnostics,
                )
                newly_removed_count += post_removed
                newly_flagged_for_removal.update(post_flagged)
                removed_ids_this_turn.update(set(post_flagged))

            # --- Precision underlimit filtering (cluster-based, per turn, before reduction) ---
            if args.precision_underlimit is not None:
                if all_valid_signatures and turn_eval_data is not None and not turn_eval_data.empty:
                    if 'cluster' in turn_eval_data.columns:
                        try:
                            keep_no_alert_mode = int(getattr(args, "precision_underlimit_keep_no_alert", 0))
                            if keep_no_alert_mode not in (0, 1, 2):
                                keep_no_alert_mode = 0

                            # keep_no_alert=2: evaluate both active and sleeping pools, then split back.
                            eval_signature_pool = dict(all_valid_signatures)
                            if keep_no_alert_mode == 2 and sleeping_signatures:
                                for _sid, _rule in sleeping_signatures.items():
                                    if _sid not in signatures_to_remove:
                                        eval_signature_pool.setdefault(_sid, _rule)

                            # Build signature list for matching
                            sorted_sigs_for_limit = sorted(eval_signature_pool.items(), key=lambda x: x[0])
                            formatted_sigs_for_limit = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in sorted_sigs_for_limit]
                            alerts_df = apply_signatures_to_dataset(turn_eval_data, formatted_sigs_for_limit)

                            sig_metrics = build_cluster_based_sig_metrics(
                                turn_eval_data,
                                sorted_sigs_for_limit,
                                alerts_df,
                            )
                            no_alert_signature_ids_this_turn = {
                                m.get("signature_id")
                                for m in sig_metrics
                                if (m.get("signature_id") is not None) and bool(m.get("no_alert", False))
                            }

                            precision_thr = float(args.precision_underlimit)
                            temporal_enabled = bool(int(getattr(args, "precision_underlimit_use_temporal", 0)))
                            temporal_mode = str(getattr(args, "precision_underlimit_temporal_mode", "rolling"))
                            temporal_window = max(1, int(getattr(args, "precision_underlimit_temporal_window", 3)))
                            keep_no_alert = keep_no_alert_mode  #bool(int(getattr(args, "precision_underlimit_keep_no_alert", 0)))
                            no_alert_max_streak = max(0, int(getattr(args, "precision_underlimit_no_alert_max_streak", 3)))
                            auto_adjust = bool(int(getattr(args, "precision_underlimit_auto_adjust", 0)))
                            auto_factor = float(getattr(args, "precision_underlimit_auto_adjust_factor", 0.9))
                            auto_max_retries = max(0, int(getattr(args, "precision_underlimit_auto_adjust_max_retries", 10)))
                            auto_start_threshold = float(getattr(args, "precision_underlimit_auto_adjust_start_threshold", 0.9))
                            auto_recall_floor = float(getattr(args, "precision_underlimit_auto_adjust_recall_floor", 0.95))
                            auto_precision_floor = float(getattr(args, "precision_underlimit_auto_adjust_precision_floor", 0.95))
                            auto_min_thr = max(0.0, float(getattr(args, "precision_underlimit_auto_adjust_min", 0.0)))

                            if auto_adjust and precision_thr > 0.0 and 0.0 < auto_factor < 1.0:
                                selected_metrics, filter_desc, precision_underlimit_state = run_precision_underlimit_auto_adjust(
                                    sig_metrics=sig_metrics,
                                    args=args,
                                    precision_underlimit_state=precision_underlimit_state,
                                    all_valid_signatures=eval_signature_pool,
                                    turn_eval_data=turn_eval_data,
                                    calculate_performance_in_batches_fn=calculate_performance_in_batches,
                                    precision_thr=precision_thr,
                                    temporal_enabled=temporal_enabled,
                                    temporal_mode=temporal_mode,
                                    temporal_window=temporal_window,
                                    keep_no_alert=keep_no_alert,
                                    no_alert_max_streak=no_alert_max_streak,
                                    auto_factor=auto_factor,
                                    auto_max_retries=auto_max_retries,
                                    auto_start_threshold=auto_start_threshold,
                                    auto_recall_floor=auto_recall_floor,
                                    auto_precision_floor=auto_precision_floor,
                                    auto_min_thr=auto_min_thr,
                                    turn_counter=turn_counter,
                                    logger=logger,
                                )
                            else:
                                selected_metrics, filter_desc, precision_underlimit_state = select_signatures_by_precision_underlimit(
                                    sig_metrics,
                                    precision_threshold=precision_thr,
                                    signature_ea=args.signature_ea,
                                    temporal_enabled=temporal_enabled,
                                    temporal_mode=temporal_mode,
                                    temporal_window=temporal_window,
                                    keep_no_alert=keep_no_alert,
                                    no_alert_max_streak=no_alert_max_streak,
                                    state=precision_underlimit_state,
                                )
                            selected_ids = {m.get("signature_id") for m in selected_metrics if m.get("signature_id") is not None}
                            before_count = len(eval_signature_pool)
                            active_before_count = len(all_valid_signatures)
                            active_before_ids = set(all_valid_signatures.keys())

                            if keep_no_alert_mode == 2:
                                metric_by_id = {
                                    m.get("signature_id"): m
                                    for m in sig_metrics
                                    if m.get("signature_id") is not None
                                }
                                next_active = {}
                                next_sleep = {}
                                for sid, rule in eval_signature_pool.items():
                                    if sid in selected_ids:
                                        next_active[sid] = rule  # stand/re-activate
                                        continue
                                    sig_m = metric_by_id.get(sid, {})
                                    is_no_alert = bool(sig_m.get("no_alert", False))
                                    if is_no_alert or sid in sleeping_signatures:
                                        next_sleep[sid] = rule  # sleep (not deleted)
                                all_valid_signatures = next_active
                                sleeping_signatures = next_sleep
                                retained_ids = set(all_valid_signatures.keys()) | set(sleeping_signatures.keys())
                                signature_turn_created = {sid: t for sid, t in signature_turn_created.items() if sid in retained_ids}
                            else:
                                if selected_ids:
                                    all_valid_signatures = {sid: rule for sid, rule in all_valid_signatures.items() if sid in selected_ids}
                                    signature_turn_created = {sid: t for sid, t in signature_turn_created.items() if sid in selected_ids}
                                else:
                                    all_valid_signatures = {}
                                    signature_turn_created = {}

                            # Clean stale precision-underlimit state for removed signatures
                            active_ids = set(all_valid_signatures.keys()) | set(sleeping_signatures.keys())
                            precision_underlimit_state = cleanup_precision_underlimit_state(
                                precision_underlimit_state,
                                active_ids,
                            )
                            active_after_ids = set(all_valid_signatures.keys())
                            underlimit_removed_ids = active_before_ids - active_after_ids
                            underlimit_removed_count = len(underlimit_removed_ids)
                            removed_ids_this_turn.update(underlimit_removed_ids)
                            after_count = len(all_valid_signatures)
                            if keep_no_alert_mode == 2:
                                logger.info(
                                    f"[UnderLimit] Turn {turn_counter}: active={after_count}, "
                                    f"sleep={len(sleeping_signatures)}, pool={before_count} ({filter_desc})."
                                )
                            else:
                                logger.info(f"[UnderLimit] Turn {turn_counter}: kept {after_count}/{before_count} signatures ({filter_desc}).")
                        except Exception as e:
                            logger.warning(f"[UnderLimit] Turn {turn_counter}: failed to apply precision_underlimit. Error: {e}")
                    else:
                        logger.warning(f"[UnderLimit] Turn {turn_counter}: 'cluster' column missing; skipping precision_underlimit.")

            # --- Pruning Step (Signature Organization) ---
            if args.prune_signatures and all_valid_signatures:
                initial_sig_count = len(all_valid_signatures)
                ids_before_pruning = set(all_valid_signatures.keys())
                
                # The organize_signatures function will prune based on redundancy against the full dataset so far.
                all_valid_signatures = organize_signatures(
                    all_signatures=all_valid_signatures,
                    data_provider=chunk_cache,
                    data_batch_size=args.evaluation_batch_size,
                    num_processes=args.num_processes,
                    run_dir=run_dir,
                    turn_counter=turn_counter,
                    coverage_threshold=args.prune_coverage_threshold,
                    enable_merging=args.merge_signatures,
                    merge_infrequent_threshold=args.merge_infrequent_threshold
                )
                
                final_sig_count = len(all_valid_signatures)
                ids_after_pruning = set(all_valid_signatures.keys())
                removed_ids_this_turn.update(ids_before_pruning - ids_after_pruning)
                # The number of rules removed by pruning, for logging purposes.
                num_pruned_this_turn = initial_sig_count - final_sig_count
                if num_pruned_this_turn > 0:
                    # Add to the turn's removed count for the history log
                    newly_removed_count += num_pruned_this_turn

                # Encourage GC to reclaim any large temporary structures created during organization
                gc.collect()

            # --- 3. Calculate Reduction/Inactive (recording; optionally applied) ---
            # NOTE: We calculate reduction/inactive removal counts for history recording.
            # If --apply_turn_reduction_inactive is set, we also apply removals to all_valid_signatures.
            
            # Create recording/evaluation view from currently active signatures.
            # IMPORTANT: Only apply inactive/reduction when corresponding --apply_turn_* flag is enabled.
            signatures_for_recording = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
            
            # --- Calculate Inactive Removal (applied only when enabled) ---
            new_signatures_before_inactive = {
                sig_id for sig_id in signatures_for_recording.keys()
                if signature_turn_created.get(sig_id) == turn_counter
            }
            inactive_removed_count = 0
            inactive_removed_from_new = 0
            inactive_ids = set()
            if args.apply_turn_inactive_removal and signatures_for_recording and turn_eval_data is not None and not turn_eval_data.empty:
                sorted_sigs_for_inactive = sorted(signatures_for_recording.items(), key=lambda x: x[0])
                formatted_sigs_for_inactive = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} 
                                               for sid, r in sorted_sigs_for_inactive]
                _, _, _, _, _, _, inactive_eval_alerts = calculate_performance_in_batches(
                    turn_eval_data,
                    formatted_sigs_for_inactive,
                    args.evaluation_batch_size,
                    label_col=eval_label_col,
                )
                if not inactive_eval_alerts.empty and 'signature_id' in inactive_eval_alerts.columns:
                    active_sig_ids = set(inactive_eval_alerts['signature_id'].dropna().unique())
                    all_sig_ids = set(signatures_for_recording.keys())
                    inactive_ids = all_sig_ids - active_sig_ids
                    inactive_removed_count = len(inactive_ids)
                    if inactive_removed_count > 0:
                        for inactive_id in inactive_ids:
                            if inactive_id in new_signatures_before_inactive:
                                inactive_removed_from_new += 1
                            if inactive_id in signatures_for_recording:
                                del signatures_for_recording[inactive_id]
                            if inactive_id in all_valid_signatures:
                                del all_valid_signatures[inactive_id]
                            if inactive_id in signature_turn_created:
                                del signature_turn_created[inactive_id]
                            signature_turn_removed.setdefault(inactive_id, turn_counter)
                    removed_ids_this_turn.update(inactive_ids)
            
            # --- Calculate Reduction (applied only when enabled) ---
            new_signatures_before_reduction = {
                sig_id for sig_id in signatures_for_recording.keys()
                if signature_turn_created.get(sig_id) == turn_counter
            }
            reduction_removed_count = 0
            reduction_removed_from_new = 0
            reduction_removed_ids = set()
            if args.apply_turn_reduction_removal and signatures_for_recording:
                reduction_count_before = len(signatures_for_recording)
                ids_before_reduction = set(signatures_for_recording.keys())
                signatures_for_recording = reduce_signatures_by_subsets(signatures_for_recording, num_processes=args.num_processes)
                reduction_count_after = len(signatures_for_recording)
                reduction_removed_count = reduction_count_before - reduction_count_after
                if reduction_removed_count > 0:
                    ids_after_reduction = set(signatures_for_recording.keys())
                    reduction_removed_ids = ids_before_reduction - ids_after_reduction
                    for removed_id in reduction_removed_ids:
                        if removed_id in new_signatures_before_reduction:
                            reduction_removed_from_new += 1
                        if removed_id in all_valid_signatures:
                            del all_valid_signatures[removed_id]
                        if removed_id in signature_turn_created:
                            del signature_turn_created[removed_id]
                        signature_turn_removed.setdefault(removed_id, turn_counter)
                    removed_ids_this_turn.update(reduction_removed_ids)
            
            # --- Exit Performance Evaluation (on reduced/inactive-removed set for recording) ---
            # Use the reduced/inactive-removed version for exit evaluation recording
            exit_sig_count = len(signatures_for_recording)
            formatted_exit_sigs = []
            exit_alerts = pd.DataFrame()
            exit_recall, exit_precision, exit_f1, exit_accuracy = 0, 0, 0, 0
            exit_cluster_recall, exit_cluster_precision, exit_cluster_f1, exit_cluster_accuracy = 0, 0, 0, 0
            if turn_eval_data is not None and not turn_eval_data.empty:
                if signatures_for_recording:
                    # Sort by signature ID to ensure consistent ordering (for "first match wins" logic)
                    sorted_exit_sigs = sorted(signatures_for_recording.items(), key=lambda x: x[0])
                    formatted_exit_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} for sid, r in sorted_exit_sigs]
                    
                    # --- Diagnostic (first exit eval with signatures): why might matching fail? ---
                    _do_match_diag = getattr(args, "_match_diag_done", None) is not True and bool(formatted_exit_sigs)
                    if _do_match_diag:
                        args._match_diag_done = True
                    if _do_match_diag:
                        _rule = formatted_exit_sigs[0].get('rule_dict', {})
                        _rule_keys = set(_rule.keys())
                        _eval_cols = set(turn_eval_data.columns)
                        _missing = _rule_keys - _eval_cols
                        logger.info(f"[MatchDiag] Turn {turn_counter} (first exit eval with sigs) first rule: {len(_rule)} conditions.")
                        if _missing:
                            logger.info(f"[MatchDiag] Turn {turn_counter} first rule: keys in rule but NOT in eval data: {sorted(_missing)}")
                        _n = len(turn_eval_data)
                        for _col, _cond in list(_rule.items())[:8]:
                            if _col not in turn_eval_data.columns:
                                logger.info(f"[MatchDiag] Turn {turn_counter} first rule: col={_col!r} -> MISSING in df, cond={_cond!r}")
                                continue
                            _s = turn_eval_data[_col]
                            if isinstance(_cond, tuple) and len(_cond) == 4:
                                _left, _right, _li, _ri = _cond
                                _num = pd.to_numeric(_s, errors='coerce')
                                _m = (_num >= _left if _li else _num > _left) & (_num <= _right if _ri else _num < _right)
                            elif isinstance(_cond, (int, float)):
                                _m = pd.to_numeric(_s, errors='coerce') == float(_cond)
                            elif isinstance(_cond, str):
                                try:
                                    _m = pd.to_numeric(_s, errors='coerce') == float(_cond)
                                except (ValueError, TypeError):
                                    _m = _s.astype(str) == str(_cond)
                            else:
                                _m = _s.astype(str) == str(_cond)
                            _cnt = int(_m.sum())
                            logger.info(f"[MatchDiag] Turn {turn_counter} first rule: col={_col!r} cond={_cond!r} -> rows_ok={_cnt}/{_n}")
                        if len(_rule) > 8:
                            logger.info(f"[MatchDiag] Turn {turn_counter} first rule: ... and {len(_rule) - 8} more conditions")
                    
                    # --- MODIFIED: Use batched performance calculation ---
                    _, _, exit_recall, exit_precision, exit_f1, exit_accuracy, exit_alerts = calculate_performance_in_batches(
                        turn_eval_data,
                        formatted_exit_sigs,
                        args.evaluation_batch_size,
                        label_col=eval_label_col,
                    )
                    # Reference-only cluster evaluation from existing alerts (no extra matching, no leakage into generation).
                    if getattr(args, "eval_target", "label") != "cluster" and "cluster" in turn_eval_data.columns:
                        exit_cluster_recall, exit_cluster_precision, exit_cluster_f1, exit_cluster_accuracy = calculate_metrics_from_alerts(
                            exit_alerts, turn_eval_data, "cluster"
                        )
                else:
                    # No signatures: calculate accuracy based on true negatives (no alerts = all normal correctly identified)
                    # TP=0, FP=0, TN=normal_count, FN=anomaly_count
                    if eval_label_col in turn_eval_data.columns:
                        total_rows = len(turn_eval_data)
                        normal_count = (turn_eval_data[eval_label_col] == 0).sum()
                        anomaly_count = (turn_eval_data[eval_label_col] == 1).sum()
                        # With no signatures: TP=0, FP=0, TN=normal_count, FN=anomaly_count
                        # Accuracy = (TP + TN) / total = TN / total = normal_count / total_rows
                        exit_accuracy = normal_count / total_rows if total_rows > 0 else 0.0
                        # Recall = TP / (TP + FN) = 0 / (0 + anomaly_count) = 0
                        exit_recall = 0.0
                        # Precision = TP / (TP + FP) = 0 / (0 + 0) = 0 (undefined, set to 0)
                        exit_precision = 0.0
                        exit_f1 = 0.0
                        logger.debug(f"[NoSignatures] Calculated accuracy with 0 signatures: {exit_accuracy:.4f} (normal: {normal_count}/{total_rows})")
                    if getattr(args, "eval_target", "label") != "cluster" and "cluster" in turn_eval_data.columns:
                        exit_cluster_recall, exit_cluster_precision, exit_cluster_f1, exit_cluster_accuracy = calculate_metrics_from_alerts(
                            exit_alerts, turn_eval_data, "cluster"
                        )
            else:
                formatted_exit_sigs = []

            exit_attack_recall_full = {}
            if getattr(args, "save_attackwise_turn_recall", False) and full_eval_cache is not None:
                if formatted_exit_sigs:
                    _, _, _, _, _, _, exit_alerts_full = calculate_performance_in_batches(
                        full_eval_cache,
                        formatted_exit_sigs,
                        args.evaluation_batch_size,
                        label_col=eval_label_col,
                        show_apply_progress=False,
                        show_apply_info=False,
                        show_batch_progress=False,
                    )
                else:
                    exit_alerts_full = pd.DataFrame()
                exit_attack_recall_full, attackwise_exit_captured_state = compute_attackwise_recall_turn_locked_cumulative(
                    exit_alerts_full,
                    attack_to_indices_by_turn_full,
                    attack_totals_full,
                    turn_counter=turn_counter,
                    captured_state=attackwise_exit_captured_state,
                )

            # --- Optional per-turn FP contribution CSV (by signature creation turn) ---
            if getattr(args, "save_turn_fp_contribution", False):
                try:
                    entry_fp_rows = build_fp_contribution_rows_by_creation_turn(
                        turn_counter=turn_counter,
                        phase_tag="entry",
                        alerts_df=entry_alerts,
                        turn_eval_data=turn_eval_data,
                        eval_label_col=eval_label_col,
                        signature_turn_created=signature_turn_created,
                    )
                    exit_fp_rows = build_fp_contribution_rows_by_creation_turn(
                        turn_counter=turn_counter,
                        phase_tag="exit",
                        alerts_df=exit_alerts,
                        turn_eval_data=turn_eval_data,
                        eval_label_col=eval_label_col,
                        signature_turn_created=signature_turn_created,
                    )
                    fp_contrib_df = pd.concat([entry_fp_rows, exit_fp_rows], ignore_index=True)
                    fp_contrib_df.to_csv(
                        turn_fp_contrib_csv,
                        mode="a",
                        index=False,
                        header=not os.path.exists(turn_fp_contrib_csv),
                        encoding='utf-8-sig',
                    )
                except Exception as e:
                    logger.warning(f"[FPContrib] Turn {turn_counter}: failed to save FP contribution CSV. Error: {e}")

                # Optional detailed per-rule phase stats for deeper FP analysis
                try:
                    entry_rule_stats = build_rule_phase_stats_by_signature(
                        turn_counter=turn_counter,
                        phase_tag="entry",
                        alerts_df=entry_alerts,
                        turn_eval_data=turn_eval_data,
                        eval_label_col=eval_label_col,
                        signatures_map=rules_at_turn_start,
                        signature_turn_created=signature_turn_created,
                    )
                    exit_rule_stats = build_rule_phase_stats_by_signature(
                        turn_counter=turn_counter,
                        phase_tag="exit",
                        alerts_df=exit_alerts,
                        turn_eval_data=turn_eval_data,
                        eval_label_col=eval_label_col,
                        signatures_map=signatures_for_recording,
                        signature_turn_created=signature_turn_created,
                    )
                    phase_df = pd.concat([entry_rule_stats, exit_rule_stats], ignore_index=True)
                    if not phase_df.empty:
                        phase_df.to_csv(
                            turn_fp_rule_phase_csv,
                            mode="a",
                            index=False,
                            header=not os.path.exists(turn_fp_rule_phase_csv),
                            encoding='utf-8-sig',
                        )
                except Exception as e:
                    logger.warning(f"[FPContrib] Turn {turn_counter}: failed to save per-rule phase stats. Error: {e}")
            
            # Calculate net generated count (new signatures minus those removed by reduction/inactive from new ones)
            # This is what will be displayed in the graph as "generated"
            net_generated = new_signatures_found - reduction_removed_from_new - inactive_removed_from_new
            created_ids_this_turn = set(newly_added_sig_ids_this_turn)
            created_and_removed_same_turn = created_ids_this_turn & removed_ids_this_turn
            created_survived_same_turn = created_ids_this_turn - removed_ids_this_turn
            removed_not_created_this_turn = removed_ids_this_turn - created_ids_this_turn
            # Actual-set based churn (excludes recording-only inactive/reduction effects).
            active_actual_ids_at_exit = set(all_valid_signatures.keys()) - set(signatures_to_remove)
            actual_removed_ids_this_turn = active_actual_ids_at_entry - active_actual_ids_at_exit
            # Intended bars:
            # - generated: newly created this turn that remain active at exit
            # - removed: active-at-entry signatures removed by exit (not recording-only removals)
            plot_generated_survived_actual_only = len(created_ids_this_turn & active_actual_ids_at_exit)
            plot_removed_not_created_actual_only = len(actual_removed_ids_this_turn)
            plot_net_change_actual_only = plot_generated_survived_actual_only - plot_removed_not_created_actual_only
            # No-alert excluded bars: remove signatures that were evaluated as no-alert in this turn.
            generated_survived_no_alert_excluded_ids = (created_ids_this_turn & active_actual_ids_at_exit) - no_alert_signature_ids_this_turn
            removed_not_created_no_alert_excluded_ids = actual_removed_ids_this_turn - no_alert_signature_ids_this_turn
            plot_generated_survived_actual_only_no_alert_excluded = len(generated_survived_no_alert_excluded_ids)
            plot_removed_not_created_actual_only_no_alert_excluded = len(removed_not_created_no_alert_excluded_ids)
            plot_net_change_actual_only_no_alert_excluded = (
                plot_generated_survived_actual_only_no_alert_excluded
                - plot_removed_not_created_actual_only_no_alert_excluded
            )
            
            # Calculate actual signature count change in the actual signature set (not recording copy).
            # NOTE: signatures_to_remove can contain stale IDs not present in all_valid_signatures,
            # so set-difference count must be used instead of len(A)-len(B) to avoid negative values.
            actual_exit_sig_count = len(set(all_valid_signatures.keys()) - set(signatures_to_remove))
            actual_net_change = actual_exit_sig_count - entry_sig_count
            
            # Calculate balanced removed count to ensure: generated - removed = actual_net_change
            # This ensures the accounting is correct: sum(generated) - sum(removed) = final signature count
            # removed = generated - actual_net_change
            balanced_removed = max(0, net_generated - actual_net_change)

            logger.info(f"End of Turn {turn_counter}. Signatures (actual): {entry_sig_count} -> {actual_exit_sig_count} (after reduction/inactive removal for recording: {exit_sig_count}). "
                       f"EXIT Recall: {exit_recall:.4f}. EXIT Precision: {exit_precision:.4f}. EXIT F1: {exit_f1:.4f}. EXIT Accuracy: {exit_accuracy:.4f}")
            if getattr(args, "eval_target", "label") != "cluster" and turn_eval_data is not None and "cluster" in turn_eval_data.columns:
                logger.info(
                    f"(cluster ref) T{turn_counter} entry->exit: "
                    f"R {entry_cluster_recall:.4f}->{exit_cluster_recall:.4f}, "
                    f"P {entry_cluster_precision:.4f}->{exit_cluster_precision:.4f}, "
                    f"F1 {entry_cluster_f1:.4f}->{exit_cluster_f1:.4f}, "
                    f"Acc {entry_cluster_accuracy:.4f}->{exit_cluster_accuracy:.4f}"
                )
            logger.info(f"[Recording] Inactive removed: {inactive_removed_count} (from new: {inactive_removed_from_new}), Reduction removed: {reduction_removed_count} (from new: {reduction_removed_from_new}) (for history recording only, not applied to actual set)")
            logger.info(f"[Balance] Actual net change: {actual_net_change} = exit({actual_exit_sig_count}) - entry({entry_sig_count}), Generated: {net_generated}, Removed (FP+Pruning): {newly_removed_count}, Removed (balanced): {balanced_removed}")
            logger.info(f"[UnderLimit] Turn {turn_counter}: active-removed-by-underlimit={underlimit_removed_count}")

            # --- Per-turn diagnostics CSV ---
            try:
                diag_row = {
                    "turn": int(turn_counter),
                    "entry_signature_count": int(entry_sig_count),
                    "generated_rules": int(gen_total),
                    "filtered_rules": int(total_filtered),
                    "new_signatures": int(new_signatures_found),
                    "min_support_used": float(min_support_eff),
                    "min_confidence_used": float(min_conf_eff),
                    "normal_min_support_used": float(normal_min_support_effective),
                    "max_level": int(max_level) if max_level is not None else None,
                    "n_splits": int(args.n_splits),
                    "dominant_cols_count": int(len(dominant_cols) if isinstance(dominant_cols, dict) else 0),
                    "dominant_cols_sample": str(list(dominant_cols.items())[:10]) if isinstance(dominant_cols, dict) else "",
                    "entry_precision": float(entry_precision),
                    "exit_precision": float(exit_precision),
                    "entry_recall": float(entry_recall),
                    "exit_recall": float(exit_recall),
                }
                pd.DataFrame([diag_row]).to_csv(
                    turn_diag_csv,
                    mode="a",
                    index=False,
                    header=not os.path.exists(turn_diag_csv),
                    encoding='utf-8-sig',
                )
            except Exception as e:
                logger.warning(f"[TurnDiag] Turn {turn_counter}: failed to save diagnostics CSV. Error: {e}")
            
            history_row = {
                'turn': turn_counter, 
                'entry_signature_count': entry_sig_count, 
                'generated': net_generated,  # Net generated (after reduction/inactive removal from new signatures)
                'removed': balanced_removed,  # Balanced to ensure: generated - removed = actual_net_change (so accounting sums correctly)
                'generated_new_signatures_raw': new_signatures_found,  # Raw new signatures before net adjustment
                'removed_fp_pruning_actual': newly_removed_count,  # Actual removals by FP/pruning path
                'removed_underlimit_actual': underlimit_removed_count,  # Active signatures removed/deactivated by underlimit
                'plot_generated_excl_inactive_reduction': new_signatures_found,  # For external plotting (exclude inactive/reduction)
                'plot_removed_excl_inactive_reduction': (newly_removed_count + underlimit_removed_count),  # For external plotting (exclude inactive/reduction)
                'generated_survived_same_turn': len(created_survived_same_turn),  # Newly created this turn and NOT removed this turn
                'removed_not_created_this_turn': len(removed_not_created_this_turn),  # Removed this turn among signatures NOT created this turn
                'created_and_removed_same_turn': len(created_and_removed_same_turn),  # Diagnostic: created and removed within same turn
                'plot_generated_survived_same_turn': len(created_survived_same_turn),  # Preferred generation bar for external plotting
                'plot_removed_not_created_this_turn': len(removed_not_created_this_turn),  # Preferred removal bar for external plotting
                'plot_generated_survived_actual_only': plot_generated_survived_actual_only,  # New: generation bar aligned to actual active-set churn
                'plot_removed_not_created_actual_only': plot_removed_not_created_actual_only,  # New: removal bar aligned to actual active-set churn
                'plot_net_change_actual_only': plot_net_change_actual_only,  # New: sanity-check net = generated - removed (actual-only bars)
                'plot_generated_survived_actual_only_no_alert_excluded': plot_generated_survived_actual_only_no_alert_excluded,  # New: actual-only generation bar excluding no-alert signatures
                'plot_removed_not_created_actual_only_no_alert_excluded': plot_removed_not_created_actual_only_no_alert_excluded,  # New: actual-only removal bar excluding no-alert signatures
                'plot_net_change_actual_only_no_alert_excluded': plot_net_change_actual_only_no_alert_excluded,  # New: net change for no-alert-excluded actual-only bars
                'actual_exit_signature_count': actual_exit_sig_count,  # Actual active signatures after this turn
                'actual_net_change': actual_net_change,  # actual_exit_signature_count - entry_signature_count
                'inactive_removed': inactive_removed_count,  # Total inactive removals (for logging)
                'inactive_removed_from_new': inactive_removed_from_new,  # Inactive from new signatures (for reference)
                'reduction_removed': reduction_removed_count,  # Total reduction removals (for logging)
                'reduction_removed_from_new': reduction_removed_from_new,  # Reduction from new signatures (for reference)
                'exit_signature_count': exit_sig_count,  # After reduction/inactive removal for recording
                'entry_recall': entry_recall, 
                'entry_precision': entry_precision, 
                'entry_f1': entry_f1, 
                'entry_accuracy': entry_accuracy, 
                'exit_recall': exit_recall, 
                'exit_precision': exit_precision, 
                'exit_f1': exit_f1, 
                'exit_accuracy': exit_accuracy
            }
            if getattr(args, "save_attackwise_turn_recall", False):
                all_attacks = sorted(set(entry_attack_recall_full.keys()) | set(exit_attack_recall_full.keys()))
                for atk in all_attacks:
                    atk_col = _sanitize_attack_name_for_column(atk)
                    history_row[f"entry_attack_recall_{atk_col}"] = float(entry_attack_recall_full.get(atk, np.nan))
                    history_row[f"exit_attack_recall_{atk_col}"] = float(exit_attack_recall_full.get(atk, np.nan))
            history.append(history_row)
            
            # --- NEW: Save Checkpoint ---
            save_checkpoint(run_dir, turn_counter, all_valid_signatures, signatures_to_remove, history,
                            signature_turn_created, signature_turn_removed, sleeping_signatures)

            # Turn-level GC to clean up any remaining temporary objects before next chunk
            try:
                del turn_eval_data
            except NameError:
                pass
            gc.collect()

        # --- Finalization ---
        # NOTE: Apply reduction and inactive removal to final signatures for saving
        # (but the actual signature set during turns was not affected)
        final_signatures_before_reduction = {sig_id: rule for sig_id, rule in all_valid_signatures.items() if sig_id not in signatures_to_remove}
        
        logger.info("--- Applying final reduction and inactive removal to signatures for saving ---")
        
        # Final inactive removal: remove signatures that never triggered alerts
        if final_signatures_before_reduction and chunk_cache is not None and not chunk_cache.is_empty():
            logger.info(f"Evaluating {len(final_signatures_before_reduction)} final signatures to identify inactive ones...")
            formatted_final_sigs = [{'id': sid, 'name': f'Sig_{sid}', 'rule_dict': r} 
                                   for sid, r in final_signatures_before_reduction.items()]
            
            # Evaluate against entire dataset to find which signatures triggered alerts
            eval_label_col_final = "cluster" if getattr(args, "eval_target", "label") == "cluster" else "label"
            _, _, _, _, _, _, final_eval_alerts = calculate_performance_in_batches(
                chunk_cache,
                formatted_final_sigs,
                args.evaluation_batch_size,
                label_col=eval_label_col_final,
                show_apply_progress=False,
                show_apply_info=False,
                show_batch_progress=True,
            )
            
            inactive_count_before_final = len(final_signatures_before_reduction)
            if not final_eval_alerts.empty and 'signature_id' in final_eval_alerts.columns:
                # Get signature IDs that triggered alerts
                active_sig_ids = set(final_eval_alerts['signature_id'].dropna().unique())
                # Find inactive signatures
                all_sig_ids = set(final_signatures_before_reduction.keys())
                inactive_ids = all_sig_ids - active_sig_ids
                
                if inactive_ids:
                    # Remove inactive signatures
                    for inactive_id in inactive_ids:
                        if inactive_id in final_signatures_before_reduction:
                            del final_signatures_before_reduction[inactive_id]
                            # Also remove from signature_turn_created for consistency
                            if inactive_id in signature_turn_created:
                                del signature_turn_created[inactive_id]
                    
                    inactive_count_after_final = len(final_signatures_before_reduction)
                    final_inactive_removed = inactive_count_before_final - inactive_count_after_final
                    logger.info(f"[Final Inactive Removal] Removed {final_inactive_removed} inactive signatures. "
                               f"Signatures: {inactive_count_before_final} -> {inactive_count_after_final}")
        
        # Final reduction: remove supersets when subsets exist
        reduction_count_before_final = len(final_signatures_before_reduction)
        final_signatures = reduce_signatures_by_subsets(final_signatures_before_reduction, num_processes=args.num_processes)
        reduction_count_after_final = len(final_signatures)
        final_reduction_removed = reduction_count_before_final - reduction_count_after_final
        if final_reduction_removed > 0:
            logger.info(f"[Final Reduction] Removed {final_reduction_removed} superset signatures. "
                       f"Signatures: {reduction_count_before_final} -> {reduction_count_after_final}")

        logger.info("--- Process Complete ---")
        logger.info(f"Initial unique signatures generated: {len(all_valid_signatures)}")
        logger.info(f"Signatures removed due to FPs: {len(signatures_to_remove)}")
        logger.info(f"Final count of validated signatures (after reduction/inactive removal): {len(final_signatures)}")

        # Attack-type extraction fallback extracted to utils.signature_reporting

        attack_cols = get_attack_columns_for_report(args.file_type)
        signature_records = []
        #for sig_id, rule in final_signatures.items():
        signature_records_reversed = []  # For reverse-mapped signatures
        
        reverse_mapping = {}
        original_value_range_mapping = {}
        if category_mapping is not None and mapping_features is not None and mapping_features_original is not None:
            # Build reverse mapping once
            logger.info("Building reverse mapping for signature report...")
            reverse_mapping = build_reverse_mapping(category_mapping)
            logger.info(f"Reverse mapping built for {len(reverse_mapping)} features.")

            # Build original-value-range reverse mapping (group -> original value range string)
            logger.info("Building original value-range mapping for signature report...")
            original_value_range_mapping = build_original_value_range_mapping(
                category_mapping,
                mapping_features,
                mapping_features_original,
                prefer_interval_bounds=True,
                mapped_df=None # Global mapping info might not have a single mapped_df, keep as None
            )
            n_final = len(original_value_range_mapping)
            rev_final_sample = ""
            if n_final:
                fc = next(iter(original_value_range_mapping))
                gr = original_value_range_mapping[fc]
                one = next(iter(gr.items()), (None, "")) if gr else (None, "")
                rev_final_sample = f", sample {fc} group{one[0]}->{str(one[1])[:40]}"
            logger.info(f"[RevMap] Final: value-range mapping for {n_final} features{rev_final_sample}.")
        else:
            logger.info("Skipping global reverse mapping (turnmap uses per-turn mapping and raw range rules).")
        
        # Sort by signature ID to ensure consistent CSV output order
        sorted_final_signatures = sorted(final_signatures.items(), key=lambda x: x[0])
        total_sigs = len(final_signatures)
        
        # Calculate individual signature performance metrics (TP, FP, TN, FN, F1)
        logger.info(f"Calculating individual performance metrics for {total_sigs} signatures...")
        individual_metrics = {}
        if chunk_cache is not None and not chunk_cache.is_empty() and sorted_final_signatures:
            eval_label_col_metrics = "cluster" if getattr(args, "eval_target", "label") == "cluster" else "label"
            individual_metrics = compute_metrics_by_turn_range(
                chunk_cache,
                sorted_final_signatures,
                signature_turn_created,
                signature_turn_removed,
                eval_label_col_metrics,
                calculate_signature,
            )
            logger.info(f"Completed individual performance calculation for {len(individual_metrics)} signatures.")
        else:
            logger.warning("Chunk cache is empty or no signatures. Skipping individual performance calculation.")
        
        logger.info(f"Extracting attack types for {total_sigs} signatures...")
        attack_types_by_sig = try_extract_attack_types_c_ext(
            sorted_final_signatures,
            chunk_cache,
            attack_cols,
            args.evaluation_batch_size,
        )
        for sig_idx, (sig_id, rule) in enumerate(sorted_final_signatures, 1):
            if sig_idx % 100 == 0 or sig_idx == 1 or sig_idx == total_sigs:
                logger.info(f"  Processing signature {sig_idx}/{total_sigs}...")
            created_turn = signature_turn_created.get(sig_id, None)
            if isinstance(attack_types_by_sig, dict) and sig_id in attack_types_by_sig:
                attacks = attack_types_by_sig[sig_id]
            else:
                attacks = extract_attack_types_for_signature_from_cache(
                    rule,
                    chunk_cache,
                    attack_cols,
                    args.evaluation_batch_size,
                )
            
            # Get individual metrics for this signature
            metrics = individual_metrics.get(sig_id, {})
            
            # Original report (with group numbers)
            signature_records.append({
                'signature_rule': str(rule),
                'created_turn': created_turn,
                'attack_types': "|".join(sorted(attacks)),
                'tp': metrics.get('tp', 0),
                'fp': metrics.get('fp', 0),
                'tn': metrics.get('tn', 0),
                'fn': metrics.get('fn', 0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0)
            })
            
            # Reverse-mapped report (with ORIGINAL value ranges when possible)
            if _rule_looks_range(rule):
                reversed_rule = rule
            else:
                reversed_rule = reverse_map_rule_with_fallback(rule, original_value_range_mapping, reverse_mapping)
            if sig_idx == 1:
                s = str(reversed_rule)[:120]
                logger.info(f"[RevMap] Report sample (sig 1): {s}...")
            signature_records_reversed.append({
                'signature_rule': str(reversed_rule),
                'created_turn': created_turn,
                'attack_types': "|".join(sorted(attacks)),
                'tp': metrics.get('tp', 0),
                'fp': metrics.get('fp', 0),
                'tn': metrics.get('tn', 0),
                'fn': metrics.get('fn', 0),
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1_score', 0.0)
            })
            
        logger.info(f"Finished extracting attack types and metrics for all {total_sigs} signatures.")

        final_signatures_df = pd.DataFrame(signature_records)
        final_signatures_df_reversed = pd.DataFrame(signature_records_reversed)

        # The output directory is now the same as the run_dir for artifacts/checkpoints
        output_dir = run_dir

        # UPDATED: Add parameters to filenames for clarity
        param_str = params_str  # already includes _turneval

        # Save original report (with group numbers) - KEEP EXISTING
        output_filename = f"{args.file_type}_{args.file_number}_{param_str}_incremental_signatures_eex.csv"
        output_path = os.path.join(output_dir, output_filename)

        final_signatures_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Final signatures (group numbers) saved to: {output_path}")
        
        # Save reverse-mapped report (with original intervals/values) - NEW
        output_filename_reversed = f"{args.file_type}_{args.file_number}_{param_str}_incremental_signatures_eex_reversed.csv"
        output_path_reversed = os.path.join(output_dir, output_filename_reversed)

        final_signatures_df_reversed.to_csv(output_path_reversed, index=False, encoding='utf-8-sig')
        logger.info(f"Final signatures (original intervals/values) saved to: {output_path_reversed}")

        # --- PLOTTING and HISTORY CSV---
        if history:
            history_df = pd.DataFrame(history)

            performance_filename = f"{args.file_type}_{args.file_number}_{param_str}_performance_history_eex.csv"
            performance_path = os.path.join(output_dir, performance_filename)
            history_df.to_csv(performance_path, index=False, encoding='utf-8-sig')
            logger.info(f"Performance history saved to: {performance_path}")

            if plt:
                logger.info("Generating performance graph...")
                
                # Create a single figure and a primary axis for performance metrics
                fig, ax_perf = plt.subplots(figsize=(18, 8)) # Wider figure for better readability
                fig.suptitle(f'Incremental Signature Performance for {args.file_type}\n(support={args.min_support}, confidence={args.min_confidence})', fontsize=16)

                # Create a secondary axis for signature counts that shares the same x-axis
                ax_counts = ax_perf.twinx()

                x_labels = []
                x_ticks = []
                bar_width = 0.35

                # --- Plotting Loop for both lines and bars ---
                for i, row in history_df.iterrows():
                    turn = row['turn']
                    x_entry = i * 2
                    x_exit = i * 2 + 1

                    # --- 1. Plot Performance Lines on the primary axis (ax_perf) ---
                    # Plot Learning phase (solid line)
                    ax_perf.plot([x_entry, x_exit], [row['entry_recall'], row['exit_recall']], 'o-', color='blue', label='Recall (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_precision'], row['exit_precision']], 'x-', color='purple', label='Precision (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_f1'], row['exit_f1']], 's-', color='orange', label='F1-Score (Learning)' if i == 0 else "")
                    ax_perf.plot([x_entry, x_exit], [row['entry_accuracy'], row['exit_accuracy']], 'd-', color='green', label='Accuracy (Learning)' if i == 0 else "")
                    
                    # Plot Adaptation phase (dotted line)
                    if i < len(history_df) - 1:
                        next_row = history_df.iloc[i+1]
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_recall'], next_row['entry_recall']], 'o--', color='blue', alpha=0.5, label='Recall (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_precision'], next_row['entry_precision']], 'x--', color='purple', alpha=0.5, label='Precision (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_f1'], next_row['entry_f1']], 's--', color='orange', alpha=0.5, label='F1-Score (Adaptation)' if i == 0 else "")
                        ax_perf.plot([x_exit, x_exit + 1], [row['exit_accuracy'], next_row['entry_accuracy']], 'd--', color='green', alpha=0.5, label='Accuracy (Adaptation)' if i == 0 else "")

                    # --- 2. Plot Count Bars on the secondary axis (ax_counts) ---
                    # Position the bars in the middle of the entry-exit gap
                    bar_center = x_entry + 0.5
                    ax_counts.bar(bar_center - bar_width/2, row['generated'], bar_width, label='Generated' if i == 0 else "", color='green', alpha=0.6)
                    ax_counts.bar(bar_center + bar_width/2, row['removed'], bar_width, label='Removed' if i == 0 else "", color='red', alpha=0.6)

                    x_ticks.extend([x_entry, x_exit])
                    x_labels.extend([f"{turn}-entry", f"{turn}-exit"])

                # --- Formatting and Labels ---
                ax_perf.set_xticks(x_ticks)
                ax_perf.set_xticklabels(x_labels, rotation=45, ha='right')
                ax_perf.set_xlabel(f'Turn ({args.chunk_size}-row chunks)')
                ax_perf.set_ylabel('Metric Value (Recall, Precision, F1)')
                ax_perf.set_ylim(0, 1.05)
                ax_perf.grid(True, linestyle='--')

                ax_counts.set_ylabel('Signature Count (Generated/Removed)', color='gray')
                ax_counts.tick_params(axis='y', labelcolor='gray')
                # Ensure the bottom of the bar chart is at 0
                ax_counts.set_ylim(bottom=0)

                # Combine legends from both axes
                handles_perf, labels_perf = ax_perf.get_legend_handles_labels()
                handles_counts, labels_counts = ax_counts.get_legend_handles_labels()
                ax_perf.legend(handles_perf + handles_counts, labels_perf + labels_counts, loc='best')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                graph_dir = "../isv_graph/"
                if not os.path.exists(graph_dir):
                    try:
                        os.makedirs(graph_dir)
                    except OSError as e:
                        logger.error(f"Could not create graph directory {graph_dir}: {e}")
                        graph_dir = "."
                
                graph_filename = f"{args.file_type}_{args.file_number}_{param_str}_metrics_eex.jpg"
                graph_path = os.path.join(graph_dir, graph_filename)
                
                try:
                    plt.savefig(graph_path, format='jpg', dpi=150)
                    logger.info(f"Performance graph saved to: {graph_path}")
                except Exception as e:
                    logger.error(f"Failed to save graph: {e}")
            
            # --- NEW: Final Summary Printout ---
            final_stats = history_df.iloc[-1]
            logger.info("--- Final Summary ---")
            logger.info(f"Total Validated Signatures: {len(final_signatures)}")
            logger.info(f"Final Recall (at Turn {int(final_stats['turn'])}): {final_stats['exit_recall']:.4f}")
            logger.info(f"Final Precision (at Turn {int(final_stats['turn'])}): {final_stats['exit_precision']:.4f}")
            logger.info(f"Final F1-Score (at Turn {int(final_stats['turn'])}): {final_stats['exit_f1']:.4f}")
            logger.info(f"Final Accuracy (at Turn {int(final_stats['turn'])}): {final_stats.get('exit_accuracy', 0):.4f}")
            logger.info("--------------------")
        else:
            final_stats = None

        end_time = time.time()

        # --- NEW: Save concise final metrics CSV ---
        try:
            total_generation_time = end_time - start_time
            num_final_signatures = len(final_signatures)
            avg_conditions = float(np.mean([len(rule) for rule in final_signatures.values()])) if final_signatures else 0.0
            final_recall = final_stats['exit_recall'] if final_stats is not None else 0.0

            summary_record = {
                "time_to_generate_signatures_sec": total_generation_time,
                "num_signatures_final": num_final_signatures,
                "avg_conditions_per_signature": avg_conditions,
                "total_recall_final_turn_exit": final_recall
            }

            summary_df = pd.DataFrame([summary_record])
            summary_filename = f"{args.file_type}_{args.file_number}_{param_str}_summary_metrics.csv"
            summary_path = os.path.join(output_dir, summary_filename)
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            logger.info(f"Summary metrics saved to: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save summary metrics CSV: {e}")

        #end_time = time.time()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

        # --- Interval mapping debug flush (console only; no files) ---
        try:
            flush_interval_mapping_debug(top_payloads=30)
        except Exception as e:
            logger.warning(f"[IntervalDebug] Failed to flush interval debug buffer: {e}")

        # --- NaN-in-signature summary flush (console/log only; no files) ---
        try:
            if nan_sig_turn_stats:
                logger.info("[NaN-in-Signatures] Per-turn summary (generated/filtered/added_new):")
                for rec in nan_sig_turn_stats:
                    logger.info(
                        f"  Turn {rec['turn']}: "
                        f"gen {rec['generated_with_nan']}/{rec['generated_total']}, "
                        f"filtered {rec['filtered_with_nan']}/{rec['filtered_total']}, "
                        f"added_new {rec['added_new_with_nan']}/{rec['added_new_total']}, "
                        f"fallback={rec['is_fallback_rule']}"
                    )
            else:
                logger.info("[NaN-in-Signatures] No per-turn stats collected.")
        except Exception as e:
            logger.warning(f"[NaN-in-Signatures] Failed to print summary: {e}")

        # --- Optional final FP contribution summary log (printed at the end) ---
        if getattr(args, "save_turn_fp_contribution", False):
            try:
                if os.path.exists(turn_fp_contrib_csv):
                    fp_contrib_all = pd.read_csv(turn_fp_contrib_csv)
                    if not fp_contrib_all.empty:
                        top_n = max(1, int(getattr(args, "turn_fp_contribution_top_n", 5)))
                        logger.info(f"[FPContrib] Final per-turn Top-{top_n} creation-turn contributors (entry/exit):")
                        for turn_val in sorted(fp_contrib_all["turn"].dropna().unique().tolist()):
                            turn_part = fp_contrib_all[fp_contrib_all["turn"] == turn_val]
                            for phase_tag in ["entry", "exit"]:
                                phase_part = turn_part[turn_part["phase"] == phase_tag].copy()
                                if phase_part.empty:
                                    logger.info(f"[FPContrib] T{int(turn_val)} {phase_tag}: no rows.")
                                    continue
                                phase_part["signature_created_turn_num"] = pd.to_numeric(
                                    phase_part["signature_created_turn"], errors="coerce"
                                )
                                phase_part = phase_part[
                                    phase_part["signature_created_turn_num"].notna()
                                    & (phase_part["signature_created_turn_num"] >= 0)
                                    & (phase_part["fp_alert_events"] > 0)
                                ]
                                if phase_part.empty:
                                    logger.info(f"[FPContrib] T{int(turn_val)} {phase_tag}: no FP events.")
                                    continue
                                top_part = phase_part.sort_values(
                                    ["fp_alert_events", "fp_alert_share"], ascending=[False, False]
                                ).head(top_n)
                                top_text = ", ".join(
                                    [
                                        f"created@T{int(r['signature_created_turn_num'])}: "
                                        f"{int(r['fp_alert_events'])} ({float(r['fp_alert_share']):.2%})"
                                        for _, r in top_part.iterrows()
                                    ]
                                )
                                logger.info(f"[FPContrib] T{int(turn_val)} {phase_tag}: {top_text}")
                    else:
                        logger.info("[FPContrib] CSV is empty. No final summary to print.")
                else:
                    logger.info(f"[FPContrib] CSV not found: {turn_fp_contrib_csv}")
            except Exception as e:
                logger.warning(f"[FPContrib] Failed to print final Top-N summary: {e}")

        # --- Optional final detailed FP-priority CSVs (cohort and rule deletion priority) ---
        if getattr(args, "save_turn_fp_contribution", False):
            try:
                if os.path.exists(turn_fp_rule_phase_csv):
                    report_result = generate_fp_priority_reports_from_csv(
                        fp_rule_phase_csv_path=turn_fp_rule_phase_csv,
                        fp_top_cohort_rule_details_csv=fp_top_cohort_rule_details_csv,
                        fp_rule_delete_priority_csv=fp_rule_delete_priority_csv,
                        fp_rule_delete_turn_impact_csv=fp_rule_delete_turn_impact_csv,
                        cohort_top_n=max(1, int(getattr(args, "fp_cohort_top_n", 5))),
                        delete_top_k=max(1, int(getattr(args, "fp_delete_priority_top_k", 50))),
                        chunksize=200000,
                    )
                    if report_result.get("ok"):
                        logger.info(f"[FPContrib] Top-cohort rule details saved to: {fp_top_cohort_rule_details_csv}")
                        logger.info(f"[FPContrib] Rule delete-priority saved to: {fp_rule_delete_priority_csv}")
                        logger.info(f"[FPContrib] Rule delete-priority per-turn impact saved to: {fp_rule_delete_turn_impact_csv}")
                    else:
                        logger.info(f"[FPContrib] Skip detailed FP-priority CSVs: {report_result.get('reason', 'unknown')}")
            except Exception as e:
                logger.warning(f"[FPContrib] Failed to build final detailed FP-priority CSVs: {e}")

    except Exception as e:
        # This is a critical logging block. If the main process fails for any reason
        # (including potential memory issues leading to other errors), this will be the
        # last thing logged before worker processes might start failing with BrokenPipeError.
        logger.error("="*80)
        logger.error("! AN UNHANDLED EXCEPTION OCCURRED IN THE MAIN PROCESS !")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Error Details: {e}")
        logger.error("This is very likely the ROOT CAUSE of the process terminating unexpectedly.")
        logger.error("If you see a flood of 'BrokenPipeError' messages after this, they are a SYMPTOM, not the cause.")
        logger.error("The main process died, and the worker processes could no longer communicate with it.")
        logger.error("Please check the traceback below for the actual error.")
        logger.error("="*80)
        
        # It's crucial to re-raise the exception to see the full traceback of the root cause.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Incrementally generate and validate signatures from a dataset.")
    parser.add_argument('--file_type', type=str, default="MiraiBotnet", help="Type of the dataset file.")
    parser.add_argument('--file_number', type=int, default=1, help="Number of the dataset file.")
    parser.add_argument('--association_method', type=str, default='rarm', help="Association rule algorithm to use.")
    parser.add_argument('--min_support', type=float, default=0.3, help="Minimum support for association rule mining.")
    parser.add_argument('--normal_min_support', type=float, default=None, help="Optional override for normal-data filtering support threshold (defaults to min_support).")
    parser.add_argument('--min_confidence', type=float, default=0.8, help="Minimum confidence for association rule mining.")
    # MODIFIED: Default to None to detect if the user has provided a value.
    parser.add_argument('--num_processes', type=int, default=None, help="Number of processes to use for parallel tasks. Defaults to all available cores.")
    parser.add_argument('--chunk_size', type=int, default=500, help="Number of rows to process in each incremental turn.")
    parser.add_argument('--cstemporal', action='store_true', help="Use dataset-specific temporal window preset instead of numeric chunk_size. File names will use 'tem' for cs.")
    parser.add_argument('--itemset_limit', type=int, default=10000000, help="Safety limit for frequent itemsets to prevent memory overflow before rule generation.")
    parser.add_argument('--n_splits', type=int, default=40, help="Number of splits to use for dynamic interval mapping. Default is 40.")
    parser.add_argument('--max_level', type=int, default=None, help="Global max_level override for association mining. If omitted, uses LEVEL_LIMITS_BY_FILE_TYPE.")
    parser.add_argument('--signature_batch_size', type=int, default=20000, help="Batch size for validating signatures to conserve memory.")
    parser.add_argument('--normal_data_batch_size', type=int, default=30000, help="Batch size for splitting the turn's normal data to conserve memory during filtering. If not set, normal data is not batched.")
    parser.add_argument('--evaluation_batch_size', type=int, default=20000, help="Batch size for processing the full dataset during final performance evaluation to conserve memory.")
    parser.add_argument('--rule_spool_chunk_size', type=int, default=None, help="Number of rules to store per chunk when spooling to disk before filtering (auto if omitted).")
    parser.add_argument('--precision_underlimit', type=float, default=None, help="Per-turn precision lower bound for signatures (cluster-based).")
    parser.add_argument('--signature_ea', type=int, default=None, help="Top-N signatures to consider when applying precision_underlimit.")
    parser.add_argument('--precision_underlimit_use_temporal', type=int, choices=[0, 1], default=0, help="Use temporal (rolling/cumulative) precision filtering instead of current-turn only. Default=0 (OFF).")
    parser.add_argument('--precision_underlimit_temporal_mode', type=str, choices=['rolling', 'cumulative'], default='rolling', help="Temporal precision mode when --precision_underlimit_use_temporal=1.")
    parser.add_argument('--precision_underlimit_temporal_window', type=int, default=3, help="Window size (turn count) for temporal precision checks (recommended 3~5).")
    parser.add_argument('--precision_underlimit_keep_no_alert', type=int, choices=[0, 1, 2], default=0, help="No-alert handling mode for precision_underlimit: 0=drop no-alert, 1=keep no-alert (optional streak drop), 2=sleep/stand mode (no-alert underlimit-fail rules go to sleep, can stand again in later turns). Default=0.")
    parser.add_argument('--precision_underlimit_no_alert_max_streak', type=int, default=0, help="Drop rule if no-alert streak reaches this value when keep_no_alert=1. 0 disables streak-based no-alert drop.")
    parser.add_argument('--precision_underlimit_auto_adjust', type=int, choices=[0, 1], default=0, help="Auto-adjust precision_underlimit per turn using cluster-based precision/recall only.")
    parser.add_argument('--precision_underlimit_auto_adjust_factor', type=float, default=0.9, help="Multiplier applied to precision_underlimit each retry (e.g., 0.9). Must be in (0,1).")
    parser.add_argument('--precision_underlimit_auto_adjust_max_retries', type=int, default=10, help="Max retry count per turn for auto-adjust.")
    parser.add_argument('--precision_underlimit_auto_adjust_start_threshold', type=float, default=0.9, help="Start auto-adjust only when base cluster precision/recall is below this threshold (mode trigger).")
    parser.add_argument('--precision_underlimit_auto_adjust_recall_floor', type=float, default=0.95, help="Counterpart floor: when raising precision, stop if cluster-recall falls below this value.")
    parser.add_argument('--precision_underlimit_auto_adjust_precision_floor', type=float, default=0.95, help="Counterpart floor: when raising recall, stop if cluster-precision falls below this value.")
    parser.add_argument('--precision_underlimit_auto_adjust_min', type=float, default=0.0, help="Lower bound for auto-adjusted precision_underlimit.")
    parser.add_argument('--prune_signatures', action='store_true', help="If set, enables the signature pruning (subsumption) process to remove redundant rules.")
    parser.add_argument('--prune_coverage_threshold', type=float, default=0.9, help="Coverage threshold for the signature pruning process.")
    parser.add_argument('--merge_signatures', action='store_true', help="If set, enables merging of similar, infrequent signatures. (Requires --prune_signatures)")
    parser.add_argument('--merge_infrequent_threshold', type=int, default=5, help="TP count at or below which a rule is considered infrequent and eligible for merging.")
    parser.add_argument('--signature_organize', action='store_true', help="A shorthand to enable both --prune_signatures and --merge_signatures.")
    parser.add_argument('--apply_turn_reduction_removal', action='store_true', help="Apply reduction removals to the signature set each turn (default: record only).")
    parser.add_argument('--apply_turn_inactive_removal', action='store_true', help="Apply inactive removals to the signature set each turn (default: record only).")
    parser.add_argument('--save_artifacts', action='store_true', help="If set, save intermediate itemsets and rules for debugging.")
    parser.add_argument('--save_attackwise_turn_recall', action='store_true', help="Record full-dataset attack-wise recall (entry/exit) into history CSV columns. This is reporting-only and does not affect rule generation.")
    # --- Dynamic Support Arguments ---
    parser.add_argument('--dynamic_support', action='store_true', help="Enable adaptive support thresholding to prevent memory overflow.")
    parser.add_argument('--itemset_count_threshold', type=int, default=500000, help="Itemset count limit at any level that triggers dynamic support adjustment.")
    parser.add_argument('--support_increment_factor', type=float, default=1.2, help="Factor by which to multiply min_support when the threshold is exceeded (e.g., 1.2 for a 20% increase).")
    # --- Auto Support Tuning ---
    parser.add_argument('--auto_support', action='store_true', help="Auto-adjust min_support based on per-turn rule count targets.")
    parser.add_argument('--auto_support_target_min', type=int, default=1000, help="Lower bound for per-turn rule count target.")
    parser.add_argument('--auto_support_target_max', type=int, default=5000, help="Upper bound for per-turn rule count target.")
    parser.add_argument('--auto_support_step', type=float, default=1.2, help="Multiplicative step for min_support adjustment.")
    parser.add_argument('--auto_support_min', type=float, default=1e-6, help="Lower bound for auto min_support.")
    parser.add_argument('--auto_support_max', type=float, default=0.9, help="Upper bound for auto min_support.")
    # --- Turn-internal adaptive support retry (same turn, before filtering) ---
    parser.add_argument('--adaptive_support_retry', action='store_true', help="Retry association mining within the same turn when shallow level is detected (default: raises min_support).")
    parser.add_argument('--adaptive_support_retry_max_retries', type=int, default=7, help="Maximum retry count inside one turn when adaptive_support_retry is enabled.")
    parser.add_argument('--adaptive_support_retry_factor', type=float, default=1.3, help="Multiplicative factor for min_support on each retry (e.g., 1.3 = +30%).")
    parser.add_argument('--adaptive_support_retry_target_level', type=int, default=None, help="Target level for adaptive retry. If omitted, uses the turn's effective max_level.")
    parser.add_argument('--adaptive_support_retry_rule_threshold', type=int, default=400000, help="Retry trigger: only retry when generated rule count is at least this value and reached level is below target.")
    parser.add_argument('--adaptive_support_retry_itemset_l1_threshold', type=int, default=10000, help="Retry trigger (optional): L1 frequent itemset count threshold for the current support. 0 disables this trigger.")
    parser.add_argument('--adaptive_support_retry_candidate_l1_threshold', type=int, default=10000, help="Retry trigger (optional): estimated L1 potential candidate count threshold (C(n_l1,2)). 0 disables this trigger.")
    parser.add_argument('--adaptive_support_retry_bidirectional', action='store_true', help="Enable bidirectional retry: raise min_support on high-complexity signals, and lower it on low-L1 signals.")
    parser.add_argument('--adaptive_support_retry_low_itemset_l1_threshold', type=int, default=0, help="Low-L1 trigger (optional, bidirectional mode): retry downward when L1 itemset count is at most this value. 0 disables.")
    parser.add_argument('--adaptive_support_retry_low_candidate_l1_threshold', type=int, default=0, help="Low-L1-candidate trigger (optional, bidirectional mode): retry downward when estimated L1 candidate count is at most this value. 0 disables.")
    parser.add_argument('--adaptive_support_retry_min_support', type=float, default=1e-10, help="Lower bound for min_support during adaptive retries (used for bidirectional downward retries).")
    parser.add_argument('--adaptive_support_retry_on_level_shortfall', action='store_true', help="Retry when max_level_reached is below target_level (default: OFF).")
    parser.add_argument('--adaptive_support_retry_max_support', type=float, default=0.9, help="Upper bound for min_support during adaptive retries.")
    # --- Separability Diagnostics ---
    parser.add_argument('--separability', action='store_true', help="Log feature separability per turn using cluster labels.")
    parser.add_argument('--separability_top_n', type=int, default=20, help="Top-N features to log for separability diagnostics.")
    parser.add_argument('--separability_sample_size', type=int, default=20000, help="Sample size for separability diagnostics.")
    parser.add_argument('--separability_max_unique', type=int, default=200, help="Max unique values for categorical MI computation.")
    # --- Optional separability-based feature selection for association mining (default OFF) ---
    add_separation_filter_args(parser)
    parser.add_argument('--negative_filtering', action='store_true', help="Enable negative-aware filtering using P(rule|normal) thresholds.")
    parser.add_argument('--negative_filter_threshold', type=float, default=0.05, help="Maximum allowed P(rule|normal) when negative-aware filtering is enabled.")
    parser.add_argument('--strict_normal_zero', action='store_true', help="Require zero matches in normal data for rule acceptance.")
    parser.add_argument('--reset', action='store_true', help="If set, deletes the checkpoint and artifact directory for the given parameters and exits.")
    parser.add_argument('--mask_dominant_cols', action='store_true', default=True, help="Mask near-constant columns (freq>0.99) from support counting and re-attach them to generated rules.")
    parser.add_argument('--dominant_freq_threshold', type=float, default=0.99, help="Frequency threshold to detect dominant (near-constant) columns for masking.")
    parser.add_argument('--dominant_min_support', type=float, default=None, help="Optional min_support override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_min_confidence', type=float, default=None, help="Optional min_confidence override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_normal_min_support', type=float, default=None, help="Optional normal_min_support override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_level', type=int, default=None, help="Optional max_level override when masking is skipped due to all features being dominant.")
    parser.add_argument('--dominant_attach_filter', action='store_true', help="Exclude dominant cols from re-attachment if they are near-constant in both anomaly and normal.")
    parser.add_argument('--dominant_attach_threshold', type=float, default=0.98, help="Threshold for dominant_attach_filter (exclude if anom/norm freq both >= this).")
    parser.add_argument('--rule_make_label', action='store_true', help="Generate rules using label (label=1) instead of cluster.")
    parser.add_argument('--eval_target', type=str, choices=['label', 'cluster'], default='label', help="Performance evaluation target: 'label' (default) or 'cluster'.")
    parser.add_argument('--range_match_engine', type=str, choices=['bool', 'bitset'], default='bitset', help="Range matcher engine: bool or bitset (default: bitset).")
    parser.add_argument('--range_match_jit', type=str, choices=['none', 'numba'], default='numba', help="JIT engine for range matcher when not using C extension (default: numba).")
    parser.add_argument('--no_rule_eval_c', action='store_true', help="Disable C extension for rule evaluation kernel; use Numba/Python path instead.")
    parser.add_argument('--run_turn_start', type=int, default=None, help="Start turn (inclusive) to run experiment")
    parser.add_argument('--run_turn_end', type=int, default=None, help="End turn (inclusive) to run experiment")
    # --- NRA/UFP/HAF based FP removal parameters ---
    parser.add_argument('--use_fp_metrics', action='store_true', help="Use NRA/UFP/HAF based FP removal instead of simple normal alert detection")
    parser.add_argument('--fp_t0_nra', type=int, default=60, help='Time window (seconds) for NRA calculation')
    parser.add_argument('--fp_n0_nra', type=int, default=20, help='Normalization factor for NRA calculation')
    parser.add_argument('--fp_lambda_haf', type=float, default=100.0, help='Lambda parameter for HAF score calculation')
    parser.add_argument('--fp_lambda_ufp', type=float, default=10.0, help='Lambda parameter for UFP score calculation')
    parser.add_argument('--fp_combine_method', type=str, default='max', choices=['max', 'weighted_sum'], help='Method to combine NRA, HAF, UFP scores')
    parser.add_argument('--fp_belief_threshold', type=float, default=0.5, help='Threshold for FP belief score to classify a signature as FP')
    parser.add_argument('--fp_superset_strictness', type=float, default=0.9, help='Strictness multiplier for superset FP detection')
    parser.add_argument('--disable_fp_removal', action='store_true', help='Disable FP removal entirely. Signatures will not be removed due to false positives.')
    parser.add_argument('--fp_reduce_supersets', action='store_true', help='Remove FP-heavy rules if they are supersets of another rule.')
    parser.add_argument('--fp_replace_by_coverage', action='store_true', help='Replace FP-heavy rules if lower-FP rules cover sufficient TP.')
    parser.add_argument('--fp_replace_tp_coverage_threshold', type=float, default=0.8, help='TP coverage threshold for replacement (default=0.8).')
    parser.add_argument('--save_turn_fp_contribution', action='store_true', help='Save per-turn FP contribution CSV (entry/exit) grouped by signature creation turn. Duplicate FP alert events are counted.')
    parser.add_argument('--turn_fp_contribution_top_n', type=int, default=5, help='Top-N creation turns to print in final FP contribution summary log (used when --save_turn_fp_contribution is enabled).')
    parser.add_argument('--fp_cohort_top_n', type=int, default=5, help='Top-N FP-heavy creation turns (cohorts) to export detailed rule CSV for (used with --save_turn_fp_contribution).')
    parser.add_argument('--fp_delete_priority_top_k', type=int, default=50, help='Top-K delete-priority rules to export per-turn recall impact CSV for (used with --save_turn_fp_contribution).')

    cli_args = parser.parse_args()

    # --- NEW: Handle the --signature_organize shorthand ---
    if cli_args.signature_organize:
        cli_args.prune_signatures = True
        cli_args.merge_signatures = True

    apply_separation_cli_postprocess(cli_args)

    # If --num_processes is not provided by the user (i.e., it's None), default to all available cores.
    if cli_args.num_processes is None:
        try:
            # Use os.cpu_count() which is recommended for getting the number of CPUs.
            cpu_count = os.cpu_count()
            cli_args.num_processes = cpu_count
            logger.info(f"--num_processes not set, defaulting to all available cores: {cpu_count}")
        except NotImplementedError:
            logger.warning("os.cpu_count() is not implemented. Defaulting to 4 processes.")
            cli_args.num_processes = 4 # Fallback

    main(cli_args) 
