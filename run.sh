#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-./config.toml}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_Lite}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: $CONFIG_FILE not found." >&2
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python not found (looked for: $PYTHON_BIN)." >&2
  exit 1
fi

# ---- Parse config.toml: target_ids, backend, model, and API keys ----
TARGET_IDS=()
BACKEND=""
MODEL=""
OPENAI_API_KEY_FROM_CONFIG=""
ANTHROPIC_API_KEY_FROM_CONFIG=""
DEEPSEEK_API_KEY_FROM_CONFIG=""

while IFS='=' read -r key value; do
  case "$key" in
    TARGET_ID)
      TARGET_IDS+=("$value")
      ;;
    BACKEND)
      BACKEND="$value"
      ;;
    MODEL)
      MODEL="$value"
      ;;
    OPENAI_API_KEY)
      OPENAI_API_KEY_FROM_CONFIG="$value"
      ;;
    ANTHROPIC_API_KEY)
      ANTHROPIC_API_KEY_FROM_CONFIG="$value"
      ;;
    DEEPSEEK_API_KEY)
      DEEPSEEK_API_KEY_FROM_CONFIG="$value"
      ;;
  esac
done < <("$PYTHON_BIN" - "$CONFIG_FILE" << 'PY'
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
if not cfg_path.exists():
    print(f"ERROR: {cfg_path} does not exist.", file=sys.stderr)
    sys.exit(1)

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli
    except ImportError:
        print("ERROR: Need tomllib (3.11+) or tomli installed.", file=sys.stderr)
        sys.exit(1)

data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))

# Parse target_ids (required)
tids = data.get("target_ids")
if not isinstance(tids, list) or not all(isinstance(t, str) for t in tids):
    print('ERROR: config.toml must define: target_ids = ["django__django-10914", ...]', file=sys.stderr)
    sys.exit(1)

for tid in tids:
    print(f"TARGET_ID={tid}")

# Parse backend and model (optional, will use defaults if not specified)
backend = data.get("backend", "openai")
model = data.get("model", "gpt-4o-2024-05-13")
print(f"BACKEND={backend}")
print(f"MODEL={model}")

# Parse API keys (optional)
api_key = data.get("openai_api_key")
if isinstance(api_key, str) and api_key:
    print(f"OPENAI_API_KEY={api_key}")

api_key = data.get("anthropic_api_key")
if isinstance(api_key, str) and api_key:
    print(f"ANTHROPIC_API_KEY={api_key}")

api_key = data.get("deepseek_api_key")
if isinstance(api_key, str) and api_key:
    print(f"DEEPSEEK_API_KEY={api_key}")
PY
)

if [[ ${#TARGET_IDS[@]} -eq 0 ]]; then
  echo "ERROR: No target_ids found in $CONFIG_FILE." >&2
  exit 1
fi

# Set defaults if not specified in config
BACKEND="${BACKEND:-openai}"
MODEL="${MODEL:-gpt-4o-2024-05-13}"

# Export appropriate API key based on backend
if [[ -n "$OPENAI_API_KEY_FROM_CONFIG" ]]; then
  export OPENAI_API_KEY="$OPENAI_API_KEY_FROM_CONFIG"
fi

if [[ -n "$ANTHROPIC_API_KEY_FROM_CONFIG" ]]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_FROM_CONFIG"
fi

if [[ -n "$DEEPSEEK_API_KEY_FROM_CONFIG" ]]; then
  export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY_FROM_CONFIG"
fi

# ---- Static environment ----
export PROJECT_FILE_LOC="${PROJECT_FILE_LOC:-/proj/arise/arise/hj2742/AgentlessData/repo_structure/repo_structures}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Verify API key is set for the selected backend
case "$BACKEND" in
  openai)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "ERROR: OPENAI_API_KEY is not set (required for backend=openai)." >&2
      exit 1
    fi
    ;;
  anthropic)
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "ERROR: ANTHROPIC_API_KEY is not set (required for backend=anthropic)." >&2
      exit 1
    fi
    ;;
  deepseek)
    if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
      echo "ERROR: DEEPSEEK_API_KEY is not set (required for backend=deepseek)." >&2
      exit 1
    fi
    ;;
  vertexai)
    if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]] || [[ -z "${GOOGLE_CLOUD_LOCATION:-}" ]]; then
      echo "WARNING: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION should be set for backend=vertexai." >&2
      echo "         Using defaults: GOOGLE_CLOUD_PROJECT=triangulate-396717, GOOGLE_CLOUD_LOCATION=us-central1" >&2
    fi
    ;;
  *)
    echo "ERROR: Unknown backend: $BACKEND" >&2
    exit 1
    ;;
esac

echo "Using backend: $BACKEND, model: $MODEL"

# ---- Main loop over target_ids ----
for target_id in "${TARGET_IDS[@]}"; do
  repo_name="${target_id%%__*}"

  echo
  echo "========================================="
  echo "Processing target_id = ${target_id} (repo: ${repo_name})"
  echo "========================================="

  # 1) file-level localization
  python agentless/fl/localize.py --file_level \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --output_folder results/swe-bench-lite/file_level \
    --num_threads 10 \
    --skip_existing \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 1/16: file_level localization completed"

  # 2) file-level irrelevant localization
  python agentless/fl/localize.py --file_level \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --output_folder results/swe-bench-lite/file_level_irrelevant \
    --num_threads 10 \
    --skip_existing \
    --target_id "${target_id}" \
    --irrelevant
  echo "[${repo_name}] Step 2/16: file_level irrelevant localization completed"

  # 3) retrieval
  python agentless/fl/retrieve.py --index_type simple \
    --filter_type given_files \
    --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
    --output_folder results/swe-bench-lite/retrievel_embedding \
    --persist_dir embedding/swe-bench_simple \
    --num_threads 10 \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 3/16: retrieval completed"

  # 4) combine
  python agentless/fl/combine.py \
    --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
    --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
    --top_n 3 \
    --output_folder results/swe-bench-lite/file_level_combined
  echo "[${repo_name}] Step 4/16: combine file_level + retrieval completed"

  # 5) related-level localization
  python agentless/fl/localize.py --related_level \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --output_folder results/swe-bench-lite/related_elements \
    --top_n 3 \
    --compress_assign \
    --compress \
    --start_file results/swe-bench-lite/file_level_combined/combined_locs.jsonl \
    --num_threads 10 \
    --skip_existing \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 5/16: related_level localization completed"

  # 6) fine-grain line-level
  python agentless/fl/localize.py --fine_grain_line_level \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --output_folder results/swe-bench-lite/edit_location_samples \
    --top_n 3 \
    --compress \
    --temperature 0.8 \
    --num_samples 4 \
    --start_file results/swe-bench-lite/related_elements/loc_outputs.jsonl \
    --num_threads 10 \
    --skip_existing \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 6/16: fine_grain_line_level localization completed"

  # 7) merge
  python agentless/fl/localize.py --merge \
    --output_folder results/swe-bench-lite/edit_location_individual \
    --top_n 3 \
    --num_samples 4 \
    --start_file results/swe-bench-lite/edit_location_samples/loc_outputs.jsonl \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 7/16: merge edit locations completed"

  # 8) repair (all 4 samples as in README)
  python agentless/repair/repair.py \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --loc_file results/swe-bench-lite/edit_location_individual/loc_merged_0-0_outputs.jsonl \
    --output_folder results/swe-bench-lite/repair_sample_1 \
    --loc_interval \
    --top_n=3 \
    --context_window=10 \
    --max_samples 10 \
    --cot \
    --diff_format \
    --gen_and_process \
    --num_threads 2 \
    --target_id "${target_id}"

  for i in {1..3}; do
    python agentless/repair/repair.py \
      --backend "${BACKEND}" \
      --model "${MODEL}" \
      --loc_file "results/swe-bench-lite/edit_location_individual/loc_merged_${i}-${i}_outputs.jsonl" \
      --output_folder "results/swe-bench-lite/repair_sample_$((i+1))" \
      --loc_interval \
      --top_n=3 \
      --context_window=10 \
      --max_samples 10 \
      --cot \
      --diff_format \
      --gen_and_process \
      --num_threads 2 \
      --target_id "${target_id}"
  done
  echo "[${repo_name}] Step 8/16: repair generation completed"

  # 9) generate passing regression tests
  python agentless/test/run_regression_tests.py \
    --run_id generate_regression_tests \
    --output_file results/swe-bench-lite/passing_tests.jsonl \
    --instance_ids "${target_id}"
  echo "[${repo_name}] Step 9/16: initial regression tests completed"

  # 10) select regression tests
  python agentless/test/select_regression_tests.py \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --passing_tests results/swe-bench-lite/passing_tests.jsonl \
    --output_folder results/swe-bench-lite/select_regression
  echo "[${repo_name}] Step 10/16: regression test selection completed"

  # 11) run regression tests against repair samples (repair_sample_1)
  folder="results/swe-bench-lite/repair_sample_1"
  run_id_prefix="$(basename "${folder}")"

  for num in {0..9}; do
    python agentless/test/run_regression_tests.py \
      --regression_tests results/swe-bench-lite/select_regression/output.jsonl \
      --predictions_path="${folder}/output_${num}_processed.jsonl" \
      --run_id="${run_id_prefix}_regression_${num}" \
      --num_workers 10
  done
  echo "[${repo_name}] Step 11/16: regression tests on repair candidates completed"

  # 12) generate reproduction tests (initial, no select)
  python agentless/test/generate_reproduction_tests.py \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --max_samples 40 \
    --output_folder results/swe-bench-lite/reproduction_test_samples \
    --num_threads 10
  echo "[${repo_name}] Step 12/16: initial reproduction test generation completed"

  # 13) run reproduction tests (generation filter), matching README loops
  for st in {0..36..4}; do
    en=$((st + 3))
    echo "Processing ${st} to ${en}"
    for num in $(seq "$st" "$en"); do
      echo "Processing ${num}"
      python agentless/test/run_reproduction_tests.py \
        --run_id="reproduction_test_generation_filter_sample_${num}" \
        --test_jsonl="results/swe-bench-lite/reproduction_test_samples/output_${num}_processed_reproduction_test.jsonl" \
        --num_workers 6 \
        --testing
    done &
  done
  wait

  for st in {0..36..4}; do
    en=$((st + 3))
    echo "Processing ${st} to ${en}"
    for num in $(seq "$st" "$en"); do
      echo "Processing ${num}"
      python agentless/test/run_reproduction_tests.py \
        --run_id="reproduction_test_generation_filter_sample_${num}" \
        --test_jsonl="results/swe-bench-lite/reproduction_test_samples/output_${num}_processed_reproduction_test.jsonl" \
        --num_workers 6 \
        --testing
    done &
  done
  wait
  echo "[${repo_name}] Step 13/16: reproduction test filtering completed"

  # 14) generate reproduction tests with selection
  python agentless/test/generate_reproduction_tests.py \
    --backend "${BACKEND}" \
    --model "${MODEL}" \
    --max_samples 40 \
    --output_folder results/swe-bench-lite/reproduction_test_samples \
    --output_file reproduction_tests.jsonl \
    --select \
    --target_id "${target_id}"
  echo "[${repo_name}] Step 14/16: reproduction test selection completed"

  # 15) run reproduction tests against repair samples (using selected tests)
  for num in {0..9}; do
    python agentless/test/run_reproduction_tests.py \
      --test_jsonl results/swe-bench-lite/reproduction_test_samples/reproduction_tests.jsonl \
      --predictions_path="${folder}/output_${num}_processed.jsonl" \
      --run_id="${run_id_prefix}_reproduction_${num}" \
      --num_workers 10
  done
  echo "[${repo_name}] Step 15/16: reproduction tests on repair candidates completed"

  # 16) rerank patches across all 4 repair_sample_* folders
  python agentless/repair/rerank.py \
    --patch_folder results/swe-bench-lite/repair_sample_1/,results/swe-bench-lite/repair_sample_2/,results/swe-bench-lite/repair_sample_3/,results/swe-bench-lite/repair_sample_4/ \
    --num_samples 40 \
    --deduplicate \
    --regression \
    --reproduction
  echo "[${repo_name}] Step 16/16: patch reranking completed"

  echo
  echo "✅ Done with target_id = ${target_id} (repo: ${repo_name})"
done

echo
echo "🎉 All target_ids completed."
