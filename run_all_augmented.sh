#!/usr/bin/env bash
# Runs run_code.sh (augmented, AUGMENT=True) sequentially once per cardinality
# type (est, dd, act, wj) for a single held-out database. Everything else is
# whatever is already configured inside run_code.sh -- only CARDINALITY_TYPE
# and TEST_DB are overridden here.
#
# Usage:
#   tmux new -s aug_name
#   epochs test - 100 and 200 epochs epoch_test and epoch_test1
#   TEST_DB=carcinogenesis bash run_all_augmented.sh
#   # or just edit TEST_DB below and run: bash run_all_augmented.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_code.sh"

TEST_DB="${TEST_DB:-basketball}"  #? Held-out database, e.g. basketball, consumer, carcinogenesis, ...
                                   #? Override at runtime: TEST_DB=consumer bash run_all_augmented.sh
DEVICE="${DEVICE:-0}"              #? Which GPU to run on.
                                   #? Override at runtime: DEVICE=1 bash run_all_augmented.sh
EPOCHS="${EPOCHS:-50}"            #? Number of training epochs.
                                   #? Override at runtime: EPOCHS=50 bash run_all_augmented.sh

CARDINALITY_TYPES=(est dd act wj)

declare -A exit_codes=()

for card_type in "${CARDINALITY_TYPES[@]}"; do
    echo "=== $(date) Starting test_db=$TEST_DB cardinality_type=$card_type (cuda:$DEVICE) ==="
    CARDINALITY_TYPE="$card_type" TEST_DB="$TEST_DB" DEVICE="$DEVICE" EPOCHS="$EPOCHS" bash "$RUN_SCRIPT"
    exit_codes["$card_type"]="$?"
    echo "=== $(date) Finished cardinality_type=$card_type (exit ${exit_codes[$card_type]}) ==="
done

echo
echo "=== SUMMARY (test_db=$TEST_DB) ==="
overall_exit_code=0
for card_type in "${CARDINALITY_TYPES[@]}"; do
    code="${exit_codes[$card_type]}"
    status="OK"
    if [[ "$code" -ne 0 ]]; then
        status="FAILED (exit $code)"
        overall_exit_code=1
    fi
    echo "  $card_type: $status"
done

exit "$overall_exit_code"