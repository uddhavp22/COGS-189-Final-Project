#!/usr/bin/env bash
# Download and extract all MLSPred-Bench preprocessed data from OSF
# Project: https://osf.io/2j7c8/?view_only=cadb4371a57f4a06a51f2d7e8b00e4db
#
# Benchmark 11 has no training split on OSF (validation only — by design).
# Test splits for all 12 benchmarks are in the combined zip and extracted
# automatically into each bmrkXX/ directory.

set -euo pipefail

TOKEN="cadb4371a57f4a06a51f2d7e8b00e4db"
BASE="https://osf.io/download"
ROOT="/teamspace/studios/this_studio/Seizure_Prediction/data"

# ── helpers ───────────────────────────────────────────────────

dl() {
    local dest="$1" url="$2"
    if [ -f "$dest" ]; then
        echo "[SKIP] $(basename "$dest") already exists"
    else
        echo "[DL]   $(basename "$dest")"
        wget -q --show-progress -c -O "$dest" "${url}?view_only=${TOKEN}"
    fi
}

# Unzip a benchmark zip into its directory, then delete the zip.
# Safe to call for train.zip and valid.zip independently — tracks by zip existence.
unzip_into() {
    local dir="$1" zip="$2"
    if [ ! -f "$zip" ]; then
        echo "[SKIP] $(basename "$zip") already extracted"
    else
        echo "[UNZIP] $(basename "$zip") → $(basename "$dir")/"
        unzip -j -q "$zip" -d "$dir"
        rm -f "$zip"   # remove zip to save space after extraction
    fi
}

# ── Benchmark 01 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk01"
dl "$ROOT/bmrk01/train_values.hdf5"  "$BASE/z7d2q"
dl "$ROOT/bmrk01/train_labels.csv"   "$BASE/myjea"
dl "$ROOT/bmrk01/valid_values.hdf5"  "$BASE/wsqz6"
dl "$ROOT/bmrk01/valid_labels.csv"   "$BASE/p6mqv"

# ── Benchmark 02 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk02"
dl "$ROOT/bmrk02/train_values.hdf5"  "$BASE/677ffd8f1cad2eeb8e31fe37"
dl "$ROOT/bmrk02/train_labels.csv"   "$BASE/677ffb40ee66904619809bb2"
dl "$ROOT/bmrk02/valid_values.hdf5"  "$BASE/677ffe6803f17d4c5c34d57f"
dl "$ROOT/bmrk02/valid_labels.csv"   "$BASE/677ffb13a0b7e18cdf31f5b5"

# ── Benchmark 03 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk03"
dl "$ROOT/bmrk03/train_values.hdf5"  "$BASE/677ffd8c0b53cdda5334d156"
dl "$ROOT/bmrk03/train_labels.csv"   "$BASE/677ffaf7a0b7e18cdf31f59c"
dl "$ROOT/bmrk03/valid_values.hdf5"  "$BASE/677fff96d9b97ffa85809a5d"
dl "$ROOT/bmrk03/valid_labels.csv"   "$BASE/677ffbb1a0ae8de743809010"

# ── Benchmark 04 (zipped train + valid) ───────────────────────
mkdir -p "$ROOT/bmrk04"
dl "$ROOT/bmrk04/train.zip"  "$BASE/t4kbn"
dl "$ROOT/bmrk04/valid.zip"  "$BASE/b7qd3"
unzip_into "$ROOT/bmrk04" "$ROOT/bmrk04/train.zip"
unzip_into "$ROOT/bmrk04" "$ROOT/bmrk04/valid.zip"

# ── Benchmark 05 (zipped train + valid) ───────────────────────
mkdir -p "$ROOT/bmrk05"
dl "$ROOT/bmrk05/train.zip"  "$BASE/677ffbcb830817335831f8aa"
dl "$ROOT/bmrk05/valid.zip"  "$BASE/677ffba1a0b7e18cdf31f6ff"
unzip_into "$ROOT/bmrk05" "$ROOT/bmrk05/train.zip"
unzip_into "$ROOT/bmrk05" "$ROOT/bmrk05/valid.zip"

# ── Benchmark 06 (zipped train + valid) ───────────────────────
mkdir -p "$ROOT/bmrk06"
dl "$ROOT/bmrk06/train.zip"  "$BASE/rygjk"
dl "$ROOT/bmrk06/valid.zip"  "$BASE/fspra"
unzip_into "$ROOT/bmrk06" "$ROOT/bmrk06/train.zip"
unzip_into "$ROOT/bmrk06" "$ROOT/bmrk06/valid.zip"

# ── Benchmark 07 (zipped train + valid) ───────────────────────
mkdir -p "$ROOT/bmrk07"
dl "$ROOT/bmrk07/train.zip"  "$BASE/677ff120d9b97ffa858091d3"
dl "$ROOT/bmrk07/valid.zip"  "$BASE/677ff10f5cd20d9eba08835a"
unzip_into "$ROOT/bmrk07" "$ROOT/bmrk07/train.zip"
unzip_into "$ROOT/bmrk07" "$ROOT/bmrk07/valid.zip"

# ── Benchmark 08 (zipped train + valid) ───────────────────────
mkdir -p "$ROOT/bmrk08"
dl "$ROOT/bmrk08/train.zip"  "$BASE/677fec7225ed845c76809195"
dl "$ROOT/bmrk08/valid.zip"  "$BASE/677ff4c98df5c32916088965"
unzip_into "$ROOT/bmrk08" "$ROOT/bmrk08/train.zip"
unzip_into "$ROOT/bmrk08" "$ROOT/bmrk08/valid.zip"

# ── Benchmark 09 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk09"
dl "$ROOT/bmrk09/train_values.hdf5"  "$BASE/677fead777f8449e4534d324"
dl "$ROOT/bmrk09/train_labels.csv"   "$BASE/677fe8f7830817335831f0be"
dl "$ROOT/bmrk09/valid_values.hdf5"  "$BASE/677feb2a1411a34fe3088d09"
dl "$ROOT/bmrk09/valid_labels.csv"   "$BASE/677fe8fe50bf78348e31f6e2"

# ── Benchmark 10 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk10"
dl "$ROOT/bmrk10/train_values.hdf5"  "$BASE/g387t"
dl "$ROOT/bmrk10/train_labels.csv"   "$BASE/ay4kg"
dl "$ROOT/bmrk10/valid_values.hdf5"  "$BASE/sk28v"
dl "$ROOT/bmrk10/valid_labels.csv"   "$BASE/q8vya"

# ── Benchmark 11 (validation only — no training split on OSF) ─
mkdir -p "$ROOT/bmrk11"
dl "$ROOT/bmrk11/valid_values.hdf5"  "$BASE/dc9b3"
dl "$ROOT/bmrk11/valid_labels.csv"   "$BASE/pkrg9"

# ── Benchmark 12 ──────────────────────────────────────────────
mkdir -p "$ROOT/bmrk12"
dl "$ROOT/bmrk12/train_values.hdf5"  "$BASE/4xefv"
dl "$ROOT/bmrk12/train_labels.csv"   "$BASE/3ead8"
dl "$ROOT/bmrk12/valid_values.hdf5"  "$BASE/a2esd"
dl "$ROOT/bmrk12/valid_labels.csv"   "$BASE/w9r7q"
dl "$ROOT/bmrk12/tests_values.hdf5"  "$BASE/mygqe"
dl "$ROOT/bmrk12/tests_labels.csv"   "$BASE/dvshj"

# ── All test data (benchmarks 1-12 combined zip) ──────────────
# Extracts tests_values.hdf5 / tests_labels.csv into each bmrkXX/ dir
mkdir -p "$ROOT/all_tests"
dl "$ROOT/all_tests/benchmarks_01_to_12_test_data.zip"  "$BASE/ayc6q"

echo ""
echo "=== Extracting combined test splits ==="
# Check if already extracted (bmrk01 test file is a proxy)
if [ -f "$ROOT/bmrk01/tests_values.hdf5" ]; then
    echo "[SKIP] Combined test zip already extracted"
else
    TMP="$ROOT/.test_extract_tmp"
    mkdir -p "$TMP"
    unzip -j -q "$ROOT/all_tests/benchmarks_01_to_12_test_data.zip" -d "$TMP"
    for f in "$TMP"/*.hdf5 "$TMP"/*.csv; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [[ "$fname" =~ bmrk([0-9]{2}) ]]; then
            bnum="${BASH_REMATCH[1]}"
            dest="$ROOT/bmrk${bnum}"
            mkdir -p "$dest"
            if [[ "$fname" == *tests_values* ]]; then   clean="tests_values.hdf5"
            elif [[ "$fname" == *tests_labels* ]]; then clean="tests_labels.csv"
            else clean="$fname"; fi
            mv "$f" "$dest/$clean"
            echo "  bmrk${bnum}/$clean"
        fi
    done
    rm -rf "$TMP"
fi

echo ""
echo "=== Download + extraction complete ==="
echo ""
echo "Split availability per benchmark:"
for i in $(seq -w 1 12); do
    dir="$ROOT/bmrk$i"
    splits=""
    { ls "$dir"/*train* 2>/dev/null | grep -q .; } && splits="$splits train"
    { ls "$dir"/*valid* 2>/dev/null | grep -q .; } && splits="$splits valid"
    { ls "$dir"/*test*  2>/dev/null | grep -q .; } && splits="$splits test"
    printf "  bmrk%s:%s\n" "$i" "${splits:- MISSING}"
done
echo ""
du -sh "$ROOT"/bmrk*/
