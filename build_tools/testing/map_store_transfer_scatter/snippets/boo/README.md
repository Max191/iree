# BOO convolution extraction index

No BOO IR is checked in here unless it is produced by a local BOO run. Do not
invent placeholder IR. Use this index to regenerate real convolution snippets
and then copy only complete modules or full dispatch funcs that still contain
`iree_linalg_ext.map_store` before generic vectorization.

## Prerequisites

- `IREE_BUILD=/path/to/iree-build` with `tools/iree-compile` and Python bindings
- `VENV=/path/to/.venv` containing `iree-turbine` and PyTorch ROCm packages
- `FILTERED_CONVS=/path/to/filtered_convs.txt` for the source BOO command list
- Optional `ROCM_LIBRARY_DIR=/path/to/site-packages/_rocm_sdk_libraries_*/lib`

## Filter commands with pre-vectorization `map_store`

```bash
WORKDIR=/tmp/boo_map_store_filter

python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  filter-convs \
  --commands-file "$FILTERED_CONVS" \
  --iree-build "$IREE_BUILD" \
  --venv "$VENV" \
  --work-dir "$WORKDIR" \
  --matches-out "$WORKDIR/map_store_convs.txt" \
  --nonmatches-out "$WORKDIR/non_map_store_convs.txt" \
  --errors-out "$WORKDIR/errors.txt" \
  --report-out "$WORKDIR/boo_map_store_filter_report.json" \
  --stop-before-file-regex generic-vectorization \
  --timeout-seconds 0 \
  --limit 0 \
  --verify-numerics
```

Expected dump paths for each matched case:

- `$WORKDIR/case_<N>/compile_command_*.txt`
- `$WORKDIR/case_<N>/dump_tree/`
- `$WORKDIR/case_<N>/dump_tree/**generic-vectorization*` as the stop boundary
- `$WORKDIR/boo_map_store_filter_report.json` with matched dump filenames

## Create a small reproducible subset

```bash
python3 build_tools/testing/map_store_transfer_scatter/boo_map_store_filter.py \
  select-small "$WORKDIR/map_store_convs.txt" "$WORKDIR/map_store_convs_small.txt" \
  --count 10
```

## Extract real snippets

For each selected case, copy a complete module or full dispatch function from
the latest dump file before the first `generic-vectorization` boundary into:

```text
build_tools/testing/map_store_transfer_scatter/snippets/boo/case_<N>_<shape>.mlir
```

Keep a short header comment with:

- source command file path
- BOO command line
- dump tree path
- pre-vectorization dump filename

Then validate with `iree-opt --split-input-file --verify-diagnostics` if the
same IREE build is available.

