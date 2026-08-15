#!/usr/bin/env bash
set -euo pipefail

# Capture the exact local Project27 code/patch state from the M1 Max into this repo.
# Run this on the Mac that contains ~/src/mlx-m1-qmv and the Project27 artifacts.

REPO_DIR="${REPO_DIR:-$HOME/src/m1-max-llm-tuning}"
MLX_DIR="${MLX_DIR:-$HOME/src/mlx-m1-qmv}"
PATCH_DIR="${PATCH_DIR:-$HOME/project24-patches}"
VENV="${VENV:-$HOME/.venvs/mlx-dspark}"
TARGET="${TARGET:-$HOME/models/Qwen3.8-27B-MLX-6bit-FP16-Q8HEAD}"
DRAFT="${DRAFT:-$HOME/models/Qwen3.8-27B-MTP-Q3MLP-Q6ATTN-FP16-27.305}"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: clone j-kuman/m1-max-llm-tuning to $REPO_DIR first"
  exit 1
fi
if [[ ! -d "$MLX_DIR/.git" ]]; then
  echo "ERROR: MLX tuning repo not found at $MLX_DIR"
  exit 1
fi

mkdir -p \
  "$REPO_DIR/project27/benchmarks" \
  "$REPO_DIR/project27/builders" \
  "$REPO_DIR/project27/patches/mlx" \
  "$REPO_DIR/project27/patches/mlx-vlm" \
  "$REPO_DIR/project27/source-snapshots" \
  "$REPO_DIR/project27/environment" \
  "$REPO_DIR/project27/model-manifests" \
  "$REPO_DIR/project27/negative-results"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst"
    echo "captured: $src"
  else
    echo "missing (skipped): $src"
  fi
}

# ---------------------------------------------------------------------------
# 1) Exact benchmark and correctness scripts used during the campaign.
# ---------------------------------------------------------------------------
for f in \
  q38-persistent-rawx.py \
  q38-q3mlp-full.py \
  q38-persistent-warmup-only.py \
  q38-cross-drafter-exact.py \
  q38-mtp-block-sweep.py \
  q38-m4-prof.py \
  q38-q8head-regress.py \
  q38-q8head-helper-only.py \
  q38-q8head-split.py; do
  copy_if_exists "/tmp/$f" "$REPO_DIR/project27/benchmarks/$f"
done

# Builders / MTP quantization helpers.
for f in \
  build-q38-mtp-bits.py \
  build-q38-mtp-q3mlp.py \
  compare-q38-mtp-weights.py \
  audit-q38-mtp-q8-runtime.py; do
  copy_if_exists "/tmp/$f" "$REPO_DIR/project27/builders/$f"
done

# ---------------------------------------------------------------------------
# 2) Exact Q8-head mlx-vlm patch/snapshot.
# ---------------------------------------------------------------------------
copy_if_exists \
  "$PATCH_DIR/language.py.q8shared4-26.69-exact" \
  "$REPO_DIR/project27/source-snapshots/language.py.q8shared4-26.69-exact"
copy_if_exists \
  "$PATCH_DIR/project24-q8head-shared4-26.69.patch" \
  "$REPO_DIR/project27/patches/mlx-vlm/project24-q8head-shared4-26.69.patch"

LANGFILE="$VENV/lib/python3.14/site-packages/mlx_vlm/models/qwen3_5/language.py"
copy_if_exists \
  "$LANGFILE" \
  "$REPO_DIR/project27/source-snapshots/language.py.installed-project27"

# ---------------------------------------------------------------------------
# 3) MLX fork: preserve history metadata + exact final source patch.
# ---------------------------------------------------------------------------
pushd "$MLX_DIR" >/dev/null

{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "DESCRIBE=$(git describe --tags --always --dirty 2>/dev/null || true)"
  echo "BRANCH=$(git branch --show-current)"
  echo
  echo "--- STATUS ---"
  git status --short
  echo
  echo "--- REMOTES ---"
  git remote -v
} > "$REPO_DIR/project27/environment/mlx-git-state.txt"

git log --all --decorate --oneline --graph \
  > "$REPO_DIR/project27/environment/mlx-all-history.txt"
git tag --list --sort=creatordate \
  > "$REPO_DIR/project27/environment/mlx-tags.txt"
git branch -a -vv \
  > "$REPO_DIR/project27/environment/mlx-branches.txt"

# Final champion is known to have been frozen at this tag during Project27.
FINAL_TAG="project24-q6-m4-k4-qdot4-rawx-exact-26.57"
if git rev-parse "$FINAL_TAG" >/dev/null 2>&1; then
  git show --no-ext-diff --binary --format=fuller "$FINAL_TAG" \
    > "$REPO_DIR/project27/patches/mlx/${FINAL_TAG}.commit.txt"

  # Snapshot the exact modified files at the tag.
  git show "$FINAL_TAG:mlx/backend/metal/kernels/quantized.h" \
    > "$REPO_DIR/project27/source-snapshots/quantized.h.rawx-26.57"
  git show "$FINAL_TAG:mlx/backend/metal/quantized.cpp" \
    > "$REPO_DIR/project27/source-snapshots/quantized.cpp.rawx-26.57"
  git show "$FINAL_TAG:mlx/backend/metal/kernels/quantized.metal" \
    > "$REPO_DIR/project27/source-snapshots/quantized.metal.rawx-26.57"

  # Best-effort consolidated patch against the first parent of the earliest
  # Project24 tagged commit. The source snapshots above remain authoritative.
  EARLY_TAG="project24-q6-m3-4x2-22.0"
  if git rev-parse "$EARLY_TAG^" >/dev/null 2>&1; then
    BASE="$(git rev-parse "$EARLY_TAG^")"
    git diff --binary "$BASE" "$FINAL_TAG" \
      > "$REPO_DIR/project27/patches/mlx/project24-to-rawx-26.57.patch"
    echo "$BASE" > "$REPO_DIR/project27/environment/mlx-project24-base.txt"
  fi
fi

popd >/dev/null

# Preserve the known negative-result patch files if they still exist.
for f in \
  project24-rawx-mask-cleanup-flat.patch \
  project24-rawx-float4-flat.patch \
  project24-rawx-8x1-dead.patch \
  project24-rawx-4x1-dead.patch \
  project24-rawx-4x4-dead.patch \
  project24-rawx-8x2-tie.patch \
  project24-rawx-halfx-tie-production-loss.patch \
  project24-rawx-k2-loss.patch; do
  copy_if_exists "/tmp/$f" "$REPO_DIR/project27/negative-results/$f"
done

# ---------------------------------------------------------------------------
# 4) Environment capture.
# ---------------------------------------------------------------------------
{
  date
  echo
  sw_vers || true
  echo
  uname -a
  echo
  "$VENV/bin/python" --version || true
  "$VENV/bin/python" -m pip --version || true
  echo
  "$VENV/bin/python" -m pip freeze || true
} > "$REPO_DIR/project27/environment/python-and-system.txt"

{
  sysctl iogpu.wired_limit_mb 2>/dev/null || true
  command -v cmake || true
  cmake --version 2>/dev/null || true
  xcrun -sdk macosx metal --version 2>/dev/null || true
} > "$REPO_DIR/project27/environment/metal-and-memory.txt"

# ---------------------------------------------------------------------------
# 5) Model manifests. Do NOT commit model weight files.
# ---------------------------------------------------------------------------
{
  echo "TARGET=$TARGET"
  if [[ -f "$TARGET/config.json" ]]; then
    shasum -a 256 "$TARGET/config.json"
    cp -f "$TARGET/config.json" "$REPO_DIR/project27/model-manifests/target-config.json"
  fi
  if [[ -f "$TARGET/model.safetensors" ]]; then
    shasum -a 256 "$TARGET/model.safetensors"
  fi
} > "$REPO_DIR/project27/model-manifests/target-sha256.txt"

{
  echo "DRAFT=$DRAFT"
  if [[ -f "$DRAFT/config.json" ]]; then
    shasum -a 256 "$DRAFT/config.json"
    cp -f "$DRAFT/config.json" "$REPO_DIR/project27/model-manifests/draft-config.json"
  fi
  if [[ -f "$DRAFT/model.safetensors" ]]; then
    shasum -a 256 "$DRAFT/model.safetensors"
  fi
} > "$REPO_DIR/project27/model-manifests/draft-sha256.txt"

# ---------------------------------------------------------------------------
# 6) Hash every captured reproducibility artifact.
# ---------------------------------------------------------------------------
(
  cd "$REPO_DIR"
  find project27 -type f -print0 \
    | sort -z \
    | xargs -0 shasum -a 256 \
    > project27/SHA256SUMS.txt
)

# ---------------------------------------------------------------------------
# 7) Commit and push.
# ---------------------------------------------------------------------------
pushd "$REPO_DIR" >/dev/null

git add project27 scripts/snapshot_local_project27.sh
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "Snapshot exact Project27 implementation and reproducibility assets"
  git push origin main
fi

popd >/dev/null

echo
echo "Project27 local implementation snapshot complete."
echo "Repo: $REPO_DIR"
