#!/usr/bin/env bash
# Fine-tune all 6 models with best-practice regime.
#
# Assumptions (verified before running):
#   1. VPN active — CVL cluster reachable via `ssh samal`
#   2. Using b-m-samal namespace (independent of b-r-gupta — where ollama runs).
#      Namespace quota: 1 GPU / 18 CPU / 35 GiB; verified free at setup.
#   3. Mac has conda env at /Users/roshangupta/miniconda3/envs/agri
#   4. Cluster workspace at /shared/hdd/b-m-samal/crop_yield_benchmark/
#      (b-m-samal owns it — copied from /shared/hdd/b-r-gupta/... on 2026-07-16
#       to avoid cross-account write issues on the b-r-gupta-owned tree)
#
# Live progress:
#   - Mac steps: python -u output stream directly to your terminal
#   - H100 steps: `kubectl logs -f` streams pod stdout live
#
# Interrupt safety:
#   - Ctrl-C on a step aborts the LOCAL command (kubectl logs -f detach)
#   - The pod itself keeps running server-side; reattach with:
#       ssh samal 'kubectl logs -f $(kubectl get pods -l job-name=finetune-<MODEL> -o jsonpath="{.items[0].metadata.name}")'
#   - Rerun the script — steps that already produced a fresh checkpoint are re-run cleanly (checkpoints just get overwritten)
#
# Runtime estimate (with H100 GPU):
#   0/8 Pre-flight      : 30 s
#   1/8 Config + sync   : 1 min
#   2/8 XGBoost (Mac)   : ~5 min
#   3/8 DLinear   (pod) : ~30 min
#   4/8 LSTM      (pod) : ~60 min
#   5/8 PatchTST  (pod) : ~90 min
#   6/8 iTransformer(pod): ~60 min
#   7/8 TFT       (pod) : up to 240 min (4 h cap)
#   8/8 Downstream refresh: ~45 min
#   ─────────────────────────────
#   TOTAL              : ~9-12 hours

set -e
set -o pipefail

REPO=/Users/roshangupta/Documents/Claude/Projects/Agriculture/crop_yield_benchmark
CLUSTER_WORKSPACE=/shared/hdd/b-m-samal/crop_yield_benchmark
PY=/Users/roshangupta/miniconda3/envs/agri/bin/python

cd "$REPO"

# ───────────────────────────────────────────────────────────────────────
# 0/8  Pre-flight — GPU free? cluster reachable?
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " 0/8  PRE-FLIGHT"
echo "═══════════════════════════════════════════════════════════════════"

if ! ssh -o ConnectTimeout=5 samal true 2>/dev/null; then
    echo "FAIL: cannot reach H100 cluster. Is VPN up?"
    exit 1
fi
echo "  cluster reachable"

# (No ollama check — ollama lives in b-r-gupta's namespace, not b-m-samal's.)
echo "  b-m-samal namespace GPU: independent of b-r-gupta"

echo "  namespace quota:"
ssh samal 'kubectl describe resourcequota 2>&1 | grep -E "requests.nvidia|requests.memory|requests.cpu" | head -4'

# ───────────────────────────────────────────────────────────────────────
# 1/8  Bump training config on cluster + sync latest scripts
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " 1/8  BUMP CONFIG + SYNC"
echo "═══════════════════════════════════════════════════════════════════"

echo "  syncing latest scripts + src + configs..."
rsync -avh --exclude=__pycache__ scripts/ samal:$CLUSTER_WORKSPACE/scripts/ 2>&1 | tail -2
rsync -avh --exclude=__pycache__ src/ samal:$CLUSTER_WORKSPACE/src/ 2>&1 | tail -2
rsync -avh --exclude=experiment.yaml configs/ samal:$CLUSTER_WORKSPACE/configs/ 2>&1 | tail -2
#   experiment.yaml is bumped AFTER rsync (below) so cluster keeps GPU + fine-tune regime
#   while Mac config stays on CPU for local XGBoost step

ssh samal "
    cd $CLUSTER_WORKSPACE
    sed -i 's/^  max_epochs:.*/  max_epochs: 200                              # fine-tune regime/' configs/experiment.yaml
    sed -i 's/^  early_stopping_patience:.*/  early_stopping_patience: 20      # fine-tune regime/' configs/experiment.yaml
    sed -i 's/^  max_minutes:.*/  max_minutes: 240                             # fine-tune regime (4 h cap)/' configs/experiment.yaml
    sed -i 's/^device:.*/device: \"cuda\"                                       # cluster GPU/' configs/experiment.yaml
    echo '  cluster config now:'
    grep -E '^device|max_epochs|early_stopping_patience|max_minutes' configs/experiment.yaml | sed 's/^/    /'
"

# ───────────────────────────────────────────────────────────────────────
# 2/8  XGBoost fine-tune on Mac (CPU-native, fast)
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " 2/8  XGBoost fine-tune (Mac, ~5 min)"
echo "═══════════════════════════════════════════════════════════════════"

rm -f results/checkpoints/xgboost.ubj results/checkpoints/xgboost.pipeline.json
env OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
    "$PY" -u scripts/03_train.py --model xgboost

echo "  XGBoost checkpoint size: $(du -h results/checkpoints/xgboost.ubj | cut -f1)"

# ───────────────────────────────────────────────────────────────────────
# Helper — launch one deep model on H100 with live log streaming
# ───────────────────────────────────────────────────────────────────────
finetune_deep() {
    local MODEL=$1
    local STEP=$2

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo " $STEP/8  $MODEL fine-tune on H100 (up to 240 min cap)"
    echo "═══════════════════════════════════════════════════════════════════"

    # Generate + apply pod YAML
    ssh samal "cat > /tmp/finetune_${MODEL}.yaml" <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: finetune-${MODEL}
  namespace: b-m-samal
spec:
  ttlSecondsAfterFinished: 86400
  backoffLimit: 0
  activeDeadlineSeconds: 18000   # 5 h pod ceiling (training cap is 4 h)
  template:
    spec:
      containers:
        - name: train
          image: abrainone/ai-linux
          imagePullPolicy: IfNotPresent
          workingDir: /workspace
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
              hostname
              nvidia-smi -L
              free -h
              pip install --no-cache-dir --quiet \\
                "pytorch-lightning" "plotly" "streamlit" "geopandas" \\
                "pyarrow==18.1.0" "pytorch-forecasting==1.8.0" "marshmallow<4"
              cd /workspace
              rm -f results/checkpoints/${MODEL}.pt results/checkpoints/${MODEL}.pipeline.json
              python -u scripts/03_train.py --model ${MODEL}
              chmod -R a+r results/checkpoints/ 2>/dev/null || true
              chmod a+rx results/checkpoints/ 2>/dev/null || true
              ls -lh results/checkpoints/${MODEL}.pt results/checkpoints/${MODEL}.pipeline.json
          resources:
            requests: { cpu: "8", memory: 32Gi, nvidia.com/gpu: 1 }
            limits:   { cpu: "8", memory: 32Gi, nvidia.com/gpu: 1 }
          volumeMounts:
            - { mountPath: /workspace, name: work }
            - { mountPath: /dev/shm,  name: shm  }
      restartPolicy: Never
      volumes:
        - name: work
          hostPath:
            path: /shared/hdd/b-m-samal/crop_yield_benchmark
            type: Directory
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 4Gi
YAML

    ssh samal "kubectl delete job finetune-${MODEL} --ignore-not-found=true; kubectl apply -f /tmp/finetune_${MODEL}.yaml"

    echo "  waiting for pod to schedule..."
    for i in $(seq 1 40); do
        POD=$(ssh samal "kubectl get pods -l job-name=finetune-${MODEL} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null" || true)
        if [ -n "$POD" ]; then
            STATUS=$(ssh samal "kubectl get pod \"$POD\" -o jsonpath='{.status.phase}' 2>/dev/null" || echo "?")
            echo "  [$i] pod=$POD status=$STATUS"
            [ "$STATUS" = "Running" ] || [ "$STATUS" = "Succeeded" ] || [ "$STATUS" = "Failed" ] && break
        fi
        sleep 3
    done

    if [ -z "$POD" ]; then
        echo "  pod never scheduled — check quota + node availability"
        return 1
    fi

    echo ""
    echo "  ↓ streaming live logs from $POD (Ctrl-C detaches; pod keeps running)"
    echo "─────────────────────────────────────────────────────────────────────"
    ssh samal "kubectl logs -f $POD"
    echo "─────────────────────────────────────────────────────────────────────"

    echo ""
    echo "  pulling ${MODEL} checkpoint back to Mac..."
    scp -q samal:$CLUSTER_WORKSPACE/results/checkpoints/${MODEL}.pt results/checkpoints/${MODEL}.pt
    scp -q samal:$CLUSTER_WORKSPACE/results/checkpoints/${MODEL}.pipeline.json results/checkpoints/${MODEL}.pipeline.json
    echo "  ${MODEL} checkpoint: $(du -h results/checkpoints/${MODEL}.pt | cut -f1)"
}

# ───────────────────────────────────────────────────────────────────────
# 3-7/8  DLinear → LSTM → PatchTST → iTransformer → TFT (sequential)
# ───────────────────────────────────────────────────────────────────────
finetune_deep dlinear      3
finetune_deep lstm         4
finetune_deep patchtst     5
finetune_deep itransformer 6
finetune_deep tft          7

# ───────────────────────────────────────────────────────────────────────
# 8/8  Full downstream refresh
# ───────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " 8/8  DOWNSTREAM REFRESH (on Mac, ~45 min)"
echo "═══════════════════════════════════════════════════════════════════"

echo "  re-eval multicrop_summary + horizons..."
rm -f results/metrics/multicrop_summary.csv results/metrics/multicrop_leaderboard.csv
env OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
    "$PY" -u scripts/05b_eval_multicrop.py

echo "  refit fusion (5-model + 6-model)..."
rm -f results/metrics/fusion_predictions_val.parquet results/metrics/fusion_predictions_test.parquet
env PYTHONUNBUFFERED=1 "$PY" -u scripts/07_fusion.py --exclude tft --suffix _5models
env PYTHONUNBUFFERED=1 "$PY" -u scripts/07_fusion.py --suffix _6models

echo "  rerun counterfactual (3-model CMIP6 ensemble)..."
rm -f results/metrics/counterfactual_daily_summary.csv results/metrics/counterfactual_daily_district.csv
env OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 \
    "$PY" -u scripts/13_counterfactual_daily.py

echo "  rerun taxonomy validation..."
env PYTHONUNBUFFERED=1 "$PY" -u scripts/14_taxonomy_validate.py

echo "  regenerate composite ranking..."
env PYTHONUNBUFFERED=1 "$PY" -u scripts/15_composite_ranking.py

echo "  regenerate 7 publication figures..."
env PYTHONUNBUFFERED=1 "$PY" -u paper/figures/generate_figures.py

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo " FINE-TUNE COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "New leaderboard:"
"$PY" -c "
import pandas as pd
df = pd.read_csv('results/metrics/multicrop_summary.csv')
lb = df.groupby('model').agg(r2_mean=('r2','mean'), r2_med=('r2','median')).sort_values('r2_mean', ascending=False)
print(lb.round(3).to_string())
print()
print('Per-crop best (excluding broken TFT cells):')
best = df[df.r2 > -1].loc[df[df.r2 > -1].groupby('crop')['r2'].idxmax()][['crop','model','r2']]
print(best.sort_values('r2', ascending=False).round(3).to_string(index=False))
"
echo ""
echo "New taxonomy hit rate:"
"$PY" -c "
import pandas as pd
v = pd.read_csv('results/metrics/taxonomy_validation.csv')
print(f'  top-1: {int(v[\"hit_at_1\"].sum())}/{len(v)}   top-3: {int(v[\"hit_at_3\"].sum())}/{len(v)}')
"
echo ""
echo "Next: commit + push to GitLab benchmark-release."
