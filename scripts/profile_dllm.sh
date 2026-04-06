#!/bin/bash
# Profile dLLM inference pipeline with nsys
# Usage:
#   bash scripts/profile_dllm.sh LowConfidence
#   bash scripts/profile_dllm.sh LowConfidenceFDFO

set -e

ALGORITHM=${1:-LowConfidence}
MODEL_PATH=${MODEL_PATH:-/home/wxt/LLaDA2.0-mini}
TP_SIZE=${TP_SIZE:-2}
PORT=${PORT:-30000}
OUTPUT_DIR=${OUTPUT_DIR:-/home/wxt/nsys_reports}
PYTHON=${PYTHON:-/home/wxt/miniconda3/envs/wxt-sglang/bin/python}
MAX_TOKENS=${MAX_TOKENS:-128}
NUM_REQUESTS=${NUM_REQUESTS:-3}
ALGORITHM_CONFIG=${ALGORITHM_CONFIG:-}

mkdir -p "$OUTPUT_DIR"

REPORT_NAME="dllm_${ALGORITHM}_tp${TP_SIZE}"
REPORT_PATH="${OUTPUT_DIR}/${REPORT_NAME}"

echo "============================================"
echo "Profiling dLLM: ${ALGORITHM}"
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP_SIZE}"
echo "Output: ${REPORT_PATH}.nsys-rep"
echo "============================================"

# Launch server under nsys
echo "[1/4] Starting sglang server under nsys..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --output="${REPORT_PATH}" \
    --force-overwrite=true \
    --kill=sigkill \
    "$PYTHON" -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --trust-remote-code \
        --tp "$TP_SIZE" \
        --attention-backend flashinfer \
        --dllm-algorithm "$ALGORITHM" \
        --mem-fraction-static 0.75 \
        --cuda-graph-max-bs 32 \
        --max-running-requests 256 \
        --skip-server-warmup \
        ${ALGORITHM_CONFIG:+--dllm-algorithm-config "$ALGORITHM_CONFIG"} &

SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

# Wait for server to be ready
echo "[2/4] Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "object"; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

# Verify server is up
if ! curl -s http://localhost:${PORT}/v1/models 2>/dev/null | grep -q "object"; then
    echo "ERROR: Server failed to start within 120s"
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# Send requests
echo "[3/4] Sending ${NUM_REQUESTS} requests (max_tokens=${MAX_TOKENS})..."

# Use a longer prompt to trigger multiple block iterations
PROMPT="Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, gravitational time dilation, and the predictions that have been experimentally verified. Start from the basic postulates and build up to the field equations."

for i in $(seq 1 "$NUM_REQUESTS"); do
    echo "  Request ${i}/${NUM_REQUESTS}..."
    RESPONSE=$(curl -s http://localhost:${PORT}/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${MODEL_PATH}\", \"prompt\": \"${PROMPT}\", \"max_tokens\": ${MAX_TOKENS}, \"temperature\": 0.7}")
    TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens','?'))" 2>/dev/null || echo "?")
    echo "    -> Generated ${TOKENS} tokens"
done

# Stop server: kill python child so nsys can exit cleanly and write the report
echo "[4/4] Stopping server and collecting nsys report..."
sleep 3

# Find the actual sglang python process (grandchild of nsys) and send SIGINT
SGLANG_PID=$(pgrep -f "sglang.launch_server" 2>/dev/null | head -1)
if [ -n "$SGLANG_PID" ]; then
    echo "Sending SIGINT to sglang PID: $SGLANG_PID"
    kill -INT "$SGLANG_PID" 2>/dev/null || true
fi

# Wait for nsys to finish writing the report (may take up to 120s for large traces)
echo "Waiting for nsys to write report..."
for i in $(seq 1 120); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "nsys exited after ${i}s"
        break
    fi
    sleep 1
done

# Force-kill if still running
kill -9 $SERVER_PID 2>/dev/null || true
sleep 2

echo ""
echo "============================================"
echo "Done! Report saved to:"
echo "  ${REPORT_PATH}.nsys-rep"
echo ""
echo "View with: nsys-ui ${REPORT_PATH}.nsys-rep"
echo "Or stats:  nsys stats ${REPORT_PATH}.nsys-rep"
echo "============================================"
