#!/usr/bin/env bash
# Port-forward the deployed API and Prometheus, then check health, inference, scrapes, and the HPA.
set -euo pipefail

PYTHON=${PYTHON:-python3}
NAMESPACE=${NAMESPACE:-ml-monitor}
KUBE_CONTEXT=${KUBE_CONTEXT:-kind-ml-monitor}
API_LOCAL_PORT=${API_LOCAL_PORT:-18000}
PROMETHEUS_LOCAL_PORT=${PROMETHEUS_LOCAL_PORT:-19090}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

kubectl() {
  command kubectl --context "$KUBE_CONTEXT" "$@"
}

API_PF_PID=""
PROM_PF_PID=""

cleanup() {
  if [[ -n "$API_PF_PID" ]]; then
    kill "$API_PF_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$PROM_PF_PID" ]]; then
    kill "$PROM_PF_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_http() {
  local url=$1
  local log=$2
  local attempt
  for attempt in $(seq 1 40); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  if [[ -f "$log" ]]; then
    echo "port-forward log:" >&2
    cat "$log" >&2
  fi
  return 1
}

echo "== pods =="
kubectl -n "$NAMESPACE" get pods -o wide

echo "== injected configuration =="
expected_model=$(kubectl -n "$NAMESPACE" get configmap ml-api-config -o jsonpath='{.data.MODEL_NAME}')
expected_threshold=$(kubectl -n "$NAMESPACE" get secret ml-api-settings -o jsonpath='{.data.DRIFT_THRESHOLD}' | base64 -d)
actual=$(kubectl -n "$NAMESPACE" exec deploy/ml-api -- python -c 'import os; print(os.environ["MODEL_NAME"]); print(os.environ["DRIFT_THRESHOLD"])')
actual_model=$(printf '%s\n' "$actual" | sed -n '1p')
actual_threshold=$(printf '%s\n' "$actual" | sed -n '2p')
echo "MODEL_NAME expected=${expected_model} actual=${actual_model}"
echo "DRIFT_THRESHOLD expected=${expected_threshold} actual=${actual_threshold}"
if [[ "$actual_model" != "$expected_model" || "$actual_threshold" != "$expected_threshold" ]]; then
  echo "pod environment does not match the ConfigMap and Secret" >&2
  exit 1
fi

kubectl -n "$NAMESPACE" port-forward svc/ml-api "${API_LOCAL_PORT}:8000" >/tmp/ml-api-port-forward.log 2>&1 &
API_PF_PID=$!
wait_http "http://127.0.0.1:${API_LOCAL_PORT}/livez" /tmp/ml-api-port-forward.log

echo "== livez =="
curl -fsS "http://127.0.0.1:${API_LOCAL_PORT}/livez"
echo
echo "== readyz =="
curl -fsS "http://127.0.0.1:${API_LOCAL_PORT}/readyz"
echo

API_URL="http://127.0.0.1:${API_LOCAL_PORT}" "$PYTHON" "$ROOT/scripts/smoke_test.py"

kubectl -n "$NAMESPACE" port-forward svc/prometheus "${PROMETHEUS_LOCAL_PORT}:9090" >/tmp/prometheus-port-forward.log 2>&1 &
PROM_PF_PID=$!
wait_http "http://127.0.0.1:${PROMETHEUS_LOCAL_PORT}/-/ready" /tmp/prometheus-port-forward.log

echo "== prometheus ml-api targets =="
if ! PROMETHEUS_LOCAL_PORT="$PROMETHEUS_LOCAL_PORT" "$PYTHON" - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request

base = f"http://127.0.0.1:{os.environ['PROMETHEUS_LOCAL_PORT']}"
query = 'up{job="ml-api"}'
url = base + "/api/v1/query?" + urllib.parse.urlencode({"query": query})
deadline = time.time() + 120
last = None
while time.time() < deadline:
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.load(response)
    results = payload.get("data", {}).get("result", [])
    last = results
    if len(results) >= 2 and all(float(item["value"][1]) == 1.0 for item in results):
        print(json.dumps(results, indent=2))
        print(f"ml-api scrape targets up: {len(results)}")
        raise SystemExit(0)
    time.sleep(3)
print(json.dumps(last, indent=2))
raise SystemExit("Prometheus did not report at least two ml-api pods up")
PY
then
  echo "prometheus logs:" >&2
  kubectl -n "$NAMESPACE" logs deploy/prometheus --tail=120 >&2 || true
  exit 1
fi

echo "== hpa =="
utilization=""
for _ in $(seq 1 36); do
  utilization=$(kubectl -n "$NAMESPACE" get hpa ml-api -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null || true)
  if [[ -n "$utilization" ]]; then
    kubectl -n "$NAMESPACE" get hpa ml-api
    echo "cpu averageUtilization=${utilization}"
    break
  fi
  sleep 5
done
if [[ -z "$utilization" ]]; then
  kubectl -n "$NAMESPACE" describe hpa ml-api >&2 || true
  echo "HPA did not report CPU utilization" >&2
  exit 1
fi

echo
echo "KUBERNETES SMOKE CHECK PASSED"
