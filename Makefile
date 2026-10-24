PYTHON ?= python3
CLUSTER ?= ml-monitor
IMAGE ?= ml-api-monitor:local
NAMESPACE ?= ml-monitor
KUBE_CONTEXT ?= kind-$(CLUSTER)
K8S_VERSION ?= 1.37.0
METRICS_SERVER_VERSION ?= v0.9.0
API_LOCAL_PORT ?= 18000
PROMETHEUS_LOCAL_PORT ?= 19090

KUBECTL ?= kubectl --context $(KUBE_CONTEXT)

.PHONY: test lint-manifests image kind-up kind-load metrics-server deploy smoke rollout teardown verify

test:
	$(PYTHON) -m pytest -q

lint-manifests:
	kubectl kustomize deploy/k8s | kubeconform -kubernetes-version $(K8S_VERSION) -strict -summary

image:
	docker build -t $(IMAGE) .

kind-up:
	@test "$$(awk '/^name: /{print $$2; exit}' deploy/kind/cluster.yaml)" = "$(CLUSTER)"
	@if ! kind get clusters | grep -qx '$(CLUSTER)'; then \
		kind create cluster --config deploy/kind/cluster.yaml; \
	else \
		echo "cluster $(CLUSTER) already exists"; \
	fi
	kubectl config use-context $(KUBE_CONTEXT)

kind-load: kind-up image
	kind load docker-image $(IMAGE) --name $(CLUSTER)

metrics-server:
	$(KUBECTL) apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/$(METRICS_SERVER_VERSION)/components.yaml
	@if ! $(KUBECTL) -n kube-system get deployment metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q kubelet-insecure-tls; then \
		$(KUBECTL) -n kube-system patch deployment metrics-server --type json --patch-file deploy/kind/metrics-server-kind-patch.json; \
	fi
	$(KUBECTL) -n kube-system rollout status deployment/metrics-server --timeout=300s

deploy: kind-load
	$(KUBECTL) apply -k deploy/k8s
	$(MAKE) metrics-server
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/ml-api --timeout=300s
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/prometheus --timeout=300s

smoke:
	PYTHON=$(PYTHON) NAMESPACE=$(NAMESPACE) KUBE_CONTEXT=$(KUBE_CONTEXT) API_LOCAL_PORT=$(API_LOCAL_PORT) PROMETHEUS_LOCAL_PORT=$(PROMETHEUS_LOCAL_PORT) ./scripts/k8s_smoke.sh

rollout:
	$(KUBECTL) -n $(NAMESPACE) rollout restart deployment/ml-api
	$(KUBECTL) -n $(NAMESPACE) rollout status deployment/ml-api --timeout=300s
	$(KUBECTL) -n $(NAMESPACE) get pods -l app.kubernetes.io/name=ml-api -o wide

teardown:
	kind delete cluster --name $(CLUSTER)

verify: test lint-manifests deploy smoke rollout
