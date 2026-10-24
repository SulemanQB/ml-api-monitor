#!/usr/bin/env bash
# Install pinned kind, kubectl, and kubeconform binaries for Linux.
set -euo pipefail

KUBECTL_VERSION=v1.37.1
KIND_VERSION=v0.33.0
KUBECONFORM_VERSION=v0.8.0

arch=$(uname -m)
case "$arch" in
  x86_64) goarch=amd64 ;;
  aarch64 | arm64) goarch=arm64 ;;
  *)
    echo "unsupported architecture: $arch" >&2
    exit 1
    ;;
esac

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/kubectl" "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${goarch}/kubectl"
curl -fsSL -o "$tmp/kind" "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${goarch}"
curl -fsSL -o "$tmp/kubeconform.tar.gz" "https://github.com/yannh/kubeconform/releases/download/${KUBECONFORM_VERSION}/kubeconform-linux-${goarch}.tar.gz"
tar -xzf "$tmp/kubeconform.tar.gz" -C "$tmp" kubeconform

sudo install -m 0755 "$tmp/kubectl" /usr/local/bin/kubectl
sudo install -m 0755 "$tmp/kind" /usr/local/bin/kind
sudo install -m 0755 "$tmp/kubeconform" /usr/local/bin/kubeconform

kubectl version --client
kind version
kubeconform -v
