#!/bin/bash
# Compatibility wrapper. Prefer a hardware-specific launcher:
# - 8b_trloo_hfsdp8_pytorch_eager.8xH20.sh
# - 8b_trloo_hfsdp8_pytorch_eager.16xA800.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/8b_trloo_hfsdp8_pytorch_eager.8xH20.sh" "$@"
