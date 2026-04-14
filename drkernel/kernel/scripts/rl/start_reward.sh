#!/bin/bash
# No set -e: we handle errors explicitly with checks after each step

# =============================================================================
# Start Reward Infrastructure (containers + Redis + API + GPU workers)
#
# Usage:
#   bash start_reward.sh       # skip if already healthy
#   bash start_reward.sh -f    # force restart everything
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/infra_common.sh"

FORCE=false
if [ "$1" = "-f" ]; then
    FORCE=true
    info "Force restart mode"
fi

ENABLE_REFERENCE_CACHE="${ENABLE_REFERENCE_CACHE:-true}"
REFERENCE_CACHE_DATASET_PATH="${REFERENCE_CACHE_DATASET_PATH:-}"
VAL_DATA_CACHE_DATASET_PATH="${VAL_DATA_CACHE_DATASET_PATH:-}"

# =============================================================================
# Step 0: Check if existing reward infra is healthy — skip if so
# =============================================================================
if [ "$FORCE" != "true" ]; then
    echo ""
    info "=== Checking existing reward infrastructure ==="

    EXISTING_HEALTHY=true

    # Check API health
    if curl -s --connect-timeout 5 "http://${REWARD_API_HOST}:${REWARD_API_PORT}/health" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='healthy'" 2>/dev/null; then
        info "API server: healthy"
    else
        info "API server: not healthy"
        EXISTING_HEALTHY=false
    fi

    # Check worker pools on each node (only if API is up)
    if [ "$EXISTING_HEALTHY" = "true" ]; then
        for node in "${REWARD_NODES[@]}"; do
            container=$(container_name "$node")
            # Check container running
            status=$(run_on_reward "$node" "docker inspect -f '{{.State.Status}}' ${container} 2>/dev/null") || true
            if ! echo "$status" | grep -q "running"; then
                info "  ${container}: not running"
                EXISTING_HEALTHY=false
                break
            fi
            # Check worker process alive
            worker_alive=$(docker_exec "$node" "$container" "pgrep -c -f gpu_worker 2>/dev/null || echo 0")
            worker_alive=$(echo "$worker_alive" | tr -d '[:space:]')
            if [ -z "$worker_alive" ] || [ "$worker_alive" -lt 1 ] 2>/dev/null; then
                info "  ${container}: no GPU worker running"
                EXISTING_HEALTHY=false
                break
            fi
            # Check worker pools initialized (from log)
            pools=$(docker_exec "$node" "$container" "grep -c 'Worker pool initialized' /tmp/worker.log 2>/dev/null || echo 0")
            pools=$(echo "$pools" | tr -d '[:space:]')
            if [ -z "$pools" ] || [ "$pools" -lt 8 ] 2>/dev/null; then
                info "  ${container}: only ${pools} worker pools (need 8+)"
                EXISTING_HEALTHY=false
                break
            fi
            info "  ${container}: ${pools} worker pools, worker alive"
        done
    fi

    if [ "$EXISTING_HEALTHY" = "true" ]; then
        info "=== Reward infrastructure already healthy, skipping startup ==="
        info "API:  http://${REWARD_API_HOST}:${REWARD_API_PORT}/health"
        exit 0
    fi

    info "Existing infrastructure not fully healthy, proceeding with startup..."
fi

# =============================================================================
# Step 1: Start containers
# =============================================================================
echo ""
info "=== Step 1: Starting reward containers ==="

for node in "${REWARD_NODES[@]}"; do
    container=$(container_name "$node")
    info "Starting ${container} on ${node}..."

    # Check if container exists and its status
    status=$(run_on_reward "$node" "docker inspect -f '{{.State.Status}}' ${container} 2>/dev/null")
    inspect_rc=$?
    info "  Current status: '${status}' (rc=${inspect_rc})"

    if echo "$status" | grep -q "running"; then
        info "  Already running, stopping first..."
        run_on_reward "$node" "docker stop -t 5 ${container}" || true
        sleep 2
        # Re-check status after stop
        status=$(run_on_reward "$node" "docker inspect -f '{{.State.Status}}' ${container} 2>/dev/null")
        inspect_rc=$?
        info "  Status after stop: '${status}'"
    fi

    # Try to start existing container. If it fails, recreate.
    STARTED=false
    if [ $inspect_rc -eq 0 ] && echo "$status" | grep -qE "exited|created"; then
        info "  Starting existing container..."
        start_output=$(run_on_reward "$node" "docker start ${container}")
        start_rc=$?
        info "  docker start rc=${start_rc}, output: ${start_output}"
        if [ $start_rc -eq 0 ]; then
            sleep 1
            verify_status=$(run_on_reward "$node" "docker inspect -f '{{.State.Status}}' ${container} 2>/dev/null")
            if echo "$verify_status" | grep -q "running"; then
                STARTED=true
                info "  Container started successfully"
            else
                warn "  docker start returned 0 but container not running (status: ${verify_status})"
            fi
        else
            warn "  docker start failed (rc=${start_rc})"
        fi
    fi

    if [ "$STARTED" != "true" ]; then
        warn "  Container in bad state or start failed. Recreating..."
        if ! recreate_container "$node"; then
            error "  Failed to recreate container on ${node}!"
            exit 1
        fi
        sleep 3
    fi

    # Verify running
    status=$(run_on_reward "$node" "docker inspect -f '{{.State.Status}}' ${container} 2>/dev/null")
    if ! echo "$status" | grep -q "running"; then
        error "  Container ${container} not running after start/recreate! Status: ${status}"
        exit 1
    fi
    info "  Container running"

    # Verify NFS
    if docker_exec "$node" "$container" "ls ${REWARD_REPO_PATH}/kernelgym/__init__.py >/dev/null 2>&1"; then
        info "  NFS OK"
    else
        error "  NFS NOT visible in ${container}!"
        error "  Fix: mount NFS on ${node}, then re-run this script"
        exit 1
    fi

    # Verify GPUs
    gpu_count=$(docker_exec "$node" "$container" "nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l")
    gpu_count=$(echo "$gpu_count" | tr -d '[:space:]')
    info "  GPUs visible: ${gpu_count}"
    if [ -z "$gpu_count" ] || [ "$gpu_count" -lt 8 ] 2>/dev/null; then
        error "  Expected 8 GPUs, got '${gpu_count}'!"
        exit 1
    fi
done

# =============================================================================
# Step 1.5: Install Python dependencies in containers
# =============================================================================
echo ""
info "=== Step 1.5: Ensuring Python dependencies ==="

REWARD_PIP_DEPS="pydantic-settings redis uvicorn[standard] aiohttp python-multipart structlog psutil tenacity fastapi"

for node in "${REWARD_NODES[@]}"; do
    container=$(container_name "$node")
    info "Checking Python deps on ${container}..."
    if docker_exec "$node" "$container" "python3 -c 'import pydantic_settings, fastapi, redis, uvicorn, aiohttp' 2>/dev/null"; then
        info "  Python deps OK"
    else
        info "  Installing missing deps..."
        if ! docker_exec "$node" "$container" "pip install -q ${REWARD_PIP_DEPS}"; then
            error "  pip install failed on ${container}!"
            error "  Try: ssh chenshuailin@${node} 'sudo docker exec ${container} pip install ${REWARD_PIP_DEPS}'"
            exit 1
        fi
        info "  Deps installed"
    fi
done

# =============================================================================
# Step 2: Start Redis
# =============================================================================
echo ""
info "=== Step 2: Starting Redis ==="

api_container=$(container_name "${REWARD_API_HOST}")

# Kill any existing redis
docker_exec "${REWARD_API_HOST}" "$api_container" "pkill redis-server 2>/dev/null || true" || true
sleep 1

# Ensure redis is installed
info "Checking if redis-server is installed..."
if ! docker_exec "${REWARD_API_HOST}" "$api_container" "which redis-server >/dev/null 2>&1"; then
    info "redis-server not found, installing..."
    if ! docker_exec "${REWARD_API_HOST}" "$api_container" "apt-get update -qq && apt-get install -y -qq redis-server"; then
        error "Failed to install redis-server in ${api_container}!"
        error "Try manually: ssh chenshuailin@${REWARD_API_HOST} 'sudo docker exec ${api_container} bash -c \"apt-get update && apt-get install -y redis-server\"'"
        exit 1
    fi
    info "redis-server installed successfully"
else
    info "redis-server already installed"
fi

# Start redis
info "Starting redis on port ${REDIS_PORT}..."
if ! docker_exec "${REWARD_API_HOST}" "$api_container" "redis-server --port ${REDIS_PORT} --bind 0.0.0.0 --protected-mode no --daemonize yes"; then
    error "redis-server --daemonize failed!"
    exit 1
fi
sleep 2

# Verify PONG
pong_output=$(docker_exec "${REWARD_API_HOST}" "$api_container" "redis-cli -p ${REDIS_PORT} ping")
pong_rc=$?
info "  redis-cli ping: rc=${pong_rc}, output='${pong_output}'"
if [ $pong_rc -ne 0 ] || ! echo "$pong_output" | grep -q "PONG"; then
    error "Redis failed to start! Expected PONG, got: ${pong_output}"
    error "Check: ssh chenshuailin@${REWARD_API_HOST} 'sudo docker exec ${api_container} redis-cli -p ${REDIS_PORT} ping'"
    exit 1
fi
info "Redis OK"

# =============================================================================
# Step 3: Start API server
# =============================================================================
echo ""
info "=== Step 3: Starting API server ==="

# Kill any existing API
docker_exec "${REWARD_API_HOST}" "$api_container" "pkill -f kernelgym.server.api.server 2>/dev/null || true" || true
sleep 1

info "Launching API server (detached)..."
api_launch_output=$(docker_exec_env "${REWARD_API_HOST}" "$api_container" \
    "-e" "ENABLE_REFERENCE_CACHE=${ENABLE_REFERENCE_CACHE}" \
    "-e" "REFERENCE_CACHE_DATASET_PATH=${REFERENCE_CACHE_DATASET_PATH}" \
    "-e" "VAL_DATA_CACHE_DATASET_PATH=${VAL_DATA_CACHE_DATASET_PATH}" -- \
    "cd /tmp && python3 -u -m kernelgym.server.api.server > /tmp/server.log 2>&1")
api_launch_rc=$?
info "  docker exec -d rc=${api_launch_rc}"
if [ $api_launch_rc -ne 0 ]; then
    error "Failed to launch API server container! Output: ${api_launch_output}"
    exit 1
fi

info "Waiting for API to become healthy..."
API_HEALTHY=false
for i in $(seq 1 6); do
    sleep 5
    if curl -s --connect-timeout 5 "http://${REWARD_API_HOST}:${REWARD_API_PORT}/health" | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['status']=='healthy'" 2>/dev/null; then
        API_HEALTHY=true
        break
    fi
    warn "  attempt ${i}/6 — not yet healthy"
done

if [ "$API_HEALTHY" = "true" ]; then
    info "API server healthy"
else
    error "API server failed after 30s!"
    error "Server log:"
    docker_exec "${REWARD_API_HOST}" "$api_container" "tail -20 /tmp/server.log" 2>/dev/null || true
    exit 1
fi

# =============================================================================
# Step 4: Start GPU workers
# =============================================================================
echo ""
info "=== Step 4: Starting GPU workers ==="

for node in "${REWARD_NODES[@]}"; do
    container=$(container_name "$node")
    suffix="${node##*.}"
    info "Starting GPU worker on ${container} (${node})..."

    # Kill any existing worker
    docker_exec "$node" "$container" "pkill -f gpu_worker 2>/dev/null || true" || true
    sleep 1

    info "  Launching worker (detached)..."
    worker_output=$(docker_exec_env "$node" "$container" \
        "-e" "GPU_DEVICES=[0,1,2,3,4,5,6,7]" \
        "-e" "NODE_ID=reward-${suffix}" \
        "-e" "ENABLE_REFERENCE_CACHE=${ENABLE_REFERENCE_CACHE}" \
        "-e" "REFERENCE_CACHE_DATASET_PATH=${REFERENCE_CACHE_DATASET_PATH}" \
        "-e" "VAL_DATA_CACHE_DATASET_PATH=${VAL_DATA_CACHE_DATASET_PATH}" -- \
        "cd /tmp && python3 -u -m kernelgym.worker.gpu_worker > /tmp/worker.log 2>&1")
    worker_rc=$?
    info "  docker exec -d rc=${worker_rc}"
    if [ $worker_rc -ne 0 ]; then
        error "  Failed to launch GPU worker on ${node}! Output: ${worker_output}"
        exit 1
    fi
done

info "Waiting 30s for worker pools to initialize..."
sleep 30

ALL_OK=true
for node in "${REWARD_NODES[@]}"; do
    container=$(container_name "$node")
    pools=$(docker_exec "$node" "$container" "grep -c 'Worker pool initialized' /tmp/worker.log 2>/dev/null" || echo "0")
    pools=$(echo "$pools" | tr -d '[:space:]')
    if [ -n "$pools" ] && [ "$pools" -ge 8 ] 2>/dev/null; then
        info "  ${container}: ${pools} worker pools OK"
    else
        warn "  ${container}: only ${pools} worker pools (expected 8+)"
        warn "  Worker log tail:"
        docker_exec "$node" "$container" "tail -10 /tmp/worker.log" 2>/dev/null || true
        ALL_OK=false
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
if [ "$ALL_OK" = true ]; then
    info "=== Reward infrastructure ready ==="
else
    warn "=== Reward infrastructure partially ready (check warnings above) ==="
fi
info "API:  http://${REWARD_API_HOST}:${REWARD_API_PORT}/health"
info "Test: curl -s http://${REWARD_API_HOST}:${REWARD_API_PORT}/health | python3 -m json.tool"
