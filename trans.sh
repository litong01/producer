#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="transcriber"
IMAGE_NAME="transcriber:latest"
PORT="${TRANSCRIBER_PORT:-8080}"

usage() {
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  start                          Start the web UI container"
    echo "  stop                           Stop and remove the container"
    echo "  restart                        Stop then start"
    echo "  status                         Show container status"
    echo "  logs                           Tail container logs"
    echo "  build                          Build the Docker image"
    echo "  test                           Run integration test (童年.pdf → MusicXML; requires 童年.pdf in repo)"
    echo "  run <file(s)> -o <dir> [opts]  CLI: process files without the web UI"
    echo
    echo "Supported inputs:"
    echo "  Images  .jpg .jpeg .png .bmp .tiff .tif .gif .webp .heic .heif"
    echo "  PDF     .pdf"
    echo
    echo "CLI options (for 'run' command):"
    echo "  -o, --output DIR     Output directory (required)"
    echo
    echo "Examples:"
    echo "  $0 start                                      # web UI at http://localhost:8080"
    echo "  $0 run page1.png page2.png -o ./out           # images → PDF + MusicXML"
    echo "  $0 run scores.pdf -o ./out                    # PDF → MusicXML"
    echo
    echo "Environment variables:"
    echo "  TRANSCRIBER_PORT    Port to expose (default: 8080)"
    exit 1
}

do_build() {
    echo "Building image ${IMAGE_NAME} (amd64 for Audiveris)..."
    docker build --platform linux/amd64 -t "${IMAGE_NAME}" .
    echo "Done."
}

do_start() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Container '${CONTAINER_NAME}' is already running."
        echo "  → http://localhost:${PORT}"
        return
    fi

    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    echo "Starting ${CONTAINER_NAME} on port ${PORT} (amd64 for Audiveris)..."
    docker run -d \
        --platform linux/amd64 \
        --name "${CONTAINER_NAME}" \
        -p "${PORT}:8080" \
        "${IMAGE_NAME}"

    echo "Container started."
    echo "  → http://localhost:${PORT}"
}

do_stop() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping ${CONTAINER_NAME}..."
        docker rm -f "${CONTAINER_NAME}" > /dev/null
        echo "Stopped."
    else
        echo "Container '${CONTAINER_NAME}' is not running."
    fi
}

do_status() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Running"
        docker ps --filter "name=^${CONTAINER_NAME}$" --format "table {{.Status}}\t{{.Ports}}"
    elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopped"
        docker ps -a --filter "name=^${CONTAINER_NAME}$" --format "table {{.Status}}"
    else
        echo "Not found — run '$0 start' to create it."
    fi
}

do_logs() {
    docker logs -f "${CONTAINER_NAME}"
}

do_test() {
    local repo_root
    repo_root="$(cd "$(dirname "$0")" && pwd)"
    if [[ ! -f "${repo_root}/童年.pdf" ]]; then
        echo "Error: 童年.pdf not found in ${repo_root}. Place it there to run the integration test." >&2
        exit 1
    fi
    echo "Running integration test (童年.pdf → pipeline → MusicXML with title)..."
    docker run --rm \
        --platform linux/amd64 \
        -v "${repo_root}:/app:ro" \
        -w /app \
        -e PYTHONPATH=/app \
        "${IMAGE_NAME}" \
        bash -c "pip install -q -r requirements-dev.txt 2>/dev/null || true && pytest tests/ -m integration -v --timeout=600 -p no:cacheprovider"
}

do_run() {
    # Parse arguments: collect input files and pass-through flags
    local -a files=()
    local -a cli_args=()
    local output_dir=""
    local parsing_files=true

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -o|--output)
                output_dir="$2"
                shift 2
                parsing_files=false
                ;;
            -*)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
            *)
                files+=("$1")
                shift
                ;;
        esac
    done

    if [[ ${#files[@]} -eq 0 ]]; then
        echo "Error: no input files specified" >&2
        exit 1
    fi
    if [[ -z "$output_dir" ]]; then
        echo "Error: -o / --output is required" >&2
        exit 1
    fi

    mkdir -p "$output_dir"

    # Build volume mounts: mount each input file + the output dir
    local -a volumes=()
    local -a container_inputs=()

    for f in "${files[@]}"; do
        local abs_path
        abs_path="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
        if [[ ! -f "$abs_path" ]]; then
            echo "Error: file not found: $f" >&2
            exit 1
        fi
        volumes+=(-v "${abs_path}:/input/$(basename "$f"):ro")
        container_inputs+=("/input/$(basename "$f")")
    done

    local abs_output
    abs_output="$(cd "$output_dir" && pwd)"
    volumes+=(-v "${abs_output}:/output")

    docker run --rm \
        --platform linux/amd64 \
        "${volumes[@]}" \
        "${IMAGE_NAME}" \
        python -m app.cli "${container_inputs[@]}" -o /output "${cli_args[@]}"
}

case "${1:-}" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; do_start ;;
    status)  do_status ;;
    logs)    do_logs ;;
    build)   do_build ;;
    test)    do_test ;;
    run)     shift; do_run "$@" ;;
    *)       usage ;;
esac
