#!/usr/bin/env bash

PORT=3001
APP="src.main:app"
PIDFILE=".uvicorn.pid"
LOGFILE="uvicorn.log"

# ── Helpers ────────────────────────────────────────────────────────────────────

get_local_ip() {
    # Try Linux first, then macOS
    ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' \
    || ipconfig getifaddr en0 2>/dev/null \
    || ipconfig getifaddr en1 2>/dev/null \
    || hostname -I 2>/dev/null | awk '{print $1}' \
    || echo "127.0.0.1"
}

is_running() {
    if [ -f "$PIDFILE" ]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0   # running
        fi
    fi
    return 1   # not running
}

# ── Actions ────────────────────────────────────────────────────────────────────

start_service() {
    echo ""
    if is_running; then
        local pid
        pid=$(cat "$PIDFILE")
        echo "  ↻  Service already running (PID $pid) — restarting..."
        kill "$pid" 2>/dev/null
        sleep 1
    fi

    nohup uvicorn "$APP" --host 0.0.0.0 --port "$PORT" > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1

    if is_running; then
        local ip
        ip=$(get_local_ip)
        echo "  ✅  Service started (PID $(cat $PIDFILE))"
        echo "  🌐  API: http://$ip:$PORT/api"
        echo "  📄  Logs: $LOGFILE"
    else
        echo "  ❌  Failed to start — check $LOGFILE for details"
    fi
    echo ""
}

service_status() {
    echo ""
    if is_running; then
        local pid ip
        pid=$(cat "$PIDFILE")
        ip=$(get_local_ip)
        echo "  ✅  Service is running (PID $pid)"
        echo "  🌐  http://$ip:$PORT/api"

        # Quick health check
        local health
        health=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/health" 2>/dev/null)
        if [ "$health" = "200" ]; then
            echo "  💚  Health check: OK (HTTP 200)"
        else
            echo "  ⚠️   Health check: no response (HTTP $health)"
        fi
    else
        echo "  🔴  Service is NOT running"
    fi
    echo ""
}

stop_service() {
    echo ""
    if is_running; then
        local pid
        pid=$(cat "$PIDFILE")
        kill "$pid" 2>/dev/null
        rm -f "$PIDFILE"
        echo "  🛑  Service stopped (PID $pid)"
    else
        echo "  ⚠️   Service is not running"
    fi
    echo ""
}

show_api_url() {
    echo ""
    local ip
    ip=$(get_local_ip)
    echo "  🌐  Base URL : http://$ip:$PORT"
    echo ""
    echo "  Endpoints:"
    echo "    GET  http://$ip:$PORT/api/search?q=<query>"
    echo "    GET  http://$ip:$PORT/api/stats"
    echo "    GET  http://$ip:$PORT/api/health"
    echo ""
    echo "  Docs:"
    echo "    Swagger UI : http://$ip:$PORT/docs"
    echo "    OpenAPI    : http://$ip:$PORT/openapi.json"
    echo ""
}

# ── Menu ───────────────────────────────────────────────────────────────────────

while true; do
    echo "╔══════════════════════════════════╗"
    echo "║       Autism Search Service      ║"
    echo "╠══════════════════════════════════╣"
    echo "║  1) Start / Restart service      ║"
    echo "║  2) Stop service                 ║"
    echo "║  3) Service status               ║"
    echo "║  4) Show API URL                 ║"
    echo "║  0) Exit                         ║"
    echo "╚══════════════════════════════════╝"
    printf "  Choose an option: "
    read -r choice

    case "$choice" in
        1) start_service ;;
        2) stop_service ;;
        3) service_status ;;
        4) show_api_url ;;
        0) echo ""; echo "  Bye!"; echo ""; exit 0 ;;
        *) echo ""; echo "  ⚠️  Invalid option, try again."; echo "" ;;
    esac
done
