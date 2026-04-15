#!/usr/bin/env bash

PORT=3001
APP="src.main:app"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.uvicorn.pid"
LOGFILE="$SCRIPT_DIR/uvicorn.log"
CERT_DIR="$SCRIPT_DIR/../certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
VENV_DIR="/home/test/.virtualenvs/search"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_UVICORN="$VENV_DIR/bin/uvicorn"

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
            return 0   # running, PID valid
        fi
    fi
    # Fallback: check if something is actually bound to the port
    if lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | grep -q .; then
        return 0   # port is live even without a valid pidfile
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
        # Wait up to 5 s for the port to be released
        for i in {1..10}; do
            sleep 0.5
            lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | grep -q . || break
        done
    fi

    if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ] || [ ! -f "$ENV_FILE" ]; then
        echo "  ⚠️   Missing prerequisites (cert and/or .env)"
        echo "      Run option 5 first to set them up."
        echo ""
        return
    fi

    cd "$SCRIPT_DIR" || return
    nohup "$VENV_UVICORN" "$APP" --host 0.0.0.0 --port "$PORT" \
        --ssl-certfile "$CERT_FILE" \
        --ssl-keyfile  "$KEY_FILE" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 1

    if is_running; then
        local ip
        ip=$(get_local_ip)
        echo "  ✅  Service started (PID $(cat $PIDFILE))"
        echo "  🌐  API: https://$ip:$PORT/api"
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
        echo "  🌐  https://$ip:$PORT/api"

        # Quick health check
        local health
        health=$(curl -sk -o /dev/null -w "%{http_code}" "https://127.0.0.1:$PORT/api/health" 2>/dev/null)
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

setup_cert() {
    echo ""
    echo "  ── Step 1/3: Self-signed cert ──"
    if ! command -v openssl >/dev/null 2>&1; then
        echo "  ❌  openssl not found — please install it first"
        echo ""
        return
    fi

    mkdir -p "$CERT_DIR"

    local do_gen=1
    if [ -f "$CERT_FILE" ] || [ -f "$KEY_FILE" ]; then
        printf "  ⚠️   Cert already exists at $CERT_DIR — overwrite? [y/N]: "
        read -r confirm
        case "$confirm" in
            y|Y|yes|YES) ;;
            *) do_gen=0 ;;
        esac
    fi

    if [ "$do_gen" = "1" ]; then
        local ip
        ip=$(get_local_ip)
        openssl req -x509 -newkey rsa:2048 -nodes \
            -keyout "$KEY_FILE" -out "$CERT_FILE" \
            -days 365 -subj "/CN=$ip" \
            -addext "subjectAltName=IP:$ip,IP:127.0.0.1,DNS:localhost" \
            >/dev/null 2>&1

        if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
            chmod 600 "$KEY_FILE"
            echo "  ✅  Self-signed cert generated"
            echo "      cert: $CERT_FILE"
            echo "      key:  $KEY_FILE"
        else
            echo "  ❌  Failed to generate cert"
            echo ""
            return
        fi
    else
        echo "  ↪  Kept existing cert."
    fi

    echo ""
    echo "  ── Step 2/3: .env file ──"
    if [ -f "$ENV_FILE" ]; then
        echo "  ↪  $ENV_FILE already exists — kept."
    else
        if [ ! -f "$ENV_EXAMPLE" ]; then
            echo "  ❌  $ENV_EXAMPLE not found — cannot create .env"
            echo ""
            return
        fi

        echo "  Two databases are required:"
        echo "    1) Crawled content DB  (DATABASE_URL)      — stores crawled_items & embeddings"
        echo "    2) User data DB        (USER_DATABASE_URL) — stores mzhu_test_ logs/interventions (collect repo)"
        echo ""

        echo "  -- Crawled content DB --"
        printf "    host     [localhost]: "; read -r db_host;   db_host="${db_host:-localhost}"
        printf "    port     [5432]:      "; read -r db_port;   db_port="${db_port:-5432}"
        printf "    user     [dbuser]:    "; read -r db_user;   db_user="${db_user:-dbuser}"
        printf "    password:             "; read -rs db_pass;  echo ""
        printf "    database [autism_crawler]: "; read -r db_name; db_name="${db_name:-autism_crawler}"
        db_url="postgresql://${db_user}:${db_pass}@${db_host}:${db_port}/${db_name}"

        echo ""
        echo "  -- User data DB (collect) --"
        printf "    host     [localhost]: "; read -r udb_host;  udb_host="${udb_host:-localhost}"
        printf "    port     [5432]:      "; read -r udb_port;  udb_port="${udb_port:-5432}"
        printf "    user     [dbuser]:    "; read -r udb_user;  udb_user="${udb_user:-dbuser}"
        printf "    password:             "; read -rs udb_pass; echo ""
        printf "    database [mzhu_test_autism_users]: "; read -r udb_name; udb_name="${udb_name:-mzhu_test_autism_users}"
        user_db_url="postgresql://${udb_user}:${udb_pass}@${udb_host}:${udb_port}/${udb_name}"

        cp "$ENV_EXAMPLE" "$ENV_FILE"
        sed -i "s|^DATABASE_URL=.*|DATABASE_URL=$db_url|" "$ENV_FILE"
        sed -i "s|^USER_DATABASE_URL=.*|USER_DATABASE_URL=$user_db_url|" "$ENV_FILE"

        echo "  ✅  Created $ENV_FILE"
        echo "      DATABASE_URL      = postgresql://${db_user}:****@${db_host}:${db_port}/${db_name}"
        echo "      USER_DATABASE_URL = postgresql://${udb_user}:****@${udb_host}:${udb_port}/${udb_name}"
        echo "  ✏️   Edit $ENV_FILE to adjust other values (COLLECT_BASE_URL, LOG_LEVEL, etc.)"
    fi

    echo ""
    echo "  ── Step 3/3: pip install -r requirements.txt ──"
    if [ ! -x "$VENV_PIP" ]; then
        echo "  ❌  venv pip not found at $VENV_PIP"
        echo ""
        return
    fi
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        echo "  ❌  requirements.txt not found in $SCRIPT_DIR"
        echo ""
        return
    fi

    "$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ✅  Dependencies installed"
    else
        echo "  ❌  pip install failed (exit $rc)"
    fi
    echo ""
}

show_api_url() {
    echo ""
    local ip
    ip=$(get_local_ip)
    echo "  🌐  Base URL : https://$ip:$PORT"
    echo ""
    echo "  Endpoints:"
    echo "    GET  https://$ip:$PORT/api/search?q=<query>"
    echo "    GET  https://$ip:$PORT/api/stats"
    echo "    GET  https://$ip:$PORT/api/health"
    echo ""
    echo "  Docs:"
    echo "    Swagger UI : https://$ip:$PORT/docs"
    echo "    OpenAPI    : https://$ip:$PORT/openapi.json"
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
    echo "║  5) Setup cert / .env / deps     ║"
    echo "║  0) Exit                         ║"
    echo "╚══════════════════════════════════╝"
    printf "  Choose an option: "
    read -r choice

    case "$choice" in
        1) start_service ;;
        2) stop_service ;;
        3) service_status ;;
        4) show_api_url ;;
        5) setup_cert ;;
        0) echo ""; echo "  Bye!"; echo ""; exit 0 ;;
        *) echo ""; echo "  ⚠️  Invalid option, try again."; echo "" ;;
    esac
done
