#!/bin/bash
# =============================================================================
# XTTS Parallel Job Runner
# Startet mehrere Hörbuch-Jobs gleichzeitig
#
# Nutzung:
#   ./run_parallel.sh           → Normaler Durchgang (Mode 0)
#   ./run_parallel.sh --low 1   → Low Mode 1 (stabil)
#   ./run_parallel.sh --low 2   → Low Mode 2 (chunk_split)
# =============================================================================

# === KONFIGURATION ===
SCRIPT="/workspace/storypainter/voice_generator_modes.py"
PYTHON="/workspace/xtts_env/bin/python3"
LOG_DIR="/workspace/logs"

# Buch-Pfade hier eintragen:
BOOKS=(
    "/workspace/buch1"
    "/workspace/buch2"
    "/workspace/buch3"
    # "/workspace/buch4"   # auskommentieren zum deaktivieren
)
# =====================

# Low-Mode Argument weiterreichen
LOW_ARG=""
if [[ "$1" == "--low" && -n "$2" ]]; then
    LOW_ARG="--low $2"
    echo "🔧 Low Mode $2 aktiv"
fi

# Log-Verzeichnis erstellen
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "🎧 XTTS Parallel Runner"
echo "   Script:  $SCRIPT"
echo "   Jobs:    ${#BOOKS[@]}"
echo "   Mode:    ${LOW_ARG:-"Normal (0)"}"
echo "=============================================="

# Alle Jobs starten
PIDS=()
for BOOK in "${BOOKS[@]}"; do
    BOOK_NAME=$(basename "$BOOK")
    LOG_FILE="$LOG_DIR/${BOOK_NAME}.log"

    echo "▶️  Starte: $BOOK_NAME → Log: $LOG_FILE"

    $PYTHON "$SCRIPT" --path "$BOOK" $LOW_ARG > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "✅ ${#PIDS[@]} Jobs gestartet"
echo "📋 Logs live verfolgen:"
for BOOK in "${BOOKS[@]}"; do
    BOOK_NAME=$(basename "$BOOK")
    echo "   tail -f $LOG_DIR/${BOOK_NAME}.log"
done
echo ""
echo "⏳ Warte auf alle Jobs..."
echo "=============================================="

# Auf alle Jobs warten und Status prüfen
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    BOOK=${BOOKS[$i]}
    BOOK_NAME=$(basename "$BOOK")

    wait "$PID"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $BOOK_NAME fertig"
    else
        echo "❌ $BOOK_NAME fehlgeschlagen (Exit: $EXIT_CODE) → siehe $LOG_DIR/${BOOK_NAME}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "🎉 Alle Jobs erfolgreich abgeschlossen!"
else
    echo "⚠️  $FAILED Job(s) fehlgeschlagen"
fi
echo "=============================================="