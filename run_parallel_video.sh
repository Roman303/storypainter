#!/bin/bash
# =============================================================================
# XTTS Parallel Video Runner
# Startet mehrere Video-Jobs gleichzeitig
#
# Nutzung:
#   ./run_parallel_video.sh
# =============================================================================

# === KONFIGURATION ===
SCRIPT="/workspace/storypainter/video_generator_2.py"
PYTHON="/workspace/xtts_env/bin/python3"
LOG_DIR="/workspace/logs"

# Buch-Pfade hier eintragen:
BOOKS=(
    "/workspace/buch1"
    "/workspace/buch2"
    # "/workspace/buch3"   # auskommentieren zum deaktivieren
)
# =====================

# Log-Verzeichnis erstellen
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "🎬 XTTS Parallel Video Runner"
echo "   Script:  $SCRIPT"
echo "   Jobs:    ${#BOOKS[@]}"
echo "=============================================="

# Alle Jobs starten
PIDS=()
for BOOK in "${BOOKS[@]}"; do
    BOOK_NAME=$(basename "$BOOK")
    LOG_FILE="$LOG_DIR/${BOOK_NAME}_video.log"

    echo "▶️  Starte: $BOOK_NAME → Log: $LOG_FILE"

    $PYTHON "$SCRIPT" --path "$BOOK" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "✅ ${#PIDS[@]} Jobs gestartet"
echo "📋 Logs live verfolgen:"
for BOOK in "${BOOKS[@]}"; do
    BOOK_NAME=$(basename "$BOOK")
    echo "   tail -f $LOG_DIR/${BOOK_NAME}_video.log"
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
        echo "❌ $BOOK_NAME fehlgeschlagen (Exit: $EXIT_CODE) → siehe $LOG_DIR/${BOOK_NAME}_video.log"
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