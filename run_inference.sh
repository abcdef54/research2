#!/bin/bash
# Shell script to run the HumanEval inference and evaluation pipeline on Unix/Linux/macOS

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment (.venv or venv) not found, running with default python environment."
fi

# Default parameters (feel free to edit these)
MODEL="qwen2.5-3b-q4"
MODE="low"
TEMP="0.0"
RANK_MODE="tournament_no_reasoning"
RANK_TEMP="0.0"
SAMPLES="-1"
SEM="4"
OUT="samples.jsonl"

echo "Running: python inference.py --model $MODEL --mode $MODE --temp $TEMP --rank-mode $RANK_MODE --rank-temp $RANK_TEMP --samples $SAMPLES --sem $SEM --out $OUT $@"
python inference.py --model "$MODEL" --mode "$MODE" --temp "$TEMP" --rank-mode "$RANK_MODE" --rank-temp "$RANK_TEMP" --samples "$SAMPLES" --sem "$SEM" --out "$OUT" "$@"
