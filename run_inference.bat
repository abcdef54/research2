@echo off
REM Batch script to run the HumanEval inference and evaluation pipeline on Windows

if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment .venv not found, attempting to run with system python.
)

REM Default parameters (feel free to edit these)
set MODEL=qwen2.5-3b-q4
set MODE=low
set TEMP=0.0
set RANK_MODE=tournament_no_reasoning
set RANK_TEMP=0.0
set SAMPLES=-1
set SEM=4
set OUT=samples.jsonl

echo Running: python inference.py --model %MODEL% --mode %MODE% --temp %TEMP% --rank-mode %RANK_MODE% --rank-temp %RANK_TEMP% --samples %SAMPLES% --sem %SEM% --out %OUT% %*
python inference.py --model %MODEL% --mode %MODE% --temp %TEMP% --rank-mode %RANK_MODE% --rank-temp %RANK_TEMP% --samples %SAMPLES% --sem %SEM% --out %OUT% %*
pause
