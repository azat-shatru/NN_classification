@echo off
echo ── NN Pipeline Setup ──────────────────────────────────────
echo Installing required packages...
pip install -r requirements.txt
echo.
echo ── Launching app ──────────────────────────────────────────
streamlit run app/app.py
pause
