@echo off
title DEADKEY - Epistemic Verification Lab
cd /d "%~dp0"
python -m deadkey.doctor
start "" "http://127.0.0.1:8765"
python -m deadkey.server --cartridge cartridges\blank-lab\manifest.json --tools cartridges\blank-lab\tools_catalog.json --port 8765
pause
