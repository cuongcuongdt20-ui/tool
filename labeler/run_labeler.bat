@echo off
title Lifeguard Frame Labeler
cd /d "%~dp0"
python desktop_labeler.py
if errorlevel 1 pause
