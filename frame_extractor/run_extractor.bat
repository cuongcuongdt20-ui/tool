@echo off
title Video Frame Extractor
cd /d "%~dp0"
python video_frame_extractor.py
if errorlevel 1 pause
