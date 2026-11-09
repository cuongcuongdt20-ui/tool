@echo off
title YOLO Pre-label
cd /d "%~dp0.."
set /p DATASET=Dataset folder (vd: val_yolo):
python labeler\prelabel.py --dataset "%DATASET%"
pause
