@echo off
cd /d C:\Users\kr034\OneDrive\Desktop\SONAR\SONAR
.venv\Scripts\python.exe scripts/train_yolo.py --epochs 30 --batch 8 --name drishti-ss_yolov8n_e30 > train_full_log.txt 2>&1
