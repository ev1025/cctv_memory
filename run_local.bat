@echo off
REM ============================================================
REM  로컬(RTX 4060 8GB) FastAPI 실행 런처
REM  - GPU 0번 사용 + 4bit 양자화(7B 를 ~6GB 로 적재)
REM  - 실행 전 무거운 앱(Chrome/Edge 등)을 닫아 GPU 를 비우세요(8GB 거의 다 필요).
REM  - 브라우저에서 http://127.0.0.1:8000/docs  (Swagger UI 로 3개 엔드포인트 테스트)
REM ============================================================
set CUDA_VISIBLE_DEVICES=0
set LOAD_IN_4BIT=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd /d "%~dp0"
"C:\Users\eg287\venvs\3dvision\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
