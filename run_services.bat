@echo off
setlocal

start "model_service" cmd /k "python model_service.py"
start "streamlit" cmd /k "streamlit run streamlit_app.py"

endlocal
