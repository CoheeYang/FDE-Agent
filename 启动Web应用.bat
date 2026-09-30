@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 启动 FDE 售前智能体 Web 应用...
echo 浏览器访问 http://localhost:8501 （首次启动需等待本地向量模型加载）
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
