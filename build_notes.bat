@echo off
chcp 65001 >nul
echo [1Cat-vLLM 笔记生成工具]
echo 正在扫描 learn_notes/markdown/ 目录并生成护眼 HTML...
python "%~dp0build_notes.py"
if %ERRORLEVEL% EQU 0 (
    echo.
    echo [成功] 所有笔记已编译完成！
    echo 索引入口位于: %~dp0index.html
) else (
    echo.
    echo [错误] 编译失败，请检查 Python 环境。
)
pause
