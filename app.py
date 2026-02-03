# app.py (launcher)
from pathlib import Path

target = Path(__file__).parent / "389p4 streamlit app.py"
code = target.read_text(encoding="utf-8")
exec(compile(code, str(target), "exec"), {"__name__": "__main__"})
