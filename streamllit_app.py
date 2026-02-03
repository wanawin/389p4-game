# streamlit_app.py  (auto-detected entrypoint for Streamlit Cloud)
from pathlib import Path

TARGET = Path(__file__).parent / "389p4 streamlit app.py"

code = TARGET.read_text(encoding="utf-8")
exec(compile(code, str(TARGET), "exec"), {"__name__": "__main__"})
