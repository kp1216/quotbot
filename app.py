# app.py
import os
import mimetypes
from pathlib import Path
import pandas as pd
import google.generativeai as genai
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()
API_KEY   = os.getenv("GEMINI_API")
MODELNAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

BASE_SYSTEM_PROMPT = (
    "You are a friendly assistant. Answer naturally like ChatGPT.\n"
    "If an Inventory Snapshot is provided below, prefer to ground inventory-related answers in it. "
    "Do not fabricate values; if the snapshot lacks required info, say exactly what is missing and ask for it. "
    "Use concise language and small Markdown tables when helpful.\n\n"
    "IMPORTANT: Only reference the snapshot when relevant. For general questions, answer normally."
)

if not API_KEY:
    print("WARNING: GEMINI_API_KEY is missing in .env")
else:
    genai.configure(api_key=API_KEY)

def build_model(system_instruction: str):
    return genai.GenerativeModel(model_name=MODELNAME, system_instruction=system_instruction)

# ---------- MIME type helper ----------
MIME_OVERRIDES = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv":  "text/csv",
    ".json": "application/json",
    ".md":   "text/markdown",
    ".txt":  "text/plain",
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp"
}
def guess_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in MIME_OVERRIDES:
        return MIME_OVERRIDES[ext]
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"
# --------------------------------------

def read_excel_all_sheets(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xlsx":
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    elif ext == ".xls":
        sheets = pd.read_excel(path, sheet_name=None, engine="xlrd")
    else:
        raise ValueError("Only Excel files are supported: .xlsx or .xls")
    frames = []
    for sname, df in sheets.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            d2 = df.copy()
            d2["__sheet__"] = sname
            frames.append(d2)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def make_snapshot(df: pd.DataFrame, max_rows: int = 400, max_chars: int = 60_000) -> str:
    """Smaller caps to avoid overlong system prompts that can trip the API."""
    if df is None or df.empty:
        return "NO_DATA"
    head = "Columns: " + " | ".join([str(c) for c in df.columns]) + f"\nTotal Rows: {len(df)}"
    csv = df.head(max_rows).to_csv(index=False)
    if len(csv) > max_chars:
        csv = csv[:max_chars] + "\n...TRUNCATED..."
    return head + "\n\nCSV Preview (capped):\n" + csv

def build_system_with_snapshot(snapshot: str | None) -> str:
    if snapshot and snapshot.strip() and snapshot != "NO_DATA":
        return BASE_SYSTEM_PROMPT + (
            "\n---\nINVENTORY SNAPSHOT (for grounding when relevant). "
            "Do not dump this verbatim; cite only the bits you use:\n" + snapshot
        )
    return BASE_SYSTEM_PROMPT

async def ensure_chat():
    chat = cl.user_session.get("chat")
    if chat is None:
        model = cl.user_session.get("model")
        if model is None:
            sys = build_system_with_snapshot(cl.user_session.get("snapshot"))
            model = build_model(sys)
            cl.user_session.set("model", model)
        chat = model.start_chat(history=[])
        cl.user_session.set("chat", chat)
    return chat

async def rebuild_with_snapshot(snapshot: str | None):
    cl.user_session.set("snapshot", snapshot)
    sys = build_system_with_snapshot(snapshot)
    model = build_model(sys)
    cl.user_session.set("model", model)
    chat = model.start_chat(history=[])
    cl.user_session.set("chat", chat)
    return chat

def is_excel(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".xlsx", ".xls")

# SVG loader (three bouncing dots)
LOADER_HTML = """<div style="display:inline-block; padding:6px 2px;">
  <svg width="60" height="18" viewBox="0 0 60 18" xmlns="http://www.w3.org/2000/svg">
    <circle cx="10" cy="9" r="4" fill="currentColor">
      <animate attributeName="cy" values="9;3;9" dur="0.8s" repeatCount="indefinite" begin="0s"/>
    </circle>
    <circle cx="30" cy="9" r="4" fill="currentColor">
      <animate attributeName="cy" values="9;3;9" dur="0.8s" repeatCount="indefinite" begin="0.15s"/>
    </circle>
    <circle cx="50" cy="9" r="4" fill="currentColor">
      <animate attributeName="cy" values="9;3;9" dur="0.8s" repeatCount="indefinite" begin="0.30s"/>
    </circle>
  </svg>
</div>"""

@cl.on_chat_start
async def on_start():
    sys = build_system_with_snapshot(None)
    model = build_model(sys)
    chat = model.start_chat(history=[])

    cl.user_session.set("model", model)
    cl.user_session.set("chat", chat)
    cl.user_session.set("snapshot", None)

    await cl.Message(
        content=(
            "Hi! 👋 I chat like ChatGPT.\n\n"
            "Use the **paperclip in the input box** to attach your Excel inventory (and any PDFs/images). "
            "I’ll remember a snapshot of your inventory and use it when relevant."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    text = message.content or ""
    files = message.elements or []

    excel_paths, other_paths = [], []
    for el in files:
        path = getattr(el, "path", None) or getattr(el, "url", None)
        if not path:
            continue
        (excel_paths if is_excel(path) else other_paths).append(path)

    # If an Excel is present, (re)build snapshot
    if excel_paths:
        try:
            df = read_excel_all_sheets(excel_paths[-1])  # last Excel wins
            snap = make_snapshot(df)
            await rebuild_with_snapshot(snap)
            await cl.Message(
                content=f"✅ Inventory attached ({len(df)} rows, {len(df.columns)} columns). I’ll use it when relevant."
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ Error reading Excel: {e}").send()

    # Ensure chat session
    chat = await ensure_chat()

    # If there’s no user text and no non-Excel files, stop (prevents 400 on blank prompt)
    if not text.strip() and not other_paths:
        await cl.Message(
            content="📎 Inventory noted. Now type a question (e.g., *“quote 25 pcs of item X”*), or attach a PDF/image to analyze."
        ).send()
        return

    # Show loader
    loader = cl.Message(content=LOADER_HTML)
    await loader.send()

    # Upload non-Excel attachments to Gemini (with mime_type)
    gem_files = []
    for p in other_paths:
        try:
            mt = guess_mime_type(p)
            fh = genai.upload_file(path=p, mime_type=mt)
            gem_files.append(fh)
        except Exception as e:
            await cl.Message(
                content=f"⚠️ Couldn’t upload an attachment: {os.path.basename(p)} ({e})"
            ).send()

    # Stream reply
    try:
        if gem_files:
            content = [text] + gem_files if text else gem_files
            resp = chat.send_message(content, stream=True)
        else:
            # text is guaranteed non-empty here
            resp = chat.send_message(text, stream=True)

        first = True
        for chunk in resp:
            token = getattr(chunk, "text", None)
            if not token:
                continue
            if first:
                loader.content = token
                await loader.update()       # swap HTML → first tokens
                first = False
            else:
                await loader.stream_token(token)
        await loader.update()

    except TypeError:
        try:
            if gem_files:
                content = [text] + gem_files if text else gem_files
                full = chat.send_message(content)
            else:
                full = chat.send_message(text)  # no blank " "
            loader.content = full.text or "(No response.)"
            await loader.update()
        except Exception as e:
            loader.content = f"❌ Gemini error: {e}"
            await loader.update()
    except Exception as e:
        loader.content = f"❌ Gemini error: {e}"
        await loader.update()
