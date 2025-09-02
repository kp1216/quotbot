# app.py
import os
import mimetypes
from pathlib import Path
import hashlib
import uuid

import pandas as pd
import google.generativeai as genai
import chainlit as cl
from dotenv import load_dotenv
from supabase import create_client, Client

# ----------------- ENV & MODEL -----------------
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
    print("⚠️ WARNING: GEMINI_API is missing in .env or Secrets")
else:
    genai.configure(api_key=API_KEY)

def build_model(system_instruction: str):
    return genai.GenerativeModel(model_name=MODELNAME, system_instruction=system_instruction)

# ----------------- MIME HELPER -----------------
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
    ".webp": "image/webp",
}
def guess_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in MIME_OVERRIDES:
        return MIME_OVERRIDES[ext]
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

# ----------------- SUPABASE -----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_KEY")  # fallback to old name
)
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "pins")

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("⚠️ Supabase not configured (missing URL or KEY)")

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

def pin_file_supabase(user_id: str, local_path: str, mime_type: str, overwrite: bool = False) -> dict:
    """
    Upload file to Supabase Storage and insert metadata into 'pinned_files'.
    """
    assert supabase is not None, "Supabase not configured"
    filename    = os.path.basename(local_path)
    digest      = _sha256(local_path)
    size_bytes  = os.path.getsize(local_path)
    storage_key = f"{user_id}/{digest}_{filename}"

    file_options = {"content-type": mime_type}
    if overwrite:
        file_options["upsert"] = "true"

    with open(local_path, "rb") as f:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            storage_key, f, file_options
        )

    row = {
        "user_id": user_id,
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": size_bytes,
        "sha256": digest,
        "storage_path": storage_key,
    }
    supabase.table("pinned_files").insert(row).execute()
    return row

def list_pins_supabase(user_id: str) -> list[dict]:
    assert supabase is not None, "Supabase not configured"
    res = supabase.table("pinned_files").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
    return res.data or []

def signed_url(storage_key: str, expires_sec: int = 3600) -> str:
    assert supabase is not None, "Supabase not configured"
    return supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(storage_key, expires_sec)["signedURL"]

# ----------------- DATA HELPERS -----------------
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

def make_snapshot(df: pd.DataFrame, max_rows: int = 120, max_cols: int = 20, max_chars: int = 18_000) -> str:
    if df is None or df.empty:
        return "NO_DATA"

    use_cols = list(df.columns)[:max_cols]
    dsmall = df[use_cols].copy()
    for c in use_cols:
        dsmall[c] = dsmall[c].astype(str)

    head = (
        "Columns: " + " | ".join([str(c) for c in use_cols]) +
        f"\nTotal Rows: {len(df)} (showing first {min(len(df), max_rows)})"
    )
    csv = dsmall.head(max_rows).to_csv(index=False)

    if len(csv) > max_chars:
        csv = csv[:max_chars] + "\n...TRUNCATED..."

    snap = head + "\n\nCSV Preview (capped):\n" + csv

    if len(snap) > max_chars + 2000:
        snap = (
            "Columns: " + " | ".join([str(c) for c in use_cols]) +
            f"\nTotal Rows: {len(df)}\n\nCSV Preview (capped):\n(TRUNCATED)"
        )
    return snap

def build_system_with_snapshot(snapshot: str | None) -> str:
    base = BASE_SYSTEM_PROMPT
    if snapshot and snapshot.strip() and snapshot != "NO_DATA":
        chunk = snapshot[:20_000]
        return base + (
            "\n---\nINVENTORY SNAPSHOT (for grounding when relevant). "
            "Do not dump this verbatim; cite only the bits you use:\n" + chunk
        )
    return base

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

# ----------------- UI -----------------
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

    if not cl.user_session.get("user_id"):
        cl.user_session.set("user_id", str(uuid.uuid4()))

    await cl.Message(
        content="Click to view your pinned files.",
        actions=[cl.Action(name="show_pins", value="show", label="📎 Pinned Files", payload={})]
    ).send()

    await cl.Message(
        content=(
            "Hi! 👋 I chat like ChatGPT.\n\n"
            "Use the **paperclip** to attach your Excel inventory (and PDFs/images). "
            "I’ll remember a snapshot and use it when relevant."
        )
    ).send()

# ----------------- MESSAGE HANDLER -----------------
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

    if excel_paths:
        try:
            df = read_excel_all_sheets(excel_paths[-1])
            snap = make_snapshot(df)
            await rebuild_with_snapshot(snap)
            await cl.Message(
                content=f"✅ Inventory attached ({len(df)} rows, {len(df.columns)} columns). I’ll use it when relevant."
            ).send()

            if supabase:
                try:
                    mt_x = guess_mime_type(excel_paths[-1])
                    pin_file_supabase(cl.user_session.get("user_id"), excel_paths[-1], mt_x, overwrite=True)
                except Exception as pe:
                    print("Supabase pin (excel) failed:", repr(pe))
        except Exception as e:
            await cl.Message(content=f"❌ Error reading Excel: {e}").send()

    chat = await ensure_chat()

    if not text.strip() and not other_paths:
        await cl.Message(
            content="📎 Inventory noted. Now type a question (e.g., *“quote 25 pcs of item X”*), or attach a PDF/image."
        ).send()
        return

    loader = cl.Message(content=LOADER_HTML)
    await loader.send()

    gem_files = []
    for p in other_paths:
        try:
            mt = guess_mime_type(p)
            fh = genai.upload_file(path=p, mime_type=mt)
            gem_files.append(fh)
            if supabase:
                try:
                    pin_file_supabase(cl.user_session.get("user_id"), p, mt, overwrite=True)
                except Exception as pe:
                    print("Supabase pin failed:", repr(pe))
        except Exception as e:
            await cl.Message(content=f"⚠️ Couldn’t upload: {os.path.basename(p)} ({e})").send()

    try:
        if gem_files:
            content = [text] + gem_files if text else gem_files
            resp = chat.send_message(content, stream=True)
        else:
            resp = chat.send_message(text, stream=True)

        first = True
        for chunk in resp:
            token = getattr(chunk, "text", None)
            if not token:
                continue
            if first:
                loader.content = token
                await loader.update()
                first = False
            else:
                await loader.stream_token(token)
        await loader.update()
    except Exception as e:
        print("Gemini error detail:", repr(e))
        loader.content = f"❌ Gemini error: {e}"
        await loader.update()

# ----------------- ACTION: SHOW PINS -----------------
@cl.action_callback("show_pins")
async def _show_pins(action):
    if not supabase:
        await cl.Message(content="Pins DB not configured.").send()
        return

    user_id = cl.user_session.get("user_id")
    rows = list_pins_supabase(user_id)
    if not rows:
        await cl.Message(content="No pinned files yet.").send()
        return

    lines = []
    for r in rows[:20]:
        try:
            url = signed_url(r["storage_path"], 3600)
            size = r.get("size_bytes") or 0
            lines.append(f"- [{r['filename']}]({url})  \n  {size} bytes")
        except Exception:
            lines.append(f"- {r['filename']} (cannot create link)")
    await cl.Message(content="**Pinned files:**\n" + "\n".join(lines)).send()
