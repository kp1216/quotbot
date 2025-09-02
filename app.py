import os
import json
import chainlit as cl
import google.generativeai as genai
import pandas as pd
from supabase import create_client, Client
from datetime import datetime

# =============================
# Supabase Setup
# =============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# Gemini Setup
# =============================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro")

# =============================
# Helpers
# =============================

def read_excel_all_sheets(file_path: str) -> pd.DataFrame:
    """Read all sheets in an Excel file and return as a single merged DataFrame."""
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}")

def make_snapshot(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Generate a snapshot of a DataFrame for context injection."""
    snapshot = df.head(max_rows).to_string(index=False)
    if len(df) > max_rows:
        snapshot += f"\n... ({len(df) - max_rows} more rows)"
    return snapshot

def upload_to_supabase(file_path: str, bucket: str = "uploads") -> str:
    """Upload a file to Supabase storage and return its public URL."""
    file_name = os.path.basename(file_path)
    supabase.storage.from_(bucket).upload(file_name, file_path, {"upsert": True})
    public_url = supabase.storage.from_(bucket).get_public_url(file_name)
    return public_url

def convert_excel_to_csv(excel_path: str) -> str:
    """Convert Excel (.xls/.xlsx) to CSV for Gemini upload."""
    df = read_excel_all_sheets(excel_path)
    csv_path = os.path.splitext(excel_path)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    return csv_path

# =============================
# Chainlit Events
# =============================

@cl.on_chat_start
async def start():
    await cl.Message(content="Hi! Upload files (.pdf, .csv, .xlsx, .xls, images, audio, video) and ask me questions!").send()

@cl.on_message
async def on_message(message: cl.Message):
    gem_files = []
    supabase_files = []
    snapshots = []

    for f in message.elements:
        file_path = f.path
        ext = os.path.splitext(file_path)[1].lower()

        # Store original file in Supabase
        file_url = upload_to_supabase(file_path)
        supabase_files.append({"name": f.name, "url": file_url})

        if ext in (".xlsx", ".xls"):
            try:
                # Convert to CSV for Gemini
                csv_path = convert_excel_to_csv(file_path)
                fh = genai.upload_file(path=csv_path)
                gem_files.append(fh)

                # Snapshot for context
                df = read_excel_all_sheets(file_path)
                snapshots.append(make_snapshot(df))

            except Exception as e:
                await cl.Message(content=f"⚠️ Error processing Excel: {f.name} ({e})").send()

        elif ext in (".csv", ".pdf", ".png", ".jpg", ".jpeg", ".gif",
                     ".mp4", ".mov", ".avi", ".wav", ".mp3", ".aac"):
            try:
                fh = genai.upload_file(path=file_path)
                gem_files.append(fh)

                if ext == ".csv":
                    df = pd.read_csv(file_path)
                    snapshots.append(make_snapshot(df))

            except Exception as e:
                await cl.Message(content=f"⚠️ Error uploading {f.name}: {e}").send()

        else:
            await cl.Message(content=f"⚠️ Skipped unsupported file: {f.name}").send()

    # Save metadata in Supabase DB
    supabase.table("pinned_files").insert({
        "date": int(datetime.now().timestamp() * 1000),
        "files": json.dumps(supabase_files)
    }).execute()

    # Build prompt
    prompt = message.content
    if snapshots:
        prompt += "\n\nHere are file snapshots for context:\n" + "\n---\n".join(snapshots)

    # Generate response
    try:
        resp = model.generate_content([prompt] + gem_files)
        await cl.Message(content=resp.text).send()
    except Exception as e:
        await cl.Message(content=f"⚠️ Gemini error: {e}").send()
