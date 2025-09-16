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

    gem_files = []

    # If an Excel file is uploaded, convert to CSV before uploading
    if excel_paths:
        try:
            for excel_path in excel_paths:
                # Convert Excel file to CSV(s)
                csv_paths = excel_to_csv_paths(
                    excel_path,
                    mode=EXCEL_SHEETS_MODE,
                    include_index=INCLUDE_INDEX_IN_CSV,
                    max_sheets=MAX_SHEETS_TO_UPLOAD
                )

                # Upload each CSV file to Gemini
                for p in csv_paths:
                    try:
                        mt = guess_mime_type(p)  # Get MIME type for CSV
                        print(f"Uploading {os.path.basename(p)} with MIME type: {mt}")  # Debug line to see the MIME type
                        if mt == "application/octet-stream":
                            raise ValueError(f"Unsupported MIME type: {mt} for {p}")
                        fh = genai.upload_file(path=p, mime_type=mt)
                        gem_files.append(fh)
                        if supabase:
                            try:
                                pin_file_supabase(cl.user_session.get("user_id"), p, mt, overwrite=True)
                            except Exception as pe:
                                print("Supabase pin (csv) failed:", repr(pe))
                    except Exception as up_e:
                        await cl.Message(content=f"⚠️ Couldn’t upload CSV '{os.path.basename(p)}' to Gemini: {up_e}").send()

            await cl.Message(
                content=f"📄 Sent {len(csv_paths)} CSV file(s) from the workbook to Gemini."
            ).send()

        except Exception as e:
            await cl.Message(content=f"❌ Error handling Excel: {e}").send()

    chat = await ensure_chat()

    if not text.strip() and not other_paths and not gem_files:
        await cl.Message(
            content="📎 Inventory noted & CSVs uploaded. Now type a question (e.g., *“quote 25 pcs of item X”*), or attach a PDF/image."
        ).send()
        return

    loader = cl.Message(content=LOADER_HTML)
    await loader.send()

    # Upload any non-Excel attachments (PDF, images, etc.) to Gemini + pin
    for p in other_paths:
        try:
            mt = guess_mime_type(p)  # Correct MIME type for each file
            print(f"Uploading {os.path.basename(p)} with MIME type: {mt}")
            if mt == "application/octet-stream":
                raise ValueError(f"Unsupported MIME type: {mt} for {p}")
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
