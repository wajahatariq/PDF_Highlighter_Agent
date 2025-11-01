import streamlit as st
import tempfile
import fitz  # PyMuPDF
import requests
import json
import ast
from typing import List

# ---------- CONFIG ----------
st.set_page_config(page_title="PDF Highlighter Agent", layout="wide")

# Groq config stored in st.secrets
groq_key = st.secrets["groq"]["api_key"]
groq_model = st.secrets["groq"].get("model", "llama-3.1-8b-instant")

# ---------- UI ----------
st.title("PDF Highlighter Agent (Groq)")

st.write(
    "Upload one or more CV PDFs, and the app will highlight all company names "
    "where the person has worked with a red backdrop."
)

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Fixed highlight color: red
color_choice = "red"
opacity = st.slider("Backdrop opacity (0 = transparent, 1 = opaque)", 0.1, 0.9, 0.45)
process = st.button("Process PDFs")

# ---------- Helpers ----------

def call_groq_chat(pdf_text: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are a JSON-only assistant that extracts the exact names of companies, organizations, or workplaces "
        "where the candidate has experience or has worked, from the provided CV text. "
        "Include any short phrases that look like company or workplace names, even if they appear without full sentences. "
        "Return only a JSON list of strings containing these names, with no extra text."
    )

    user_msg = (
        "Extract and return a JSON array of all company names, organizations, or workplaces mentioned in the CV text below, "
        "including short phrases under experience or work history sections.\n"
        "Only return a JSON array of exact strings to highlight in the PDF."
        f"\n\nCV_TEXT_START\n{pdf_text}\nCV_TEXT_END"
    )

    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        st.error(f"Groq API Error {response.status_code}: {response.text}")
        return ""
    data = response.json()
    return data["choices"][0]["message"]["content"]

def parse_model_output(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, (str, int, float))]
    except Exception:
        pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if isinstance(x, (str, int, float))]
        except Exception:
            try:
                parsed = ast.literal_eval(snippet)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if isinstance(x, (str, int, float))]
            except Exception:
                pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, (str, int, float))]
    except Exception:
        pass
    return []

def color_to_rgb_tuple(color_name: str):
    mapping = {
        "black": (0, 0, 0),
        "yellow": (1, 1, 0),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
    }
    return mapping.get(color_name.lower(), (1, 0, 0))  # default to red

def highlight_pdf_with_backdrop(input_path: str, output_path: str, targets: List[str], rgb_fill: tuple, opacity_val: float):
    doc = fitz.open(input_path)
    for page in doc:
        found_rects = []
        for t in targets:
            if not t.strip():
                continue
            variants = [t, t.strip(), t.strip().title(), t.strip().upper(), t.strip().lower()]
            rects = []
            for v in variants:
                try:
                    r = page.search_for(v, hit_max=128)
                    if r:
                        rects.extend(r)
                except Exception:
                    pass
            for r in rects:
                r_inflated = fitz.Rect(r.x0 - 1, r.y0 - 0.5, r.x1 + 1, r.y1 + 0.5)
                found_rects.append(r_inflated)
        for rect in found_rects:
            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=None, fill=rgb_fill)
                annot.set_opacity(opacity_val)
                annot.update()
            except Exception:
                try:
                    page.draw_rect(rect, fill=rgb_fill)
                except Exception:
                    pass
    doc.save(output_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

# ---------- Main ----------

if process:
    if not uploaded_files:
        st.error("Please upload one or more PDFs first.")
    else:
        st.info("Processing PDFs. This may take a few moments.")
        rgb = color_to_rgb_tuple(color_choice)
        for uploaded in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(uploaded.read())
                tmp_in.flush()

                doc = fitz.open(tmp_in.name)
                full_text = []
                for p in doc:
                    try:
                        full_text.append(p.get_text("text"))
                    except Exception:
                        full_text.append("")
                doc.close()

                text_for_model = "\n".join(full_text)
                text_for_model = text_for_model[:13000]

                # Debug: show extracted text (optional)
                # st.text_area("Extracted text sent to model", text_for_model, height=300)

                try:
                    raw = call_groq_chat(text_for_model)
                except Exception as e:
                    st.error(f"LLM call failed for {uploaded.name}: {e}")
                    continue

                targets = parse_model_output(raw)
                if not targets:
                    st.warning(f"No company names found to highlight for {uploaded.name}.")
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
                    try:
                        highlight_pdf_with_backdrop(tmp_in.name, tmp_out.name, targets, rgb, opacity)
                    except Exception as e:
                        st.error(f"PDF highlighting failed for {uploaded.name}: {e}")
                        continue

                    st.success(f"Processed: {uploaded.name}")
                    with open(tmp_out.name, "rb") as fh:
                        st.download_button(
                            label=f"Download highlighted {uploaded.name}",
                            data=fh,
                            file_name=f"highlighted_{uploaded.name}",
                            mime="application/pdf",
                        )

        st.info("Done processing all files.")
