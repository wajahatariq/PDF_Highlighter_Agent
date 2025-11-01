import streamlit as st
import tempfile
import fitz  # PyMuPDF
import requests
import json
import ast
from typing import List

# ---------- CONFIG ----------
st.set_page_config(page_title="PDF Highlighter Agent", layout="wide")

# Groq config should be stored in st.secrets
# .streamlit/secrets.toml should have [groq] api_key and model
groq_key = st.secrets["groq"]["api_key"]
groq_model = st.secrets["groq"].get("model", "mixtral-8x7b-32768")
groq_base = "https://api.groq.com/openai/v1"  # OpenAI-compatible endpoint

# ---------- UI ----------
st.title("PDF Highlighter Agent (Groq)")

st.write(
    "Upload one or more PDFs, write what you want to stand out (natural language), "
    "and download the same PDFs with those parts highlighted or with a colored backdrop."
)

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
user_instruction = st.text_area(
    "What should stand out in the PDFs? (example: 'Highlight company names and dates with a black backdrop')",
    height=120,
)
color_choice = st.selectbox(
    "Backdrop / highlight color",
    ("black", "yellow", "red", "green", "blue"),
)
opacity = st.slider("Backdrop opacity (0 = transparent, 1 = opaque)", 0.1, 0.9, 0.45)
process = st.button("Process PDFs")

# ---------- Helpers ----------
def call_groq_chat(instruction: str, pdf_text: str) -> str:
    """
    Calls the Groq OpenAI-compatible Chat Completion endpoint.
    Returns the assistant content string.
    """
    url = f"{groq_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }

    system_msg = (
        "You are a JSON-only assistant that extracts exact phrases or short strings "
        "that should be highlighted from the provided PDF text. "
        "Return only a JSON list of strings. Example: [\"Google\", \"Techware Hub\", \"Jan 2020\"]"
    )

    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": f"Instruction: {instruction}\n\nPDF_CONTENT_START\n{pdf_text}\nPDF_CONTENT_END\n\nReturn only a JSON array of strings to highlight.",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Follow OpenAI-compatible response shape:
    return data["choices"][0]["message"]["content"]

def parse_model_output(raw: str) -> List[str]:
    """
    Safely parse model output into a Python list of strings.
    Try JSON first, then ast.literal_eval fallback.
    """
    raw = raw.strip()
    # If there is extra text surrounding JSON, try to extract first JSON array substring.
    if not raw:
        return []
    # Try JSON direct
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if isinstance(x, (str, int, float))]
    except Exception:
        pass
    # Try to find a bracketed JSON array inside the raw string
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
    # Final fallback: try ast.literal_eval on whole raw
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
    return mapping.get(color_name.lower(), (0, 0, 0))

def highlight_pdf_with_backdrop(input_path: str, output_path: str, targets: List[str], rgb_fill: tuple, opacity_val: float):
    """
    For each target string, search pages and add a semi-opaque rectangle (annotation) behind found matches.
    We attempt multiple capitalization variants to increase hit rate.
    """
    doc = fitz.open(input_path)
    for page in doc:
        found_rects = []
        for t in targets:
            if not t.strip():
                continue
            # attempt search with several variants
            variants = [t, t.strip(), t.strip().title(), t.strip().upper(), t.strip().lower()]
            rects = []
            for v in variants:
                try:
                    r = page.search_for(v, hit_max=128)
                    if r:
                        rects.extend(r)
                except Exception:
                    # ignore search errors for weird characters
                    pass
            # remove duplicates
            for r in rects:
                # Sometimes search_for returns small rects; we expand slightly to cover the backdrop nicely
                r_inflated = fitz.Rect(r.x0 - 1, r.y0 - 0.5, r.x1 + 1, r.y1 + 0.5)
                found_rects.append(r_inflated)
        # Draw annotations for found_rects
        for rect in found_rects:
            try:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=None, fill=rgb_fill)
                annot.set_opacity(opacity_val)
                annot.update()
            except Exception:
                # Fallback: draw a filled rectangle on the page content layer
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
    elif not user_instruction.strip():
        st.error("Please enter an instruction describing what should stand out.")
    else:
        st.info("Processing PDFs. This may take a few moments.")

        rgb = color_to_rgb_tuple(color_choice)
        for uploaded in uploaded_files:
            # Save input to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(uploaded.read())
                tmp_in.flush()

                # Extract text from pages (limit to reasonable length)
                doc = fitz.open(tmp_in.name)
                full_text = []
                for p in doc:
                    try:
                        full_text.append(p.get_text("text"))
                    except Exception:
                        full_text.append("")
                doc.close()

                text_for_model = "\n".join(full_text)
                # Limit to first n chars to avoid too long prompts
                text_for_model = text_for_model[:13000]

                # Call Groq (LLM) to get exact phrases to highlight
                try:
                    raw = call_groq_chat(user_instruction, text_for_model)
                except Exception as e:
                    st.error(f"LLM call failed for {uploaded.name}: {e}")
                    continue

                targets = parse_model_output(raw)
                if not targets:
                    st.warning(f"No highlight targets returned for {uploaded.name}.")
                    continue

                # Create output temp file
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
