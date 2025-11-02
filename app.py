import streamlit as st
import tempfile
import fitz  # PyMuPDF
import json
import re
import ast
from typing import List
from litellm import completion

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Company Highlighter (LiteLLM)", layout="wide")

st.title("📄 PDF Company Highlighter (LiteLLM + Groq)")
st.write("Upload a CV and highlight all company names you’ve worked for.")

# Load keys from Streamlit secrets
groq_key = st.secrets["groq"]["api_key"]
groq_model = st.secrets["groq"].get("model", "groq/llama-3.1-8b-instant")

# Upload + UI
uploaded_pdf = st.file_uploader("Upload a CV (PDF)", type=["pdf"])
highlight_color = st.selectbox("Highlight color", ["red", "yellow", "green", "blue"])
opacity = st.slider("Backdrop opacity", 0.1, 0.9, 0.45)
process_btn = st.button("Extract & Highlight")

# ---------------- HELPERS ----------------

def call_groq_via_litellm(pdf_text: str) -> str:
    """Use LiteLLM to call Groq and return model output."""
    response = completion(
        model=groq_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a JSON-only assistant. "
                    "From the CV text, extract and return ONLY a JSON array of company or organization names "
                    "where the person has worked. No explanations or other text."
                ),
            },
            {"role": "user", "content": pdf_text},
        ],
        api_key=groq_key,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]

def parse_json_output(raw: str) -> List[str]:
    """Extract a clean list of company names even if model output isn't perfectly formatted."""
    raw = raw.strip()
    if not raw:
        return []
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []

def color_to_rgb(color_name: str):
    mapping = {
        "red": (1, 0, 0),
        "yellow": (1, 1, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
    }
    return mapping.get(color_name.lower(), (1, 0, 0))

def highlight_words_in_pdf(input_path: str, output_path: str, targets: List[str], rgb_fill: tuple, opacity_val: float):
    """Highlight each target word or phrase with a colored backdrop."""
    doc = fitz.open(input_path)
    for page in doc:
        for target in targets:
            target = target.strip()
            if not target:
                continue
            rects = page.search_for(target, hit_max=256)
            for rect in rects:
                backdrop = fitz.Rect(rect.x0 - 1, rect.y0 - 0.5, rect.x1 + 1, rect.y1 + 0.5)
                annot = page.add_rect_annot(backdrop)
                annot.set_colors(stroke=None, fill=rgb_fill)
                annot.set_opacity(opacity_val)
                annot.update()
    doc.save(output_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

# ---------------- MAIN APP LOGIC ----------------
if process_btn:
    if not uploaded_pdf:
        st.error("Please upload a PDF first.")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
        tmp_in.write(uploaded_pdf.read())
        tmp_in.flush()

    # Extract text
    with fitz.open(tmp_in.name) as doc:
        text_pages = [p.get_text("text") for p in doc]
    full_text = "\n".join(text_pages)[:15000]  # limit token size

    st.subheader("Extracted Text Preview")
    st.text_area("Text sent to LLM", full_text, height=300)

    # LLM extraction via LiteLLM
    try:
        st.info("Analyzing PDF text with Groq model via LiteLLM...")
        raw_response = call_groq_via_litellm(full_text)
        st.text_area("Raw LLM Output", raw_response, height=200)
    except Exception as e:
        st.error(f"LiteLLM / Groq call failed: {e}")
        st.stop()

    companies = parse_json_output(raw_response)
    if not companies:
        st.warning("No company names detected. Check model output.")
        st.stop()

    st.write("✅ Companies found:", companies)

    # Highlight in PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
        try:
            highlight_words_in_pdf(tmp_in.name, tmp_out.name, companies, color_to_rgb(highlight_color), opacity)
        except Exception as e:
            st.error(f"Failed to edit PDF: {e}")
            st.stop()

        st.success("✅ PDF processed successfully!")
        with open(tmp_out.name, "rb") as f:
            st.download_button(
                label="Download Highlighted PDF",
                data=f,
                file_name=f"highlighted_{uploaded_pdf.name}",
                mime="application/pdf",
            )
