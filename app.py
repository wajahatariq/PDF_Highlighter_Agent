import streamlit as st
import tempfile
import fitz  # PyMuPDF
import requests
import json
import re
import ast
from typing import List

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PDF Company Highlighter", layout="wide")

st.title("📄 PDF Company Highlighter (AI + Python)")
st.write("Upload your CV and this tool will highlight all company names you’ve worked for.")

# Secret keys (from Streamlit Cloud secrets)
groq_key = st.secrets["groq"]["api_key"]
groq_model = st.secrets["groq"].get("model", "llama-3.1-8b-instant")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
highlight_color = st.selectbox("Select highlight color", ["red", "yellow", "green", "blue"])
opacity = st.slider("Backdrop opacity", 0.1, 0.9, 0.45)
process_btn = st.button("Extract & Highlight")

# ---------------- HELPERS ----------------

def call_groq_for_companies(pdf_text: str) -> str:
    """Ask the LLM to return company names in JSON only."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}

    system_prompt = (
        "You are a JSON-only assistant. "
        "From the given CV text, extract and return ONLY a JSON array of company or organization names "
        "where the person has worked. Return no explanations or extra text."
    )
    user_prompt = f"CV_TEXT_START\n{pdf_text}\nCV_TEXT_END"

    payload = {
        "model": groq_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def parse_json_output(raw: str) -> List[str]:
    """Extract list of company names from possibly messy model output."""
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
    """Highlight each target word or phrase with a colored rectangle."""
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

# ---------------- MAIN ----------------
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
    full_text = "\n".join(text_pages)[:15000]  # limit size

    st.subheader("Extracted Text Preview")
    st.text_area("Extracted CV text sent to LLM:", full_text, height=300)

    # Ask LLM for company names
    try:
        st.info("Asking Groq model to identify company names...")
        raw_response = call_groq_for_companies(full_text)
        st.text_area("Raw model output", raw_response, height=200)
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        st.stop()

    companies = parse_json_output(raw_response)
    if not companies:
        st.warning("No company names detected. Check the model output above.")
        st.stop()

    st.write("✅ Companies detected:", companies)

    # Highlight and save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
        try:
            highlight_words_in_pdf(tmp_in.name, tmp_out.name, companies, color_to_rgb(highlight_color), opacity)
        except Exception as e:
            st.error(f"Failed to highlight PDF: {e}")
            st.stop()

        st.success("✅ PDF successfully processed!")
        with open(tmp_out.name, "rb") as f:
            st.download_button(
                label="Download Highlighted PDF",
                data=f,
                file_name=f"highlighted_{uploaded_pdf.name}",
                mime="application/pdf"
            )

