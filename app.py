import streamlit as st
import tempfile
import fitz  # PyMuPDF
import json
import re
from typing import List
from litellm import completion
from PIL import Image
import os
import requests
import tempfile


# ---------- CONFIG ----------
st.set_page_config(page_title="PDF Company Highlighter", layout="wide")

st.title("PDF Company Highlighter (LiteLLM)")
st.write("Upload a CV PDF, extract company names from the Experience section, and highlight them.")

# --- User inputs ---
uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
color_choice = st.selectbox("Backdrop color", ["red", "yellow", "green", "blue", "black"], index=0)
opacity = st.slider("Backdrop opacity (0 = transparent, 1 = opaque)", 0.1, 0.9, 0.45)
process = st.button("Process PDFs")


# ---------- HELPERS ----------
def color_to_rgb_tuple(color_name: str):
    mapping = {
        "black": (0, 0, 0),
        "yellow": (1, 1, 0),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
    }
    return mapping.get(color_name.lower(), (1, 0, 0))


def extract_experience_section(text: str) -> str:
    """Extract only the 'Experience' section from CV text."""
    pattern = r"(?is)(experience[\s\S]*?)(education|extra|skill|objective|$)"
    match = re.search(pattern, text)
    if match:
        section = match.group(1).strip()
        return section
    return text.strip()
    
def call_groq_via_litellm(pdf_text: str, api_key: str) -> str:
    """Call Groq LLM via LiteLLM to extract company names."""
    try:
        response = completion(
            model="groq/llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        '''
                        You are a precise information extraction assistant. 
                        Your goal is to read a person's CV text and extract only the **names of actual companies or organizations** where they have worked.
                        
                        Ignore:
                        - Job titles, roles, and positions
                        - Project names or campaigns
                        - Educational institutions
                        - Section headings or skills
                        - Any non-company words or phrases
                        
                        Return only the company or organization names as a valid JSON array. 
                        Do not include any explanations or extra text — the entire response must be valid JSON, for example:
                        ['a', 'b']
                        '''
                    )
                },
                {"role": "user", "content": pdf_text},
            ],
            api_key=api_key,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Groq LiteLLM error: {e}")
        return "[]"
def ocr_space_extract_text(file_path: str, api_key: str) -> str:
    """Extract text from image or PDF using OCR.Space API."""
    url = "https://api.ocr.space/parse/image"
    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            data={
                "apikey": api_key,
                "language": "eng",
                "isOverlayRequired": False,
            },
        )
    result = response.json()
    try:
        return result["ParsedResults"][0]["ParsedText"]
    except Exception:
        return ""

def smart_parse_companies(raw: str) -> List[str]:
    """Parse JSON or natural text to extract company names."""
    raw = raw.strip()
    if not raw:
        return []

    # Try strict JSON first
    try:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            arr = json.loads(match.group())
            if isinstance(arr, list):
                return [x.strip() for x in arr if isinstance(x, str) and x.strip()]
    except Exception:
        pass

    # Fallback: extract from natural sentences
    lines = re.split(r"[\n,.;]", raw)
    candidates = []
    for line in lines:
        m = re.findall(r"\b([A-Z][A-Za-z0-9&\-\s]{1,40})\b", line)
        for c in m:
            if len(c.split()) <= 4 and not any(
                bad in c.lower()
                for bad in ["experience", "company", "worked", "organization"]
            ):
                candidates.append(c.strip())

    # Deduplicate, preserve order
    seen = set()
    final = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            final.append(c)
    return final


def highlight_pdf_with_backdrop(input_path: str, output_path: str, targets: List[str], rgb_fill: tuple, opacity_val: float):
    doc = fitz.open(input_path)
    for page_num, page in enumerate(doc):
        for t in targets:
            if not t.strip():
                continue
            rects = page.search_for(t, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_IGNORECASE)
            st.write(f"Page {page_num+1}: Found {len(rects)} highlights for '{t}'")
            for r in rects:
                r_inflated = fitz.Rect(r.x0 - 1, r.y0 - 0.5, r.x1 + 1, r.y1 + 0.5)
                annot = page.add_rect_annot(r_inflated)
                annot.set_colors(stroke=None, fill=rgb_fill)
                annot.set_opacity(opacity_val)
                annot.update()
    doc.save(output_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()


# ---------- MAIN ----------
if process:
    if not uploaded_files:
        st.error("Please upload one or more PDFs first.")
    else:
        api_key = st.secrets["groq"]["api_key"]
        rgb = color_to_rgb_tuple(color_choice)

        for uploaded in uploaded_files:
            st.info(f"Processing {uploaded.name}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
                tmp_in.write(uploaded.read())
                tmp_in.flush()

                # --- Step 1: Try normal text extraction ---
                doc = fitz.open(tmp_in.name)
                all_text = []
                for p in doc:
                    try:
                        page_text = p.get_text("text")
                        if page_text.strip():
                            all_text.append(page_text)
                    except Exception:
                        pass
                doc.close()

                # --- Step 2: Fallback to OCR.Space if no text found ---
                if not "".join(all_text).strip():
                    st.warning(f"No text detected in {uploaded.name}. Running OCR via OCR.Space...")
                    ocr_api_key = st.secrets["ocr"]["api_key"]
                    extracted_text = ocr_space_extract_text(tmp_in.name, ocr_api_key)
                    if extracted_text.strip():
                        all_text = [extracted_text]
                    else:
                        st.error("OCR failed to extract text.")
                        continue

                # Only send Experience section to model
                full_text = "\n".join(all_text)
                text_for_model = extract_experience_section(full_text)[:5000]

                st.text_area("Extracted Experience Section", text_for_model, height=250)

                raw_response = call_groq_via_litellm(text_for_model, api_key)
                st.text_area("Raw LLM output", raw_response, height=150)

                companies = smart_parse_companies(raw_response)
                if not companies:
                    st.warning(f"No company names detected for {uploaded.name}.")
                    continue

                st.write(f"**Detected companies:** {', '.join(companies)}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
                    highlight_pdf_with_backdrop(tmp_in.name, tmp_out.name, companies, rgb, opacity)
                    with open(tmp_out.name, "rb") as f:
                        st.download_button(
                            label=f"Download highlighted {uploaded.name}",
                            data=f,
                            file_name=f"highlighted_{uploaded.name}",
                            mime="application/pdf",
                        )

        st.success("✅ Done processing all PDFs.")
