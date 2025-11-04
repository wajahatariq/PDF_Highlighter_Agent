import streamlit as st
import tempfile
import fitz  # PyMuPDF
import json
import re
from typing import List
from litellm import completion

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


def call_groq_via_litellm(pdf_text: str, api_key: str) -> List[str]:
    """Call Groq LLM via LiteLLM to extract company names as a JSON array."""
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
                        
                        Return only the company or organization names as a valid JSON array. Don't give nothing else 
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
        content = response["choices"][0]["message"]["content"]

        # Parse LLM JSON response directly, no regex fallback
        companies = json.loads(content)
        if isinstance(companies, list):
            return [c.strip() for c in companies if isinstance(c, str) and c.strip()]
        return []
    except Exception as e:
        st.error(f"Groq LiteLLM error or JSON parsing error: {e}")
        return []


def highlight_pdf_with_backdrop(input_path: str, output_path: str, targets: List[str], rgb_fill: tuple, opacity_val: float):
    doc = fitz.open(input_path)
    for page in doc:
        for t in targets:
            if not t.strip():
                continue
            rects = page.search_for(t)
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

                doc = fitz.open(tmp_in.name)
                all_text = []
                for p in doc:
                    try:
                        all_text.append(p.get_text("text"))
                    except Exception:
                        pass
                doc.close()

                full_text = "\n".join(all_text)
                text_for_model = extract_experience_section(full_text)[:5000]

                st.text_area("Extracted Experience Section", text_for_model, height=250)

                companies = call_groq_via_litellm(text_for_model, api_key)
                st.text_area("Raw LLM output", str(companies), height=150)

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
