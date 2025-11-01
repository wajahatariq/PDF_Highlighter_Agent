import streamlit as st
import tempfile
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# =====================
# Load secrets & setup
# =====================
groq_api_key = st.secrets["groq"]["api_key"]
groq_model = st.secrets["groq"]["model"]

llm = ChatOpenAI(
    model=groq_model,
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="PDF Highlighter Agent", layout="wide")
st.title("📝 PDF Highlighter Agent (Groq + LangChain)")

st.write("Upload multiple PDFs and enter a prompt (e.g., 'Highlight all headings with a black background').")

# =====================
# Streamlit inputs
# =====================
uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)
prompt_text = st.text_area("Enter your highlighting instruction", height=100)
process_button = st.button("Process PDFs")

# =====================
# Highlighting utility
# =====================
def highlight_pdf(input_pdf_path, output_pdf_path, highlight_targets):
    doc = fitz.open(input_pdf_path)
    for page in doc:
        text_instances = []
        for target in highlight_targets:
            text_instances += page.search_for(target)
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.set_colors(stroke=(0, 0, 0))  # black highlight outline
            highlight.update()
    doc.save(output_pdf_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()

# =====================
# Main processing
# =====================
if process_button and uploaded_files and prompt_text.strip():
    st.info("Processing your PDFs... Please wait ⏳")

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_input:
            tmp_input.write(uploaded_file.read())
            tmp_input.flush()

            # Extract text from PDF
            pdf = fitz.open(tmp_input.name)
            text_content = ""
            for page in pdf:
                text_content += page.get_text("text") + "\n"
            pdf.close()

            # Use Groq LLM to find what to highlight
            template = """
            You are a document analysis agent. The user wants certain text highlighted in a PDF.
            Instruction: {instruction}
            Based on the PDF content below, list exact phrases or headings to highlight.
            Return only a JSON list of strings to highlight, like ["Education", "Experience", "Skills"].
            PDF Content:
            {pdf_text}
            """

            prompt = PromptTemplate(input_variables=["instruction", "pdf_text"], template=template)
            full_prompt = prompt.format(instruction=prompt_text, pdf_text=text_content[:8000])  # limit text

            response = llm.invoke(full_prompt)
            raw_output = response.content.strip()

            # Extract list safely
            try:
                targets = eval(raw_output)
                if not isinstance(targets, list):
                    targets = []
            except Exception:
                targets = []

            if not targets:
                st.warning(f"No highlight targets detected in {uploaded_file.name}.")
                continue

            # Highlight PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_output:
                highlight_pdf(tmp_input.name, tmp_output.name, targets)
                st.success(f"✅ Processed: {uploaded_file.name}")
                with open(tmp_output.name, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {uploaded_file.name}",
                        data=f,
                        file_name=f"highlighted_{uploaded_file.name}",
                        mime="application/pdf"
                    )
else:
    st.caption("👆 Upload files and enter your highlighting instruction above.")
