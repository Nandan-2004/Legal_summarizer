from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
import streamlit as st
import tempfile
import re
import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load pretrained FLAN-T5 model and tokenizer
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def extract_text_from_pdf(pdf_path):
    full_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'(\w)-\s*\n(\w)', r'\1\2', text)
                full_text += text + "\n"
    return clean_legal_text(full_text)

def clean_legal_text(text):
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page \d+ of \d+\s*\n', '\n', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    return text.strip()

def sentence_chunking(text, max_tokens=500):
    sents = sent_tokenize(text)
    chunks, current, tokens = [], "", 0
    for sent in sents:
        sent_len = len(tokenizer.tokenize(sent))
        if tokens + sent_len > max_tokens:
            chunks.append(current.strip())
            current, tokens = sent, sent_len
        else:
            current += " " + sent
            tokens += sent_len
    if current:
        chunks.append(current.strip())
    return chunks

def get_better_prompt(chunk):
    return (
        "You are a legal assistant. Read the following court document section and extract only the relevant legal facts, judgment reasoning, and statutory references in clear legal English:\n\n"
        + chunk
    )

def summarize_chunks_enhanced(chunks):
    summaries = []
    progress_bar = st.progress(0)
    for idx, chunk in enumerate(chunks):
        prompt = get_better_prompt(chunk)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_length=250,
            min_length=50,
            num_beams=8,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            top_p=0.9,
            temperature=0.7,
            do_sample=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        summaries.append(summary)
        progress_bar.progress((idx + 1) / len(chunks))
    return summaries

def post_process_summary(summaries):
    structured_summary = [s for s in summaries if len(s.split()) > 10 and "summary" not in s.lower()]
    return "\n\n".join(structured_summary)

def main():
    # Set page config
    st.set_page_config(
        page_title="Automated Legal Document Summarizer",
        page_icon="⚖️",
        layout="centered"
    )

    # Sidebar with options
    with st.sidebar:
        st.title("⚙️ Settings")
        max_tokens = st.slider(
            "Max tokens per chunk",
            min_value=300,
            max_value=800,
            value=500,
            help="Adjust based on document complexity"
        )
        summary_length = st.selectbox(
            "Summary detail level",
            options=["Concise", "Detailed", "Comprehensive"],
            index=1
        )
        st.markdown("---")
        st.markdown("**About this tool:**")
        st.markdown("This AI-powered summarizer specializes in legal documents, extracting key facts, judgments, and legal reasoning.")

    # Main content
    st.title("⚖️ Automated Legal Document Summarizer")
    st.markdown("Upload a legal document to receive an AI-powered summary with:")
    st.markdown("- Key legal facts")
    st.markdown("- Judgment reasoning")
    st.markdown("- Statutory references")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a legal judgment or document"
    )

    if uploaded_file is not None:
        # Document processing
        with st.expander("Document Processing", expanded=True):
            with st.spinner("Extracting text from PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                full_text = extract_text_from_pdf(tmp_path)
                st.success(f"Text extracted ({len(full_text)} characters)")
                
                if st.checkbox("Show sample of extracted text"):
                    st.text_area(
                        "Extracted Text Sample", 
                        value=full_text[:2000], 
                        height=200,
                        label_visibility="collapsed"
                    )

        # Summary generation
        with st.spinner("Generating legal summary..."):
            chunks = sentence_chunking(full_text, max_tokens=max_tokens)
            summaries = summarize_chunks_enhanced(chunks)
            final_summary = post_process_summary(summaries)
        
        st.success("Summary generated successfully!")
        st.markdown("---")

        # Results display
        st.subheader("📜 Legal Summary")
        st.text_area(
            "Summary Content",
            value=final_summary,
            height=400,
            label_visibility="collapsed"
        )

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Document Chunks Processed", len(chunks))
        with col2:
            st.metric("Summary Length", f"{len(final_summary.split())} words")

        # Download button
        st.download_button(
            "💾 Download Summary",
            data=final_summary,
            file_name="legal_summary.txt",
            mime="text/plain",
            help="Save the summary to your device"
        )

        # Clean up
        os.unlink(tmp_path)

if __name__ == "__main__":
    main()