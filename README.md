# ⚖️ Legal Document Summarizer

An AI-powered Streamlit application that automatically summarizes legal documents using the FLAN-T5 language model.

## Features

- **PDF Processing**: Extract and clean text from legal PDF documents
- **AI Summarization**: Uses Google's FLAN-T5-large model for intelligent summarization
- **Legal Focus**: Specifically designed to extract:
  - Key legal facts
  - Judgment reasoning
  - Statutory references
- **Interactive Interface**: User-friendly Streamlit web interface
- **Customizable Settings**: Adjust chunk size and summary detail level
- **Progress Tracking**: Real-time progress bars during processing
- **Export Functionality**: Download summaries as text files

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nandan-2004/Legal_summarizer.git
cd Legal_summarizer
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run legal.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload a legal PDF document using the file uploader

4. Adjust settings in the sidebar if needed:
   - Max tokens per chunk (300-800)
   - Summary detail level (Concise/Detailed/Comprehensive)

5. View the generated summary and download if needed

## How It Works

1. **Text Extraction**: Uses `pdfplumber` to extract text from PDF documents
2. **Text Cleaning**: Removes formatting artifacts and normalizes text
3. **Chunking**: Splits text into manageable chunks based on sentence boundaries
4. **AI Processing**: Each chunk is processed by FLAN-T5 with specialized legal prompts
5. **Post-processing**: Combines and filters summaries for final output

## Requirements

- Python 3.11+
- See `requirements.txt` for package dependencies

## Model Information

This application uses Google's FLAN-T5-base model (~250MB) for a balance of quality and deployment performance. It will be automatically downloaded on first use.

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to open issues or submit pull requests for improvements.
