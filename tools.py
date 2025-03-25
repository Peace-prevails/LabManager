from langchain.document_loaders import PyPDFLoader
import re
def load_pdf_with_better_parsing(file_path):
    """
    Load and process PDF files with better parsing and cleanup.

    Parameters:
        file_path (str): Path to the PDF file

    Returns:
        list: List of Document objects
    """


    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Post-processing to clean the text
        cleaned_documents = []
        for doc in documents:
            text = doc.page_content

            # 1. Remove extra blank lines
            text = re.sub(r'\n\s*\n', '\n\n', text)

            # 2. Remove headers and footers (typically fixed text repeated on every page)
            # Adjust the patterns according to your specific PDF
            text = re.sub(r'Page \d+ of \d+', '', text)
            text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers

            # 3. Remove special and control characters
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)

            # 4. Fix line break issues - merge incorrectly broken sentences
            text = re.sub(r'(?<!\n)(\w)\-\n(\w)', r'\1\2', text)
            text = re.sub(r'(\w)[\.\?\!]\s*\n+(\w)', r'\1. \2', text)

            # 5. Remove duplicate spaces
            text = re.sub(r' +', ' ', text)

            # Final cleanup
            text = text.strip()

            # Create a new document - keep original metadata but update content
            cleaned_doc = doc.copy()
            cleaned_doc.page_content = text

            # Ensure non-empty content
            if text.strip():
                cleaned_documents.append(cleaned_doc)

        print(f"Original document count: {len(documents)}, Cleaned document count: {len(cleaned_documents)}")
        return cleaned_documents

    except Exception as e:
        print(f"PDF processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
