# -*- coding: utf-8 -*-
"""labAgent.py - Laboratory Knowledge Retrieval System"""

# Import required libraries
import os
import time
import requests
import gc
import uuid
import shutil
import tempfile
from tqdm.auto import tqdm
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional, Union
from tools import *
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, 
    DirectoryLoader, 
    TextLoader, 
    CSVLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from huggingface_hub import hf_hub_download

# Set base paths
base_path = "./"  # Change if needed
knowledge_pool_path = os.path.join(base_path, "Knowledge_Pool")  # Renamed from lab_documents
db_path = os.path.join(base_path, "chroma_db")
models_dir = os.path.join(base_path, "models")
model_path = os.path.join(models_dir, "llama-2-7b-chat.Q4_K_M.gguf")  # Default model
temp_dir = os.path.join(base_path, "temp")

# Create necessary directories
os.makedirs(knowledge_pool_path, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Import tiktoken for accurate token counting
try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base as approximation for LLaMA tokenizer
    print("Tiktoken initialized for accurate token counting")
except ImportError:
    print("Tiktoken not available, using approximate token counting")
    tokenizer = None

# Function to count tokens accurately
def count_tokens(text):
    """Count the number of tokens in text"""
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Approximate token count as words * 1.3
        return int(len(text.split()) * 1.3)

# Step 1: Download model (from Hugging Face)
def download_model():
    """Download model function"""
    global model_path

    print("Starting model download process...")

    # Provide optional model list - Llama 2 only
    models = [
        {
            "name": "Llama 2 7B Chat (approx. 4GB)",
            "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
            "filename": "llama-2-7b-chat.Q4_K_M.gguf",
            "local_path": os.path.join(models_dir, "llama-2-7b-chat.Q4_K_M.gguf")
        }
    ]

    print("Select model to download:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model['name']}")

    try:
        choice = 1  # Only one option, use default
        print("Using default option 1 (Llama 2)")
    except:
        print("Invalid input, using default option 1 (Llama 2)")
        choice = 1

    selected_model = models[choice-1]
    print(f"Will download: {selected_model['name']}")

    # Check if model already exists
    if os.path.exists(selected_model['local_path']):
        print(f"Model file already exists: {selected_model['local_path']}")
        model_path = selected_model['local_path']
        return True

    # Try to download using huggingface_hub
    try:
        print(f"Downloading model from Hugging Face, this may take several minutes...")
        print(f"Downloading: {selected_model['repo_id']}/{selected_model['filename']}")

        # Download from Hugging Face
        hf_hub_download(
            repo_id=selected_model['repo_id'],
            filename=selected_model['filename'],
            local_dir=models_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=None  # Set token if needed
        )

        # Update global model path
        model_path = selected_model['local_path']
        print(f"Model download complete: {model_path}")
        return True

    except Exception as e:
        print(f"Failed to download using huggingface_hub: {str(e)}")
        print("Trying direct HTTP download...")

        # Direct download links
        direct_links = {
            "llama-2-7b-chat.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        }

        url = direct_links.get(selected_model['filename'])
        if not url:
            print(f"Could not find direct download link for {selected_model['filename']}")
            return False

        try:
            response = requests.get(url, stream=True)
            file_size = int(response.headers.get('content-length', 0))

            # Configure progress bar
            progress = tqdm(total=file_size, unit='iB', unit_scale=True)

            # Download file
            with open(selected_model['local_path'], 'wb') as file:
                for data in response.iter_content(chunk_size=1024*1024):
                    progress.update(len(data))
                    file.write(data)
            progress.close()

            # Update global model path
            model_path = selected_model['local_path']
            print(f"Model download complete: {model_path}")
            return True

        except Exception as e:
            print(f"Direct download also failed: {str(e)}")
            print("Please download the model manually and place it in the correct path")
            return False

# Step 2: Load and process documents
class ImprovedPyPDFLoader:
    """PDF loader with improved text cleaning"""
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        return load_pdf_with_better_parsing(self.file_path)
def load_documents(directory=None):
    """Load documents from a directory"""
    if directory is None:
        directory = knowledge_pool_path

    print(f"Loading documents from {directory}...")
    
    # Define loaders for different file types
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf", loader_cls=ImprovedPyPDFLoader),
        DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(directory, glob="**/*.csv", loader_cls=CSVLoader),
        DirectoryLoader(directory, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader),
        DirectoryLoader(directory, glob="**/*.xls", loader_cls=UnstructuredExcelLoader),
        DirectoryLoader(directory, glob="**/*.docx", loader_cls=Docx2txtLoader),
        DirectoryLoader(directory, glob="**/*.doc", loader_cls=Docx2txtLoader),
        DirectoryLoader(directory, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader),
        DirectoryLoader(directory, glob="**/*.ppt", loader_cls=UnstructuredPowerPointLoader)
    ]

    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            if docs:
                print(f"  Loaded {len(docs)} documents with {loader.__class__.__name__}")
                documents.extend(docs)
        except Exception as e:
            print(f"  Error loading with {loader.__class__.__name__}: {str(e)}")

    print(f"Successfully loaded {len(documents)} documents in total")
    return documents

# Step 3: Split documents
def split_documents(documents):
    """Split documents into smaller text chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,            # Smaller chunk size for more precise retrieval
        chunk_overlap=250,         # Larger overlap to maintain context between chunks
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", " ", "，", ",", "；", ";", ""]
    )

    splits = text_splitter.split_documents(documents)
    print(f"Documents split into {len(splits)} text chunks")
    return splits

# Step 4: Create or load vector store
def setup_vectorstore(splits=None):
    """Initialize or load vector database"""
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    # Check if vector store already exists
    if os.path.exists(db_path) and os.listdir(db_path):
        print("Loading existing vector database...")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        if splits is None:
            print("No text chunks, cannot create vector store")
            return None

        print("Creating new vector database...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_path
        )
        vectorstore.persist()

    return vectorstore

# Step 5: Set up LLM with optimized parameters
def setup_llm():
    """Load language model with optimized parameters"""
    # Check model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        if not download_model():
            print("Unable to download model, please download manually")
            return None

    print("Loading language model with optimized settings...")
    try:
        # Enhanced model parameters for better speed/performance balance
        model_params = {
            "model_path": model_path,
            "temperature": 0.1,           # Lower temperature for more deterministic responses
            "max_tokens": 1024,           # Increased output length for more comprehensive responses
            "n_ctx": 4096,                # Context window size
            "n_batch": 512,               # Increased batch size for faster processing
            "verbose": False,
            "f16_kv": True,               # Use half-precision for KV cache
            "use_mlock": True,            # Lock memory to prevent swapping
            "use_mmap": True,             # Use memory mapping for faster loading
            "top_p": 0.8,                 # More focused sampling
            "repeat_penalty": 1.1,
            "last_n_tokens_size": 64,     # Consider fewer tokens for repetition penalty
            "seed": -1,                   # Deterministic generation is faster
            "top_k": 40,                  # Limit vocabulary sampling
            "stop": ["</s>", "Human:", "User:"]
        }
        
        # Use LlamaCpp with optimized params
        llm = LlamaCpp(**model_params)
        
        print("Model loaded successfully")
        print("Model type: LLaMA 2")
        
        gc.collect()  # Force garbage collection
        return llm
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Rule-based query classifier to avoid using LLM tokens
def classify_query(query):
    """
    Classify query type using rule-based patterns instead of LLM
    
    Parameters:
        query (str): The user query to classify
        
    Returns:
        str: Query type (FACTUAL, CREATIVE, SUMMARY, ANALYSIS, or OTHER)
    """
    query = query.lower()
    
    # Factual patterns - typical question formats
    factual_patterns = [
        r'\bwhat\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b', r'\bwho\b', r'\bhow\b',
        r'\bcan you tell\b', r'\bexplain\b', r'\bdescribe\b', r'\bdefine\b', 
        r'\?$'  # Ends with question mark
    ]
    
    # Creative patterns - content generation requests
    creative_patterns = [
        r'\bwrite\b', r'\bcreate\b', r'\bcompose\b', r'\bdraft\b', r'\bgenerate\b',
        r'\bessay\b', r'\breport\b', r'\bstory\b', r'\bemail\b', r'\bletter\b',
        r'\bproduce\b', r'\bdesign\b', r'\bcreative\b', r'\bimagine\b'
    ]
    
    # Summary patterns - synthesizing information requests
    summary_patterns = [
        r'\bsummarize\b', r'\boverview\b', r'\bsummary\b', r'\bhighlight\b', 
        r'\bkey points\b', r'\bmain points\b', r'\btakeaways\b', r'\bsummarise\b',
        r'\breview\b', r'\bcondense\b', r'\bdigest\b', r'\bsynopsize\b', r'\bsynthesize\b'
    ]
    
    # Analysis patterns - evaluation and comparison requests
    analysis_patterns = [
        r'\banalyze\b', r'\banalysis\b', r'\bcompare\b', r'\bcontrast\b', r'\bevaluate\b',
        r'\bexamine\b', r'\bcritique\b', r'\bassess\b', r'\badvantages\b', r'\bdisadvantages\b',
        r'\bpros and cons\b', r'\bstrengths\b', r'\bweaknesses\b', r'\bcritical\b',
        r'\binterpret\b', r'\bderive insights\b', r'\blimitations\b', r'\bimplications\b',
        r'\bresearch directions\b', r'\bfuture work\b', r'\bgaps\b', r'\bapproaches\b'
    ]
    
    import re
    
    # Check each pattern category
    for pattern in factual_patterns:
        if re.search(pattern, query):
            return "FACTUAL"
            
    for pattern in creative_patterns:
        if re.search(pattern, query):
            return "CREATIVE"
            
    for pattern in summary_patterns:
        if re.search(pattern, query):
            return "SUMMARY"
            
    for pattern in analysis_patterns:
        if re.search(pattern, query):
            return "ANALYSIS"
    
    # Default to FACTUAL for unclassified queries
    return "FACTUAL"

# Streamlined prompt templates that minimize token usage
def get_optimized_prompt_templates():
    """Create optimized prompt templates that minimize token usage"""
    
    # Concise factual template
    factual_template = """
Do not make up information.Be specific. If you mention resources or items in a list, be sure to actually list them. Answer based on this context:
{context}

Question: {question}

Answer:"""

    # Concise creative template
    creative_template = """
Use fluent and friendly language. Create content based on this context:
{context}

Request: {question}

Creative response:"""

    # Concise summary template
    summary_template = """
Be logical and clear. Summarize based on this context:
{context}

Request: {question}

Summary:"""

    # Concise analysis template
    analysis_template = """
Be precise and specific. Analyze based on this context:
{context}

Request: {question}

Analysis:"""

    # Create template map with optimized prompts
    return {
        "FACTUAL": PromptTemplate(template=factual_template, input_variables=["context", "question"]),
        "CREATIVE": PromptTemplate(template=creative_template, input_variables=["context", "question"]),
        "SUMMARY": PromptTemplate(template=summary_template, input_variables=["context", "question"]),
        "ANALYSIS": PromptTemplate(template=analysis_template, input_variables=["context", "question"]),
        "OTHER": PromptTemplate(template=factual_template, input_variables=["context", "question"])
    }

# Enhanced document combiner with token management
def combine_documents_with_token_management(docs, query, max_context_size, prompt_template, similarity_threshold=0.6):
    """
    Combine documents with careful token tracking to maximize document inclusion
    
    Parameters:
        docs (list): List of retrieved documents
        query (str): Original user query
        max_context_size (int): Maximum context window size in tokens
        prompt_template: The prompt template to be used
        similarity_threshold (float): Threshold for document relevance
    
    Returns:
        tuple: (context_text, used_docs)
    """
    # Calculate tokens needed for the prompt template
    # Create a sample filled prompt to measure its size
    sample_template = prompt_template.format(context="CONTEXT_PLACEHOLDER", question=query)
    template_tokens = count_tokens(sample_template) - count_tokens("CONTEXT_PLACEHOLDER")
    
    # Reserve tokens for the prompt template and reserve a buffer of 50 tokens for safety
    available_tokens = max_context_size - template_tokens - 50
    
    print(f"\nToken budget analysis:")
    print(f"- Maximum context size: {max_context_size} tokens")
    print(f"- Template uses approximately: {template_tokens} tokens")
    print(f"- Available for documents: {available_tokens} tokens")

    try:
        # Get vector store for similarity calculation
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Use vector store for similarity scoring
        similarity_docs = vectorstore.similarity_search_with_score(query, k=len(docs))
        
        # Map document content to similarity score and rank them
        doc_info = []
        for doc in docs:
            # Find similarity score for this document
            score = 0.0
            for similarity_doc, sim_score in similarity_docs:
                if similarity_doc.page_content == doc.page_content:
                    # Convert distance to similarity (assuming smaller distance = more similar)
                    score = 1.0 / (1.0 + sim_score)
                    break
            
            # Count tokens in this document
            token_count = count_tokens(doc.page_content)
            
            # Add to document info if above threshold
            if score >= similarity_threshold:
                doc_info.append({
                    "doc": doc,
                    "score": score,
                    "tokens": token_count
                })
    except:
        # Fallback if vector store is not accessible
        print("Warning: Vector store not accessible for similarity calculation.")
        doc_info = []
        for doc in docs:
            token_count = count_tokens(doc.page_content)
            doc_info.append({
                "doc": doc,
                "score": 1.0,  # Default score
                "tokens": token_count
            })
    
    # Sort by relevance (highest score first)
    doc_info.sort(key=lambda x: x["score"], reverse=True)
    
    # Print document info for transparency
    print("\nAnalyzed documents:")
    for i, info in enumerate(doc_info):
        print(f"{i+1}. Source: {info['doc'].metadata.get('source', 'Unknown')} | "
              f"Score: {info['score']:.4f} | Tokens: {info['tokens']}")
    
    # No relevant documents found
    if not doc_info:
        print("Warning: No sufficiently relevant documents found")
        return "", []
    
    # Combine documents with token management
    context_parts = []
    used_docs = []
    used_tokens = 0
    
    for info in doc_info:
        doc = info["doc"]
        doc_tokens = info["tokens"]
        
        # Check if adding this document would exceed our token budget
        if used_tokens + doc_tokens > available_tokens:
            # If it's the first document and too large, truncate it
            if not used_docs:
                # Calculate how many tokens we can use
                usable_tokens = available_tokens
                
                # Get token-based slice of the content (approximate)
                if tokenizer:
                    tokens = tokenizer.encode(doc.page_content)
                    if len(tokens) > usable_tokens:
                        decoded_content = tokenizer.decode(tokens[:usable_tokens])
                        truncated_content = decoded_content + "... [truncated]"
                    else:
                        truncated_content = doc.page_content
                else:
                    # Approximate truncation based on character count
                    char_per_token = len(doc.page_content) / doc_tokens
                    safe_chars = int(usable_tokens * char_per_token)
                    truncated_content = doc.page_content[:safe_chars] + "... [truncated]"
                
                context_parts.append(truncated_content)
                used_docs.append(doc)
                print(f"Added truncated document: {doc.metadata.get('source', 'Unknown')}")
            
            # Stop adding more documents as we've reached the token limit
            break
        
        # Add this document to our context
        context_parts.append(doc.page_content)
        used_tokens += doc_tokens
        used_docs.append(doc)
        print(f"Added document: {doc.metadata.get('source', 'Unknown')} ({doc_tokens} tokens)")
    
    # Join all document parts with clear separators
    context_text = "\n\n---\n\n".join(context_parts)
    
    print(f"\nFinal context: {used_tokens} tokens from {len(used_docs)} documents")
    
    return context_text, used_docs

# Enhanced QA Chain with optimized token management
class EnhancedRetrievalQA:
    def __init__(self, llm, retriever, prompt_templates, max_context_size=3800):
        self.llm = llm
        self.retriever = retriever
        self.prompt_templates = prompt_templates
        self.max_context_size = max_context_size
    
    def invoke(self, query_dict):
        # Get query text
        query = query_dict["query"]
        
        # Use rule-based classification
        query_type = classify_query(query)
        print(f"Query classified as: {query_type}")
        
        # Select appropriate prompt template
        prompt_template = self.prompt_templates.get(query_type, self.prompt_templates["FACTUAL"])
        
        # Retrieve documents
        docs = self.retriever.invoke(query)
        
        # Process context with token-aware management
        context_text, used_docs = combine_documents_with_token_management(
            docs=docs, 
            query=query,
            max_context_size=self.max_context_size,
            prompt_template=prompt_template
        )
        
        # If no relevant context found
        if not context_text.strip():
            return {
                "result": "I couldn't find relevant information to address your request. Please try rephrasing your question.",
                "source_documents": []
            }
        
        # Create prompt with selected template
        filled_prompt = prompt_template.format(context=context_text, question=query)
        
        # Generate response
        result = self.llm.invoke(filled_prompt)
        result = result.strip() if isinstance(result, str) else result
        
        # Return result with sources
        return {
            "result": result,
            "source_documents": used_docs
        }

# Create QA chain
def create_qa_chain(llm, vectorstore):
    """Create an enhanced QA retrieval chain"""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 15,           # Number of documents to retrieve
            "fetch_k": 25,     # Number to fetch for diversity consideration
            "lambda_mult": 0.7 # Balance between relevance and diversity
        }
    )
    
    # Get optimized prompt templates
    prompt_templates = get_optimized_prompt_templates()
    
    # Max context size for LLaMA-2 7B
    max_context_size = 4000
    
    # Create enhanced QA chain
    qa_chain = EnhancedRetrievalQA(
        llm=llm,
        retriever=retriever,
        prompt_templates=prompt_templates,
        max_context_size=max_context_size
    )
    
    return qa_chain

# Step 7: Main function - Initialize system
def initialize_system():
    """Initialize the entire QA system"""
    # Download model if needed
    if not os.path.exists(model_path):
        success = download_model()
        if not success:
            print("Cannot initialize system, model download failed")
            return None

    # Load documents
    documents = load_documents()
    if not documents:
        print("Warning: No documents found, will continue but cannot answer specific questions")

    # Split documents
    splits = split_documents(documents) if documents else None

    # Set up vector store
    vectorstore = setup_vectorstore(splits)
    if vectorstore is None:
        print("Vector store initialization failed")
        return None

    # Set up LLM
    gc.collect()  # Garbage collection before loading model
    llm = setup_llm()
    if llm is None:
        print("Language model loading failed")
        return None

    # Create QA chain
    qa_chain = create_qa_chain(llm, vectorstore)
    print("QA system initialization successful")
    return qa_chain

# Step 8: Basic query function
def query_system(qa_chain, query):
    """Process user query"""
    if qa_chain is None:
        print("System not properly initialized, cannot process query")
        return

    print(f"Query: {query}")
    print("Processing...")

    start_time = time.time()

    try:
        # Use invoke method
        result = qa_chain.invoke({"query": query})

        print("\nAnswer:")
        print(result["result"])

        print("Reference sources:")
        unique_sources = set()
        for i, doc in enumerate(result["source_documents"][:3], 1):
            source = doc.metadata.get("source", "Unknown source")
            # Ensure sources aren't duplicated
            if source not in unique_sources:
                unique_sources.add(source)
                print(f"{len(unique_sources)}. {source}")

        elapsed_time = time.time() - start_time
        print(f"\nProcessing time: {elapsed_time:.2f} seconds")
        
        return result

    except Exception as e:
        print(f"Query processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Garbage collection after query
        gc.collect()

# =====================================================================
# New Feature: Analyze Uploaded File without Adding to Knowledge Base
# =====================================================================

def analyze_uploaded_file(file_path, query, qa_chain):
    """
    Analyze an uploaded file without adding it to the knowledge base
    
    Parameters:
        file_path (str): Path to the uploaded file
        query (str): User's query
        qa_chain: The QA chain to use for generating responses
        
    Returns:
        dict: Analysis results in format compatible with app.py and app.js
    """
    print(f"\n=== Analyzing uploaded file: {os.path.basename(file_path)} ===")
    start_time = time.time()
    
    # Step 1: Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, os.path.basename(file_path))
        
        # Copy file to temp directory
        shutil.copy2(file_path, temp_file)
        
        # Step 2: Load and process the file
        try:
            # Select appropriate loader based on file extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.pdf':
                documents = load_pdf_with_better_parsing(temp_file)
            elif extension == '.txt':
                loader = TextLoader(temp_file)
                documents = loader.load()
            elif extension == '.csv':
                loader = CSVLoader(temp_file)
                documents = loader.load()
            elif extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(temp_file)
                documents = loader.load()
            elif extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(temp_file)
                documents = loader.load()
            elif extension in ['.pptx', '.ppt']:
                loader = UnstructuredPowerPointLoader(temp_file)
                documents = loader.load()
            else:
                return {
                    "error": True,
                    "message": f"Unsupported file format: {extension}"
                }
            
            # If no documents were loaded
            if not documents or len(documents) == 0:
                return {
                    "error": True,
                    "message": "Could not extract any content from the file"
                }
            
            # Print the content of the documents
            print("\n=== File Content ===")
            for i, doc in enumerate(documents):
                print(f"\n--- Document Chunk {i+1} ---")
                print(f"Metadata: {doc.metadata}")
                print("\nContent:")
                print(doc.page_content)
                print("-" * 50)
            
            # Step 3: Generate a summary using the LLM
            print("\n=== Generating Document Summary ===")
            
            # Create summarization prompt
            summarization_prompt = """
You are a research assistant tasked with creating a concise summary of a document.
Focus on the key points, main themes, and important details.
Organize your summary in a clear, structured manner.

Content to summarize:
{text}

Summary:
"""
            
            # Combine all document chunks for summarization
            combined_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Truncate if too long (to fit in context window)
            max_text_length = 12000  # Adjust based on your model's limits
            if len(combined_text) > max_text_length:
                combined_text = combined_text[:max_text_length] + "... [content truncated due to length]"
            
            # Fill the prompt with the text
            filled_prompt = summarization_prompt.format(text=combined_text)
            
            # Generate summary with the LLM
            document_summary = qa_chain.llm.invoke(filled_prompt).strip()
            
            # Print the summary
            print("\n=== Document Summary ===")
            print(document_summary)
            print("-" * 50)
            
            # Step 4: Find related documents in the knowledge base
            print("\n=== Finding Related Content in Knowledge Base ===")
            
            # Get embedding model
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large"
            )
            
            # Create a combined query for knowledge base search
            kb_search_query = f"{query} {document_summary[:500]}"  # Use query + summary for better matching
            
            # Search knowledge base - load main vector store if it exists
            try:
                if os.path.exists(db_path) and os.listdir(db_path):
                    main_vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
                    similar_docs = main_vectorstore.similarity_search(
                        kb_search_query,
                        k=5  # Get top 5 similar documents
                    )
                    print(f"Found {len(similar_docs)} similar documents in knowledge base")
                else:
                    similar_docs = []
                    print("No knowledge base found or it's empty")
            except Exception as e:
                print(f"Error searching knowledge base: {str(e)}")
                similar_docs = []
            
            # Step 5: Generate research directions based on knowledge base findings
            research_directions = ""
            if similar_docs:
                print("\n=== Generating Research Directions ===")
                
                research_prompt = """
You are a research assistant helping identify research directions or connections.
Based on the user's query and the document they've uploaded (summarized below),
I've found some potentially related documents in our knowledge base.

Document Summary:
{summary}

User Query:
{query}

Related documents from knowledge base:
{related_docs}

Please generate 3 different future research directions that has connections between the uploaded document and our knowledge base.
Focus on potential applications, collaborations, or extensions of the research.
Format as a numbered list.

Research Directions:
"""
                
                # Format related documents for the prompt
                related_docs_text = ""
                for i, doc in enumerate(similar_docs, 1):
                    source = doc.metadata.get("source", "Unknown source")
                    related_docs_text += f"Document {i} ({source}):\n{doc.page_content[:500]}...\n\n"
                
                # Fill the research directions prompt
                filled_research_prompt = research_prompt.format(
                    summary=document_summary,
                    query=query,
                    related_docs=related_docs_text
                )
                
                # Generate research directions
                research_directions = qa_chain.llm.invoke(filled_research_prompt).strip()
                
                print("\n=== Research Directions ===")
                print(research_directions)
                print("-" * 50)
            
            # Step 6: Process the user query using the generated summary
            print(f"\n=== Processing Query: {query} ===")
            
            # Create query processing prompt that uses the generated summary
            query_prompt = """
You are a research assistant answering questions about documents. 
You have been provided with a summary of a document.

Document Summary:
{summary}

Original Document: {document_name}

User Query: {query}

Please provide a concise and accurate answer to the user's query based on the document summary.
If the summary doesn't contain relevant information to answer the query, say so clearly.
Format your answer in a clear, friendly manner.

Answer:
"""
            
            # Fill the query prompt
            filled_query_prompt = query_prompt.format(
                summary=document_summary,
                document_name=os.path.basename(file_path),
                query=query
            )
            
            # Use LLM to answer the query based on the summary
            query_answer = qa_chain.llm.invoke(filled_query_prompt).strip()
            
            # Print the query answer
            print("\n=== Query Answer ===")
            print(query_answer)
            print("-" * 50)
            
            # Calculate total processing time
            elapsed_time = time.time() - start_time
            print(f"Analysis completed in {elapsed_time:.2f} seconds")
            
            # Create source documents in the expected format for app.py
            source_documents = []
            seen_content = set()  # Track seen content to avoid duplicates
            
            # First add documents from the uploaded file
            for doc in documents:
                # Create a shorter version for deduplication checking
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    source_documents.append({
                        "source": os.path.basename(file_path),
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    })
                    
                    # Limit to 3 documents from the uploaded file
                    if len(source_documents) >= 3:
                        break
            
            # Then add similar documents from knowledge base
            for doc in similar_docs:
                source = doc.metadata.get("source", "Knowledge Base")
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    source_documents.append({
                        "source": source,
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                    })
                    
                    # Limit to total of 5 documents
                    if len(source_documents) >= 5:
                        break
                
            # Return results in format compatible with app.py/app.js
            return {
                "error": False,
                "result": query_answer,
                "document_summary": document_summary,
                "research_directions": research_directions,
                "source_documents": source_documents,
                "processing_time": f"{elapsed_time:.2f} seconds"
            }
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Error analyzing file: {str(e)}")
            print(trace)
            return {
                "error": True,
                "message": f"Error analyzing file: {str(e)}",
                "trace": trace
            }
# =====================================================================
# Vector Database CRUD Operations
# =====================================================================

# Create helper function to get vectorstore
def get_vectorstore():
    """Initialize and return the vector store connection"""
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    
    if not os.path.exists(db_path) or not os.listdir(db_path):
        print("Vector database does not exist yet")
        return None
    
    return Chroma(persist_directory=db_path, embedding_function=embeddings)

# CREATE: Add new documents to the vector database
def add_documents(documents, source_name=None):
    """
    Add new documents to the vector database
    
    Parameters:
        documents (list): List of strings or Document objects to add
        source_name (str, optional): Source name to assign to text documents
        
    Returns:
        list: IDs of the added documents
    """
    print(f"Adding {len(documents)} documents to vector database...")
    
    # Convert strings to Document objects if needed
    processed_docs = []
    for doc in documents:
        if isinstance(doc, str):
            # Generate a unique identifier for the document
            doc_id = str(uuid.uuid4())
            # Create a Document object with metadata
            metadata = {"source": source_name or f"manual_entry_{doc_id}", "manual_entry": True}
            processed_docs.append(Document(page_content=doc, metadata=metadata))
        else:
            # Assume it's already a Document object
            processed_docs.append(doc)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", "!", "?", " ", ",", ";", ""]
    )
    splits = text_splitter.split_documents(processed_docs)
    
    # Get or create embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    
    # Check if vector store exists
    if os.path.exists(db_path) and os.listdir(db_path):
        # Add to existing store
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        ids = vectorstore.add_documents(splits)
        vectorstore.persist()
    else:
        # Create new store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_path
        )
        ids = [doc.id for doc in vectorstore._collection.get()["documents"]]
        vectorstore.persist()
    
    print(f"Successfully added {len(splits)} chunks to vector database")
    return ids

# READ: Search for documents by metadata
def search_documents_by_metadata(metadata_field, metadata_value, limit=10):
    """
    Search for documents by metadata field
    
    Parameters:
        metadata_field (str): Metadata field to search
        metadata_value (str): Value to search for
        limit (int): Maximum number of results
        
    Returns:
        list: Documents matching the metadata criteria
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return []
    
    # Use the where parameter to filter by metadata
    filter_dict = {metadata_field: {"$eq": metadata_value}}
    results = vectorstore.get(where=filter_dict, limit=limit)
    
    return results.get("documents", [])

# UPDATE: Update document metadata or content
def update_document(document_id, new_content=None, new_metadata=None):
    """
    Update a document's content and/or metadata
    
    Parameters:
        document_id (str): ID of the document to update
        new_content (str, optional): New content for the document
        new_metadata (dict, optional): New metadata for the document
        
    Returns:
        bool: Success status
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return False
    
    try:
        # Get the existing document
        results = vectorstore.get(ids=[document_id])
        if not results or len(results.get("documents", [])) == 0:
            print(f"Document with ID {document_id} not found")
            return False
        
        # Update content if provided
        if new_content is not None:
            # Delete existing document and re-add with new content
            vectorstore.delete(ids=[document_id])
            
            # Create new document with updated content
            metadata = results.get("metadatas", [{}])[0]
            if new_metadata:
                metadata.update(new_metadata)
            
            doc = Document(page_content=new_content, metadata=metadata)
            
            # Re-chunk and add the document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, 
                chunk_overlap=250
            )
            splits = text_splitter.split_documents([doc])
            vectorstore.add_documents(splits)
        # If only updating metadata
        elif new_metadata:
            # Get existing metadata
            existing_metadata = results.get("metadatas", [{}])[0]
            # Update with new values
            updated_metadata = {**existing_metadata, **new_metadata}
            # Update document metadata
            vectorstore.update_document(document_id=document_id, metadata=updated_metadata)
        
        vectorstore.persist()
        return True
    
    except Exception as e:
        print(f"Error updating document: {str(e)}")
        return False

# DELETE: Remove documents from the database
def delete_documents(document_ids=None, metadata_field=None, metadata_value=None):
    """
    Delete documents from the vector database
    
    Parameters:
        document_ids (list, optional): List of document IDs to delete
        metadata_field (str, optional): Metadata field to filter by
        metadata_value (str, optional): Value to filter by
        
    Returns:
        bool: Success status
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return False
    
    try:
        if document_ids:
            # Delete by IDs
            vectorstore.delete(ids=document_ids)
            print(f"Deleted {len(document_ids)} documents")
        elif metadata_field and metadata_value is not None:
            # Delete by metadata
            filter_dict = {metadata_field: {"$eq": metadata_value}}
            results = vectorstore.get(where=filter_dict)
            if results and "ids" in results and results["ids"]:
                vectorstore.delete(ids=results["ids"])
                print(f"Deleted {len(results['ids'])} documents matching {metadata_field}={metadata_value}")
            else:
                print(f"No documents found matching {metadata_field}={metadata_value}")
        else:
            print("Must provide either document_ids or metadata criteria")
            return False
        
        vectorstore.persist()
        return True
    
    except Exception as e:
        print(f"Error deleting documents: {str(e)}")
        return False

# Reset the entire vector database
def reset_vector_database():
    """
    Reset (delete) the entire vector database
    
    Returns:
        bool: Success status
    """
    try:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            print("Vector database has been reset")
        else:
            print("Vector database does not exist")
        return True
    except Exception as e:
        print(f"Error resetting vector database: {str(e)}")
        return False

# Get database statistics
def get_database_stats():
    """
    Get statistics about the vector database
    
    Returns:
        dict: Database statistics
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return {
            "document_count": 0,
            "sources": [],
            "status": "Vector database not initialized"
        }
    
    try:
        # Get all documents
        all_docs = vectorstore.get()
        
        # Count documents
        doc_count = len(all_docs.get("ids", []))
        
        # Get unique sources
        sources = {}
        metadatas = all_docs.get("metadatas", [])
        for metadata in metadatas:
            source = metadata.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Sort sources by count
        sorted_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "document_count": doc_count,
            "sources": sorted_sources,
            "status": "active"
        }
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")
        return {
            "document_count": -1,
            "sources": [],
            "status": f"Error: {str(e)}"
        }

# Add document search helper function
def search_documents(query, k=5):
    """Search related documents directly from vector store and display details
    
    Arguments:
        query: Search query
        k: Number of documents to return
        
    Returns:
        List of retrieved documents
    """
    print(f"Search query: {query}")
    
    try:
        # Set up embedding model
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"
        )
        
        # Load vector database
        if not os.path.exists(db_path) or not os.listdir(db_path):
            print("Vector database doesn't exist, please run initialize_system() first")
            return None
        
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        # Print search results
        print(f"\nFound {len(docs)} relevant documents:")
        
        for i, doc in enumerate(docs):
            print(f"\n----- Document {i+1} -----")
            print(f"Source: {doc.metadata.get('source', 'Unknown source')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print("\nContent excerpt:")
            print(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
            print("-" * 50)
        
        return docs
    
    except Exception as e:
        print(f"Error searching documents: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to import documents from directory into knowledge base
def import_documents_from_directory(directory_path):
    """
    Import all supported documents from a directory into the knowledge base
    
    Parameters:
        directory_path (str): Path to directory containing documents
        
    Returns:
        dict: Import results
    """
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return {
            "success": False,
            "message": f"Directory not found: {directory_path}"
        }
    
    try:
        # Load documents from the directory
        documents = load_documents(directory_path)
        
        if not documents:
            return {
                "success": False,
                "message": "No supported documents found in directory"
            }
        
        # Split documents
        splits = split_documents(documents)
        
        # Add to vector store
        document_ids = add_documents(splits)
        
        return {
            "success": True,
            "message": f"Successfully imported {len(splits)} document chunks from {len(documents)} files",
            "document_count": len(documents),
            "chunk_count": len(splits)
        }
    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return {
            "success": False,
            "message": f"Error importing documents: {str(e)}",
            "trace": trace
        }

# Main function for an enhanced API
def query_with_file_analysis(query, file_path=None, qa_chain=None):
    """
    Process a query with optional file analysis
    
    Parameters:
        query (str): User query
        file_path (str, optional): Path to file for analysis
        qa_chain: QA chain to use (if None, will use global qa_chain)
        
    Returns:
        dict: Query results
    """
    # Initialize QA chain if not provided
    if qa_chain is None:
        # Try to load from global
        try:
            from __main__ import qa_system
            qa_chain = qa_system
        except:
            # Initialize new system
            qa_chain = initialize_system()
    
    # If file is provided, do special analysis
    if file_path and os.path.exists(file_path):
        return analyze_uploaded_file(file_path, query, qa_chain)
    
    # Regular query processing
    else:
        result = query_system(qa_chain, query)
        
        if result is None:
            return {
                "error": True,
                "message": "Query processing failed"
            }
        
        # Format response
        return {
            "error": False,
            "result": result["result"],
            "source_documents": [
                {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
                for doc in result.get("source_documents", [])[:5]  # Limit to top 5
            ]
        }

# If run directly, initialize system
# if __name__ == "__main__":
#     print("Initializing laboratory knowledge system...")
#     qa_system = initialize_system()
    
#     if qa_system:
#         print("\nSystem initialized successfully!")
#         print("You can now use query_system(qa_system, 'your query') to ask questions")
#         print("Use query_with_file_analysis('your query', 'path/to/file.pdf') to analyze uploaded files")
#     else:
#         print("\nSystem initialization failed")