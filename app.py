from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import time
import sys
import os
import gc
import json
import uuid
import shutil
import threading
from werkzeug.utils import secure_filename

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import labagent module
try:
    from labagent import (
        initialize_system, query_system, query_with_file_analysis,
        add_documents, delete_documents, search_documents_by_metadata,
        update_document, reset_vector_database, get_database_stats,
        search_documents, import_documents_from_directory
    )
except ImportError:
    print("Error: Cannot import labagent module. Make sure labagent.py is in the same directory.")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Configuration
MAX_QUERY_LENGTH = 1000  # Maximum query length
QUERY_TIMEOUT = 300  # Query timeout in seconds
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'csv', 'xlsx', 'xls', 'ppt', 'pptx'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store QA system instance
qa_system = None
initialization_lock = threading.Lock()
system_status = {"status": "initializing", "message": "System is starting up"}
initialization_thread = None

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class QueryResult:
    """Class to store query results"""
    def __init__(self):
        self.result = ""
        self.sources = []
        self.document_summary = ""
        self.research_directions = ""
        self.processing_time = ""
        self.error = False
        self.message = ""

# Function to process query (without signal handling, using timeout control)
def process_query(query_text, file_path=None):
    """Process query and return results and sources"""
    result = QueryResult()
    
    try:
        # Execute query
        start_time = time.time()
        
        if file_path:
            # Query with file analysis
            print(f"Processing query with file analysis: {file_path}")
            qa_response = query_with_file_analysis(query_text, file_path, qa_system)
        else:
            # Standard query
            print(f"Processing standard query")
            qa_response = query_system(qa_system, query_text)
        
        # Process response
        if qa_response:
            # Check for error
            if isinstance(qa_response, dict) and qa_response.get("error", False):
                result.error = True
                result.message = qa_response.get("message", "Unknown error occurred")
            else:
                # Extract result from response
                if isinstance(qa_response, dict):
                    result.result = qa_response.get("result", "")
                    result.document_summary = qa_response.get("document_summary", "")
                    result.research_directions = qa_response.get("research_directions", "")
                    
                    # Extract sources
                    if "source_documents" in qa_response:
                        source_docs = qa_response["source_documents"]
                        if isinstance(source_docs, list) and all(isinstance(x, dict) for x in source_docs):
                            # New format with dict sources
                            result.sources = [doc.get("source", "Unknown source") for doc in source_docs[:5]]
                        else:
                            # Old format with Document objects
                            unique_sources = set()
                            for doc in qa_response.get("source_documents", [])[:5]:
                                if hasattr(doc, 'metadata'):
                                    source = doc.metadata.get("source", "Unknown source")
                                    if source not in unique_sources:
                                        unique_sources.add(source)
                                        result.sources.append(source)
                else:
                    # Legacy format
                    result.result = qa_response["result"]
                    
                    # Extract sources
                    unique_sources = set()
                    for doc in qa_response.get("source_documents", [])[:5]:
                        source = doc.metadata.get("source", "Unknown source")
                        if source not in unique_sources:
                            unique_sources.add(source)
                            result.sources.append(source)
        else:
            result.error = True
            result.message = "No response received from the QA system"
                
    except Exception as e:
        result.error = True
        result.message = f"Error processing query: {str(e)}"
        import traceback
        traceback.print_exc()
    finally:
        # Force garbage collection
        gc.collect()
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    result.processing_time = f"{elapsed_time:.2f} seconds"
    
    return result

# API Endpoints
@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle query API endpoint"""
    global qa_system
    
    # Get request data
    data = request.json
    query = data.get('query', '')
    print(f"Query received: {query[:100]}{'...' if len(query) > 100 else ''}")
    
    # Check if query is empty
    if not query.strip():
        return jsonify({
            'error': True,
            'message': 'Query cannot be empty'
        }), 400
    
    # Check query length
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            'error': True,
            'message': f'Query length cannot exceed {MAX_QUERY_LENGTH} characters'
        }), 400
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later',
            'status': system_status
        }), 503
    
    try:
        # Process query
        print("Processing query...")
        result = process_query(query)
        
        # Handle errors
        if result.error:
            return jsonify({
                'error': True,
                'message': result.message
            }), 500
        
        # Return response
        return jsonify({
            'error': False,
            'result': result.result,
            'sources': result.sources,
            'processing_time': result.processing_time
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error processing query: {str(e)}'
        }), 500

@app.route('/api/query-with-file', methods=['POST'])
def handle_query_with_file():
    """Handle query with file analysis"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later',
            'status': system_status
        }), 503
    
    # Check if file was included
    if 'file' not in request.files:
        return jsonify({
            'error': True,
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'error': True,
            'message': 'No file selected'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': True,
            'message': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    # Get query text
    query = request.form.get('query', '')
    
    # Check if query is empty
    if not query.strip():
        return jsonify({
            'error': True,
            'message': 'Query cannot be empty'
        }), 400
    
    # Check query length
    if len(query) > MAX_QUERY_LENGTH:
        return jsonify({
            'error': True,
            'message': f'Query length cannot exceed {MAX_QUERY_LENGTH} characters'
        }), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(file_path)
        
        print(f"File uploaded: {filename} (saved as {file_path})")
        
        # Process query with file
        result = process_query(query, file_path)
        
        # Clean up file after processing
        try:
            os.remove(file_path)
        except:
            print(f"Warning: Could not remove temporary file {file_path}")
        
        # Handle errors
        if result.error:
            return jsonify({
                'error': True,
                'message': result.message
            }), 500
        
        # Return response
        return jsonify({
            'error': False,
            'result': result.result,
            'document_summary': result.document_summary,
            'research_directions': result.research_directions,
            'sources': result.sources,
            'processing_time': result.processing_time,
            'analyzed_file': filename
        })
    
    except Exception as e:
        # Clean up file in case of error
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'error': True,
            'message': f'Error processing file: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status API endpoint"""
    global qa_system, system_status
    
    # Get database stats if system is initialized
    db_stats = {}
    if qa_system is not None:
        try:
            db_stats = get_database_stats()
        except:
            db_stats = {"status": "error", "message": "Could not retrieve database statistics"}
    
    # Return status
    return jsonify({
        'status': 'ready' if qa_system is not None else 'initializing',
        'system_info': system_status,
        'max_query_length': MAX_QUERY_LENGTH,
        'allowed_file_types': list(ALLOWED_EXTENSIONS),
        'database_stats': db_stats
    })

# Vector Database Management Endpoints
@app.route('/api/documents', methods=['POST'])
def add_document():
    """API endpoint to add documents to vector database"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    # Get request data
    data = request.json
    
    # Validate required fields
    if not data.get('content'):
        return jsonify({
            'error': True,
            'message': 'Document content is required'
        }), 400
    
    content = data.get('content')
    source_name = data.get('source', 'API Upload')
    
    try:
        # Call the add_documents function
        doc_ids = add_documents([content], source_name)
        
        return jsonify({
            'error': False,
            'message': 'Document added successfully',
            'document_ids': doc_ids
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error adding document: {str(e)}'
        }), 500

@app.route('/api/documents/file', methods=['POST'])
def add_document_file():
    """API endpoint to add a file as document to vector database"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    # Check if file was included
    if 'file' not in request.files:
        return jsonify({
            'error': True,
            'message': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'error': True,
            'message': 'No file selected'
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'error': True,
            'message': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        print(f"File uploaded for knowledge base: {filename}")
        
        # Import the file to knowledge base
        result = import_documents_from_directory(UPLOAD_FOLDER)
        
        # Clean up after import
        try:
            os.remove(file_path)
        except:
            print(f"Warning: Could not remove imported file {file_path}")
        
        # Return result
        if result.get('success', False):
            return jsonify({
                'error': False,
                'message': result.get('message', 'File imported successfully'),
                'document_count': result.get('document_count', 0),
                'chunk_count': result.get('chunk_count', 0)
            })
        else:
            return jsonify({
                'error': True,
                'message': result.get('message', 'Failed to import file')
            }), 500
    
    except Exception as e:
        # Clean up file in case of error
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'error': True,
            'message': f'Error importing file: {str(e)}'
        }), 500

@app.route('/api/documents/directory', methods=['POST'])
def import_directory():
    """API endpoint to import a directory of documents"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    # Get request data
    data = request.json
    
    # Validate required fields
    if not data.get('directory_path'):
        return jsonify({
            'error': True,
            'message': 'Directory path is required'
        }), 400
    
    directory_path = data.get('directory_path')
    
    try:
        # Check if directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            return jsonify({
                'error': True,
                'message': f'Directory not found: {directory_path}'
            }), 404
        
        # Import the directory
        result = import_documents_from_directory(directory_path)
        
        # Return result
        if result.get('success', False):
            return jsonify({
                'error': False,
                'message': result.get('message', 'Directory imported successfully'),
                'document_count': result.get('document_count', 0),
                'chunk_count': result.get('chunk_count', 0)
            })
        else:
            return jsonify({
                'error': True,
                'message': result.get('message', 'Failed to import directory')
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error importing directory: {str(e)}'
        }), 500

@app.route('/api/documents', methods=['DELETE'])
def delete_document():
    """API endpoint to delete documents from vector database"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    # Get request data
    data = request.json
    
    # Check if document IDs or metadata is provided
    if not data.get('document_ids') and not (data.get('metadata_field') and 'metadata_value' in data):
        return jsonify({
            'error': True,
            'message': 'Either document_ids or metadata_field and metadata_value must be provided'
        }), 400
    
    try:
        # Delete by document IDs
        if data.get('document_ids'):
            success = delete_documents(document_ids=data.get('document_ids'))
            if success:
                return jsonify({
                    'error': False,
                    'message': f"Successfully deleted {len(data.get('document_ids'))} documents"
                })
            else:
                return jsonify({
                    'error': True,
                    'message': 'Failed to delete documents'
                }), 500
        
        # Delete by metadata
        elif data.get('metadata_field') and 'metadata_value' in data:
            success = delete_documents(
                metadata_field=data.get('metadata_field'),
                metadata_value=data.get('metadata_value')
            )
            if success:
                return jsonify({
                    'error': False,
                    'message': f"Successfully deleted documents matching {data.get('metadata_field')}={data.get('metadata_value')}"
                })
            else:
                return jsonify({
                    'error': True,
                    'message': 'Failed to delete documents'
                }), 500
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error deleting documents: {str(e)}'
        }), 500

@app.route('/api/documents/search', methods=['POST'])
def search_documents_endpoint():
    """API endpoint to search documents"""
    global qa_system
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    # Get request data
    data = request.json
    
    # Validate required fields
    if not data.get('query'):
        return jsonify({
            'error': True,
            'message': 'Search query is required'
        }), 400
    
    query = data.get('query')
    limit = data.get('limit', 5)
    
    try:
        # Search documents
        docs = search_documents(query, k=limit)
        
        if docs is None:
            return jsonify({
                'error': True,
                'message': 'Search failed'
            }), 500
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                'id': getattr(doc, 'id', 'unknown'),
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        
        return jsonify({
            'error': False,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error searching documents: {str(e)}'
        }), 500

@app.route('/api/documents/reset', methods=['POST'])
def reset_database():
    """API endpoint to reset the entire vector database"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    try:
        # Reset database
        success = reset_vector_database()
        
        if success:
            return jsonify({
                'error': False,
                'message': 'Vector database has been reset successfully'
            })
        else:
            return jsonify({
                'error': True,
                'message': 'Failed to reset vector database'
            }), 500
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error resetting vector database: {str(e)}'
        }), 500
@app.route('/api/documents/sources', methods=['GET'])
def get_document_sources():
    """API endpoint to get all sources in the knowledge base"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    try:
        # Get database stats which includes sources
        stats = get_database_stats()
        
        # Format sources for the response
        sources = []
        if stats and 'sources' in stats:
            for source_tuple in stats['sources']:
                sources.append({
                    'name': source_tuple[0],
                    'count': source_tuple[1]
                })
        
        return jsonify({
            'error': False,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error getting document sources: {str(e)}'
        }), 500
@app.route('/api/documents/stats', methods=['GET'])
def database_stats():
    """API endpoint to get vector database statistics"""
    global qa_system
    
    # Check if QA system is initialized
    if qa_system is None:
        return jsonify({
            'error': True,
            'message': 'QA system not initialized, please try again later'
        }), 503
    
    try:
        # Get database stats
        stats = get_database_stats()
        
        return jsonify({
            'error': False,
            'stats': stats
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': f'Error getting database statistics: {str(e)}'
        }), 500

def initialize():
    """Initialize QA system"""
    global qa_system, system_status
    
    with initialization_lock:
        try:
            system_status = {"status": "initializing", "message": "Starting system initialization"}
            print("Initializing QA system...")
            
            # Force garbage collection
            gc.collect()
            
            # Initialize QA system
            system_status = {"status": "initializing", "message": "Loading models and documents"}
            qa_system = initialize_system()
            
            if qa_system is None:
                system_status = {"status": "error", "message": "QA system initialization failed"}
                print("QA system initialization failed")
                return False
            
            system_status = {"status": "ready", "message": "System is ready"}
            print("QA system initialization successful")
            return True
        
        except Exception as e:
            system_status = {"status": "error", "message": f"Error during initialization: {str(e)}"}
            print(f"Error during initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def initialize_in_thread():
    """Initialize QA system in a background thread"""
    initialize()

if __name__ == '__main__':
    # Start initialization in a background thread
    print("Starting initialization in background thread...")
    initialization_thread = threading.Thread(target=initialize_in_thread)
    initialization_thread.daemon = True
    initialization_thread.start()
    
    # Start Flask application
    print("Starting web server...")
    app.run(host='0.0.0.0', port=5001, debug=False)