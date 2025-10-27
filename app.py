"""
Flask application for the research assistant.
Only the /research UI and associated APIs are retained.
"""

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS

from rag_service import get_rag_service
from search_engine import get_search_service
from langchain_service import get_langchain_service
# pdf_processor and pdf_service are no longer used
# from pdf_processor import PDFProcessor
# from pdf_service import get_pdf_service
# embedding_manager is no longer directly used here

# Optional services (fail gracefully if missing)
try:
    import conversation_logger
    CONVERSATION_LOGGING_AVAILABLE = True
except ImportError:
    CONVERSATION_LOGGING_AVAILABLE = False

try:
    import notes_service
    NOTES_SERVICE_AVAILABLE = True
except ImportError:
    NOTES_SERVICE_AVAILABLE = False

try:
    import code_indexer
    CODE_INDEXER_AVAILABLE = True
except ImportError:
    CODE_INDEXER_AVAILABLE = False

load_dotenv()

app = Flask(__name__)
CORS(app)

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize with memory-safe settings for systems with limited RAM
# Use DeepSeek OCR for lower memory consumption if available
ENABLE_PDF_OCR = os.getenv("ENABLE_PDF_OCR", "true").lower() not in {"0", "false", "no"}

# pdf_processor = PDFProcessor(memory_safe=True, max_image_size=2000, use_deepseek=True, enable_ocr=ENABLE_PDF_OCR)
# pdf_service = get_pdf_service(enable_ocr=ENABLE_PDF_OCR)
rag = get_rag_service()
search = get_search_service()
vector_store = rag._get_collection()  # This might still be used for health checks
langchain_service = get_langchain_service()

# Start resource monitor in background
try:
    from resource_monitor import ResourceMonitor
    _monitor = ResourceMonitor()
    _monitor.start()
    logger.info("Resource monitor started")
except Exception as _e:
    logger.warning("Resource monitor failed to start: %s", _e)


def _log_conversation(user_message: str, assistant_message: str, metadata: dict | None = None) -> None:
    if not CONVERSATION_LOGGING_AVAILABLE:
        return
    try:
        conversation_logger.get_conversation_logger().log_exchange(  # type: ignore[attr-defined]
            user_message=user_message,
            assistant_message=assistant_message,
            system_prompt="Research assistant interaction",
            metadata=metadata or {},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed logging conversation: %s", exc)


@app.route("/")
def index():
    return render_template("research.html")


@app.route("/research")
def research_page():
    return render_template("research.html")


@app.route("/pdf-viewer")
def pdf_viewer_page():
    return render_template("pdf_viewer.html")


@app.route("/upload-pdf")
def upload_pdf_page():
    return render_template("upload_pdf.html")


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    # top_k = int(payload.get("top_k", 3)) # top_k is handled by the retrieval chain
    stream = bool(payload.get("stream", True))
    # These flags are part of the old RAG service logic which is being replaced for PDF queries
    # use_training_data = bool(payload.get("use_training_data", False))
    # use_web = bool(payload.get("use_web", False))
    # model_type = payload.get("model_type")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not stream:
        result = langchain_service.query(question)
        _log_conversation(
            question,
            result.get("answer", ""),
            {},
        )
        return jsonify(result)

    def responder():
        for chunk in langchain_service.query_stream(question):
            yield f"data: {chunk}\n\n"

    return Response(responder(), mimetype="text/event-stream")


@app.route("/api/rag/upload-pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "A PDF file is required"}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"success": False, "error": "Empty file"}), 400

    # Process and index the PDF using the new LangChain service
    result = langchain_service.index_pdf(file_bytes, file.filename)
    
    if not result["success"]:
        return jsonify({
            "success": False, 
            "error": f"PDF processing failed: {result.get('error')}"
        }), 500
    
    # The new service returns a dict that is already suitable for the response.
    # We just need to ensure all expected keys are present.
    response_data = {
        "success": True, 
        "document_id": result["document_id"],
        "filename": file.filename,
        "page_count": result["page_count"],
        "chunks": result["chunks"],
        "metadata": {} # metadata is not extracted by the new service in the same way. Returning empty.
    }
    return jsonify(response_data)


@app.route("/api/rag/upload-image", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    data = file.read()
    if not data:
        return jsonify({"success": False, "error": "Empty file"}), 400

    result = rag.ingest_image_bytes(data, file.filename)
    return jsonify(result)


@app.route("/api/search/local", methods=["POST"])
def search_local():
    payload = request.get_json(force=True)
    query = payload.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query required"}), 400

    results = search.local_search(query)
    return jsonify({"success": True, "results": results})


@app.route("/api/search/crawl", methods=["POST"])
def search_crawl():
    payload = request.get_json(force=True)
    urls = payload.get("urls", [])
    if not urls:
        return jsonify({"error": "urls list required"}), 400

    result = search.crawl(urls, max_depth=int(payload.get("max_depth", 1)))
    return jsonify(result)


# The following PDF routes might be deprecated if their functionality is fully replaced.
# For now, they are left as they might be used by other parts of the frontend.
@app.route("/api/pdf/document/<document_id>", methods=["GET"])
def get_pdf_document(document_id):
    """Get information about a specific PDF document"""
    # This relied on pdf_service, which is being removed. 
    # Returning a placeholder response.
    return jsonify({"success": False, "error": "This endpoint is deprecated."})

@app.route("/api/pdf/documents", methods=["GET"])
def list_pdf_documents():
    """List all PDF documents with pagination"""
    # This relied on pdf_service, which is being removed. 
    # Returning a placeholder response.
    return jsonify({"success": False, "error": "This endpoint is deprecated."})

@app.route("/api/pdf/page", methods=["GET"])
def get_pdf_page():
    """Get the full text of a specific PDF page"""
    # This relied on pdf_service, which is being removed. 
    # Returning a placeholder response.
    return jsonify({"success": False, "error": "This endpoint is deprecated."})

@app.route("/api/pdf/query", methods=["POST"])
def query_pdf_document():
    """Query a specific PDF document for relevant pages"""
    # This relied on pdf_service, which is being removed. 
    # Returning a placeholder response.
    return jsonify({"success": False, "error": "This endpoint is deprecated."})

@app.route("/health", methods=["GET"])
def health_check():
    search_stats = search.get_stats()
    
    # Get GPU information if available
    gpu_info = {}
    try:
        from embedding_manager import get_gpu_info, embedding_device
        gpu_info = get_gpu_info()
        embedding_dev = embedding_device()
    except ImportError:
        embedding_dev = "unknown"
    
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vector_count": vector_store.count(),
        "indexed_pages": search_stats.get('total_pages', 0),
        "indexed_domains": search_stats.get('total_domains', 0),
        "conversation_logging": CONVERSATION_LOGGING_AVAILABLE,
        "notes_service": NOTES_SERVICE_AVAILABLE,
        "code_indexer": CODE_INDEXER_AVAILABLE,
        "embedding_device": embedding_dev,
        "gpu_info": gpu_info,
        "platform": platform.platform(),
    })


@app.route("/ready", methods=["GET"])  # Simple readiness endpoint for Gunicorn/Nginx
def ready_check():
    try:
        # Quick, lightweight checks
        _ = vector_store.count()
        return jsonify({"status": "ready"})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "not_ready", "error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)