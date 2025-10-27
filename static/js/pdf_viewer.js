/**
 * PDF Viewer Component
 * Handles PDF document viewing, search, and page navigation
 */

class PDFViewer {
    constructor(containerElement) {
        this.container = containerElement;
        this.currentDocument = null;
        this.currentPage = 1;
        this.maxPage = 1;
        
        // Initialize UI
        this.initUI();
        
        // Event listeners
        this.attachEventListeners();
    }
    
    initUI() {
        // Create UI components
        this.container.innerHTML = `
            <div class="pdf-controls mb-3">
                <div class="upload-section">
                    <h3>Upload PDF Document</h3>
                    <form id="pdf-upload-form" enctype="multipart/form-data">
                        <div class="input-group mb-3">
                            <input type="file" class="form-control" id="pdf-file" name="file" accept=".pdf">
                            <button class="btn btn-primary" type="submit" id="pdf-upload-btn">Upload</button>
                        </div>
                    </form>
                    <div id="pdf-upload-status" class="alert alert-info d-none"></div>
                </div>
                
                <div id="pdf-document-section" class="d-none">
                    <h3 id="pdf-document-title">Document Title</h3>
                    <div class="d-flex justify-content-between mb-2">
                        <div>
                            <span id="pdf-page-info">Page 1 of 1</span>
                        </div>
                        <div class="btn-group">
                            <button id="pdf-prev-page" class="btn btn-sm btn-outline-secondary">Previous</button>
                            <button id="pdf-next-page" class="btn btn-sm btn-outline-secondary">Next</button>
                        </div>
                    </div>
                    
                    <div class="pdf-search-section mb-3">
                        <div class="input-group">
                            <input type="text" class="form-control" id="pdf-search-input" placeholder="Search in document...">
                            <button class="btn btn-outline-secondary" id="pdf-search-btn">Search</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="pdf-content-container" class="mb-4">
                <div id="pdf-loading" class="d-none text-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                
                <div id="pdf-content" class="border p-3 bg-white"></div>
                
                <div id="pdf-search-results" class="mt-3 d-none">
                    <h4>Search Results</h4>
                    <div id="pdf-search-results-list"></div>
                </div>
            </div>
        `;
        
        // Store UI elements
        this.uploadForm = this.container.querySelector('#pdf-upload-form');
        this.uploadStatus = this.container.querySelector('#pdf-upload-status');
        this.documentSection = this.container.querySelector('#pdf-document-section');
        this.documentTitle = this.container.querySelector('#pdf-document-title');
        this.pageInfo = this.container.querySelector('#pdf-page-info');
        this.prevButton = this.container.querySelector('#pdf-prev-page');
        this.nextButton = this.container.querySelector('#pdf-next-page');
        this.searchInput = this.container.querySelector('#pdf-search-input');
        this.searchButton = this.container.querySelector('#pdf-search-btn');
        this.loading = this.container.querySelector('#pdf-loading');
        this.contentContainer = this.container.querySelector('#pdf-content');
        this.searchResultsContainer = this.container.querySelector('#pdf-search-results');
        this.searchResultsList = this.container.querySelector('#pdf-search-results-list');
    }
    
    attachEventListeners() {
        // Upload form submission
        this.uploadForm.addEventListener('submit', e => {
            e.preventDefault();
            this.uploadPDF();
        });
        
        // Navigation buttons
        this.prevButton.addEventListener('click', () => this.navigateToPage(this.currentPage - 1));
        this.nextButton.addEventListener('click', () => this.navigateToPage(this.currentPage + 1));
        
        // Search button
        this.searchButton.addEventListener('click', () => this.searchDocument());
        
        // Search on enter key
        this.searchInput.addEventListener('keypress', e => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.searchDocument();
            }
        });
    }
    
    uploadPDF() {
        const fileInput = this.container.querySelector('#pdf-file');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showStatus('Please select a PDF file', 'danger');
            return;
        }
        
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            this.showStatus('Only PDF files are supported', 'danger');
            return;
        }
        
        this.showStatus('Uploading PDF...', 'info');
        this.showLoading(true);
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Upload file
        fetch('/api/rag/upload-pdf', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            this.showLoading(false);
            
            if (data.success) {
                this.showStatus(`PDF "${file.name}" uploaded successfully. ${data.page_count} pages processed.`, 'success');
                this.loadDocument(data.document_id);
            } else {
                this.showStatus(`Upload failed: ${data.error}`, 'danger');
            }
        })
        .catch(error => {
            this.showLoading(false);
            this.showStatus(`Upload error: ${error.message}`, 'danger');
        });
    }
    
    loadDocument(documentId) {
        this.showLoading(true);
        
        fetch(`/api/pdf/document/${documentId}`)
            .then(response => response.json())
            .then(data => {
                this.showLoading(false);
                
                if (data.success) {
                    this.currentDocument = data;
                    this.currentPage = 1;
                    this.maxPage = data.page_count;
                    
                    // Update UI
                    this.documentTitle.textContent = data.title || data.filename;
                    this.pageInfo.textContent = `Page ${this.currentPage} of ${this.maxPage}`;
                    this.documentSection.classList.remove('d-none');
                    
                    // Load first page
                    this.loadPage(this.currentPage);
                } else {
                    this.showStatus(`Error loading document: ${data.error}`, 'danger');
                }
            })
            .catch(error => {
                this.showLoading(false);
                this.showStatus(`Error: ${error.message}`, 'danger');
            });
    }
    
    loadPage(pageNumber) {
        if (!this.currentDocument || pageNumber < 1 || pageNumber > this.maxPage) {
            return;
        }
        
        this.showLoading(true);
        
        fetch(`/api/pdf/page?document_id=${this.currentDocument.document_id}&page_number=${pageNumber}`)
            .then(response => response.json())
            .then(data => {
                this.showLoading(false);
                
                if (data.success) {
                    // Update UI
                    this.currentPage = pageNumber;
                    this.pageInfo.textContent = `Page ${this.currentPage} of ${this.maxPage}`;
                    
                    // Display page content
                    this.contentContainer.innerHTML = `<pre class="pdf-page-text">${this.escapeHtml(data.text)}</pre>`;
                    
                    // Update button states
                    this.prevButton.disabled = (this.currentPage <= 1);
                    this.nextButton.disabled = (this.currentPage >= this.maxPage);
                    
                    // Scroll to top
                    this.contentContainer.scrollTop = 0;
                } else {
                    this.contentContainer.innerHTML = `<p class="text-danger">Error loading page: ${data.error}</p>`;
                }
            })
            .catch(error => {
                this.showLoading(false);
                this.contentContainer.innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
            });
    }
    
    navigateToPage(pageNumber) {
        if (pageNumber < 1 || pageNumber > this.maxPage) {
            return;
        }
        
        this.loadPage(pageNumber);
    }
    
    searchDocument() {
        const query = this.searchInput.value.trim();
        
        if (!this.currentDocument) {
            this.showStatus('Please load a document first', 'warning');
            return;
        }
        
        if (!query) {
            this.showStatus('Please enter a search query', 'warning');
            return;
        }
        
        this.showLoading(true);
        this.searchResultsContainer.classList.add('d-none');
        
        fetch('/api/pdf/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                document_id: this.currentDocument.document_id,
                query: query,
                top_k: 5
            })
        })
        .then(response => response.json())
        .then(data => {
            this.showLoading(false);
            
            if (data.success && data.results.length > 0) {
                // Clear previous results
                this.searchResultsList.innerHTML = '';
                
                // Display search results
                data.results.forEach(result => {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'search-result-item card mb-2';
                    resultItem.innerHTML = `
                        <div class="card-body">
                            <h5 class="card-title">Page ${result.page_number} (Score: ${result.score.toFixed(2)})</h5>
                            <p class="card-text">${this.escapeHtml(result.preview)}</p>
                            <button class="btn btn-sm btn-outline-primary go-to-page-btn" data-page="${result.page_number}">Go to page</button>
                        </div>
                    `;
                    this.searchResultsList.appendChild(resultItem);
                    
                    // Add click handler
                    resultItem.querySelector('.go-to-page-btn').addEventListener('click', () => {
                        this.navigateToPage(result.page_number);
                    });
                });
                
                // Show results
                this.searchResultsContainer.classList.remove('d-none');
            } else {
                this.searchResultsList.innerHTML = '<p>No results found</p>';
                this.searchResultsContainer.classList.remove('d-none');
            }
        })
        .catch(error => {
            this.showLoading(false);
            this.showStatus(`Search error: ${error.message}`, 'danger');
        });
    }
    
    showStatus(message, type = 'info') {
        this.uploadStatus.textContent = message;
        this.uploadStatus.className = `alert alert-${type}`;
        this.uploadStatus.classList.remove('d-none');
        
        // Auto hide after 5 seconds
        setTimeout(() => {
            this.uploadStatus.classList.add('d-none');
        }, 5000);
    }
    
    showLoading(show) {
        if (show) {
            this.loading.classList.remove('d-none');
        } else {
            this.loading.classList.add('d-none');
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the PDF viewer when the page is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if the PDF viewer container exists
    const pdfViewerContainer = document.getElementById('pdf-viewer-container');
    if (pdfViewerContainer) {
        window.pdfViewer = new PDFViewer(pdfViewerContainer);
    }
});

