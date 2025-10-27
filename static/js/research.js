// Research assistant front-end logic tailored to the simplified backend.

// Global variables
let selectedModel = 'anthropic';
let isProcessing = false;
let conversationHistory = [];
let isUserScrolling = false;
let attachedFiles = [];

// Abort controller for cancelling streaming requests (prevents memory leaks)
let currentAbortController = null;
let currentReader = null;

// Smart auto-scroll management
let scrollTimeout;
const chatMessages = document.getElementById('chatMessages');

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setupSmartScroll();
    loadStats();
});

// Event listeners setup
function initializeEventListeners() {
    // Send button
    const sendBtn = document.getElementById('sendBtn');
    if (sendBtn) {
        sendBtn.addEventListener('click', sendQuery);
    }
    
    // Chat input - Enter to send
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuery();
            }
        });
        
        // Auto-resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }
    
    // Model selector
    const modelSelector = document.getElementById('modelSelector');
    if (modelSelector) {
        modelSelector.value = 'anthropic';
    }
    
    // New Chat button
    const newChatBtn = document.getElementById('newChatBtn');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', newChat);
    }
    
    // Attach button (for file/image upload)
    const attachBtn = document.getElementById('attachBtn');
    const attachPdfBtn = document.getElementById('attachPdfBtn');
    const fileInput = document.getElementById('fileInput');
    
    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', function() {
            fileInput.accept = '.png,.jpg,.jpeg,.webp,.bmp,.tiff,.gif';
            fileInput.dataset.mode = 'image';
            fileInput.click();
        });
    }

    if (attachPdfBtn && fileInput) {
        attachPdfBtn.addEventListener('click', function() {
            fileInput.accept = '.pdf';
            fileInput.dataset.mode = 'pdf';
            fileInput.click();
        });
    }
    
    if (fileInput) {
        fileInput.addEventListener('change', async function(e) {
            const file = e.target.files[0];
            if (!file) return;

            const mode = fileInput.dataset.mode || 'image';

            if (mode === 'pdf') {
                await uploadPdf(file);
            } else {
                await uploadImage(file);
            }

            fileInput.value = '';
            delete fileInput.dataset.mode;
        });
    }

    const docsBtn = document.getElementById('docsBtn');
    if (docsBtn) {
        docsBtn.addEventListener('click', () => {
            showNotification('Docs panel coming soon.', 'info');
        });
    }
}

function getModelLabel(modelId) {
    return 'Anthropic Claude';
}

// Smart scroll setup
function setupSmartScroll() {
    if (!chatMessages) return;
    
    chatMessages.addEventListener('scroll', () => {
        const scrollBottom = chatMessages.scrollHeight - chatMessages.scrollTop - chatMessages.clientHeight;
        isUserScrolling = scrollBottom > 50;
        
        const jumpBtn = document.getElementById('jumpToBottom');
        if (isUserScrolling && jumpBtn) {
            jumpBtn.classList.remove('hidden');
        } else if (jumpBtn) {
            jumpBtn.classList.add('hidden');
        }
    });
    
    // Jump to bottom button
    const jumpBtn = document.getElementById('jumpToBottom');
    if (jumpBtn) {
        jumpBtn.addEventListener('click', () => {
            scrollToBottom();
            isUserScrolling = false;
            jumpBtn.classList.add('hidden');
        });
    }
}

function scrollToBottom() {
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function smartAutoScroll() {
    if (!isUserScrolling) {
        scrollToBottom();
    }
}

// New Chat function
function newChat() {
    conversationHistory = [];
    const chatMessagesDiv = document.getElementById('chatMessages');
    if (chatMessagesDiv) {
        chatMessagesDiv.innerHTML = '';
    }
    showNotification('New chat started', 'success');
}

// ============================================
// Image Upload with OCR
// ============================================

async function uploadImage(file) {
    try {
        showNotification(`📷 Uploading ${file.name}...`, 'info');
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/rag/upload-image', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showNotification('✅ Image text ingested. Ask questions about it now.', 'success');
            addMessage('system', `📷 **Image Uploaded:** ${result.source || file.name}\nText stored for retrieval.`, false);
            loadStats();
        } else {
            showNotification(`❌ Failed to process image: ${result.error || 'Unknown error'}`, 'error');
        }

    } catch (error) {
        console.error('Error uploading image:', error);
        showNotification(`❌ Error uploading image: ${error.message}`, 'error');
    }
}

async function uploadPdf(file) {
    try {
        showNotification(`📄 Uploading ${file.name}...`, 'info');
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/rag/upload-pdf', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (result.success) {
            showNotification('✅ PDF ingested successfully.', 'success');
            addMessage('system', `📄 **PDF Uploaded:** ${result.source}\nChunks indexed: ${result.chunks}`, false);
            loadStats();
        } else {
            showNotification(`❌ PDF upload failed: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('PDF upload error:', error);
        showNotification(`❌ PDF upload error: ${error.message}`, 'error');
    }
}

// ============================================
// Enhanced Airplane Takeoff Animation
// ============================================

// Enhanced airplane takeoff animation with smoke trail and variations
function triggerAirplaneTakeoff(messageLength = 50) {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;
    
    // Calculate duration based on message length (1.5s to 3.5s range)
    // Longer messages = longer flight time
    const baseDuration = 3.5;
    const maxDuration = 6.5;
    const durationMultiplier = Math.min(messageLength / 200, 1); // Cap at 200 chars
    const duration = baseDuration + (maxDuration - baseDuration) * durationMultiplier;
    
    // Random trajectory selection
    const trajectories = [
        'airplane-straight',
        'airplane-curve-left',
        'airplane-curve-right',
        'airplane-zigzag',
        'airplane-spiral'
    ];

    const randomTrajectory = 'airplane-straight';
    // Random Trjectory const randomTrajectory = trajectories[Math.floor(Math.random() * trajectories.length)];
    
    // Get send button position
    const btnRect = sendBtn.getBoundingClientRect();
    
    // Create airplane element
    const airplane = document.createElement('img');
    airplane.src = '/static/images/Aerospace.png';
    airplane.className = `airplane-animation ${randomTrajectory}`;
    airplane.style.left = `${btnRect.left + btnRect.width / 2 - 30}px`;
    airplane.style.top = `${btnRect.top}px`;
    airplane.style.setProperty('--duration', `${duration}s`);
    
    // Add to document
    document.body.appendChild(airplane);
    
    // Create smoke trail
    createSmokeTrail(airplane, duration);
    
    // Remove after animation completes
    setTimeout(() => {
        if (airplane.parentNode) {
            airplane.parentNode.removeChild(airplane);
        }
    }, duration * 1000);
}

// Generate smoke trail that follows the airplane
function createSmokeTrail(airplane, duration) {
    const smokeInterval = 80; // Generate smoke puff every 80ms
    const totalPuffs = Math.floor((duration * 1000) / smokeInterval);
    let puffCount = 0;
    
    const smokeTimer = setInterval(() => {
        if (puffCount >= totalPuffs || !airplane.parentNode) {
            clearInterval(smokeTimer);
            return;
        }
        
        // Get current airplane position
        const rect = airplane.getBoundingClientRect();
        
        // Create smoke puff
        const smoke = document.createElement('div');
        smoke.className = 'smoke-trail smoke-puff';
        smoke.style.left = `${rect.left + rect.width / 2 - 15}px`;
        smoke.style.top = `${rect.top + rect.height - 10}px`;
        
        // Add slight random offset for more natural look
        const randomX = (Math.random() - 0.5) * 10;
        const randomY = (Math.random() - 0.5) * 10;
        smoke.style.transform = `translate(${randomX}px, ${randomY}px)`;
        
        document.body.appendChild(smoke);
        
        // Remove smoke puff after animation
        setTimeout(() => {
            if (smoke.parentNode) {
                smoke.parentNode.removeChild(smoke);
            }
        }, 1500);
        
        puffCount++;
    }, smokeInterval);
}

// Send query with streaming
async function sendQuery() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question || isProcessing) return;

    const useWeb = document.getElementById('useWebSearch').checked;
    const useTrainingData = document.getElementById('useTrainingData').checked;
    const topK = parseInt(document.getElementById('topK').value) || 3;

    addMessage(question, true);
    
    // Trigger airplane takeoff animation with message length
    triggerAirplaneTakeoff(question.length);
    
    input.value = '';
    input.style.height = 'auto';
    isProcessing = true;
    isUserScrolling = false; // Reset scroll state on new message

    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = false;  // Keep enabled for stop button
    sendBtn.innerHTML = '<i class="fas fa-stop"></i>';
    sendBtn.onclick = stopStreaming;  // Add stop handler

    const streamingContent = createStreamingMessage();
    let fullResponse = '';
    let sources = [];
    let webResults = [];
    let indexingInfo = null;

    try {
        // Create abort controller for this request
        currentAbortController = new AbortController();
        
        const response = await fetch('/api/rag/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                top_k: topK,
                use_web: useWeb,
                stream: true,
                model_type: selectedModel,
                use_training_data: useTrainingData,
            }),
            signal: currentAbortController.signal  // Add abort signal
        });

        currentReader = response.body.getReader();  // Store reader reference
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await currentReader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.content) {
                            fullResponse += data.content;
                            streamingContent.innerHTML = formatContent(fullResponse) + '<span class="streaming-cursor"></span>';
                            smartAutoScroll();
                        } else if (data.sources || data.done || data.indexing_info) {
                            const cursor = streamingContent.querySelector('.streaming-cursor');
                            if (cursor) cursor.remove();
                            
                            // Apply syntax highlighting to completed code blocks
                            if (typeof hljs !== 'undefined') {
                                streamingContent.querySelectorAll('pre code:not(.hljs)').forEach((block) => {
                                    hljs.highlightElement(block);
                                });
                            }
                            
                            // Add copy button to completed message
                            if (!streamingContent.querySelector('.copy-message-btn')) {
                                const copyBtn = document.createElement('button');
                                copyBtn.className = 'copy-message-btn';
                                copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                                copyBtn.title = 'Copy message';
                                copyBtn.onclick = () => copyMessageText(fullResponse, copyBtn);
                                streamingContent.appendChild(copyBtn);
                            }
                            
                            if (data.sources) {
                                sources = data.sources;
                                webResults = data.web_results || [];
                                indexingInfo = data.indexing_info || null;
                                addSourcesToMessage(streamingContent, sources, webResults, indexingInfo);
                                
                                // Refresh search engine stats if new domains were indexed
                                if (indexingInfo && indexingInfo.newly_indexed_domains && indexingInfo.newly_indexed_domains.length > 0) {
                                    setTimeout(() => loadStats(), 1000);
                                }
                            }
                            if (data.done) {
                                // Done signal
                            }
                        } else if (data.error) {
                            streamingContent.innerHTML = `Error: ${data.error}`;
                        }
                    } catch (e) {
                        console.error('Parse error:', e);
                    }
                }
            }
            
            smartAutoScroll();
        }

        conversationHistory.push({ role: 'user', content: question });
        if (fullResponse) conversationHistory.push({ role: 'assistant', content: fullResponse });

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('✓ Stream aborted by user');
            const cursor = streamingContent.querySelector('.streaming-cursor');
            if (cursor) cursor.remove();
            streamingContent.innerHTML += '\n\n<em style="color: var(--text-muted); font-size: 0.9em;">[Response stopped by user]</em>';
            // Add partial response to history if exists
            if (fullResponse) {
                conversationHistory.push({ role: 'assistant', content: fullResponse + ' [stopped]' });
            }
            return;  // Don't show error notification
        }
        streamingContent.innerHTML = 'Error processing request.';
        showNotification('Error sending query', 'error');
    } finally {
        // Cleanup abort controller and reader
        currentAbortController = null;
        currentReader = null;
        
        // Reset button
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        sendBtn.onclick = null;  // Remove stop handler
        isProcessing = false;
        scrollToBottom();
    }
}

// Stop streaming function (prevents memory leaks from uncancelled requests)
function stopStreaming() {
    console.log('🛑 Stopping stream...');
    
    // Abort the fetch request
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    
    // Cancel the reader
    if (currentReader) {
        currentReader.cancel().catch(err => {
            console.log('Reader cancellation:', err);
        });
        currentReader = null;
    }
    
    // Reset UI
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = false;
    sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
    sendBtn.onclick = null;  // Remove stop handler
    
    isProcessing = false;
}

// Add message to chat
function addMessage(content, isUser = false) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = isUser ? '<img src="/static/images/Hacker.svg" alt="User" style="width: 50px; height: 50px;">' : '<img src="/static/images/1D.png" alt="AI Assistant" style="width: 50px; height: 50px;">';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (isUser) {
        contentDiv.textContent = content;
    } else {
        contentDiv.innerHTML = formatContent(content);
        
        // Apply syntax highlighting to code blocks
        setTimeout(() => {
            if (typeof hljs !== 'undefined') {
                contentDiv.querySelectorAll('pre code:not(.hljs)').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
        }, 50);
    }

    // Add copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-message-btn';
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.title = 'Copy message';
    copyBtn.onclick = () => copyMessageText(content, copyBtn);
    contentDiv.appendChild(copyBtn);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

// Create streaming message
function createStreamingMessage() {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<img src="/static/images/1D.png" alt="AI Assistant" style="width: 50px; height: 50px;">';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<span class="streaming-cursor"></span>';

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);
    scrollToBottom();

    return contentDiv;
}

// Format content with markdown
function formatContent(text) {
    const codeBlocks = [];
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
        codeBlocks.push({ language: lang || 'plaintext', code: code.trim() });
        return placeholder;
    });

    const inlineCodes = [];
    text = text.replace(/`([^`]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodes.length}__`;
        inlineCodes.push(code);
        return placeholder;
    });

    text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" style="color: var(--accent);">$1</a>');
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/\n/g, '<br>');

    inlineCodes.forEach((code, i) => {
        text = text.replace(`__INLINE_CODE_${i}__`, `<code>${escapeHtml(code)}</code>`);
    });

    codeBlocks.forEach((block, i) => {
        const escapedCode = escapeHtml(block.code);
        text = text.replace(`__CODE_BLOCK_${i}__`, 
            `<pre><code class="language-${block.language}">${escapedCode}</code></pre>`);
    });

    return text;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add sources to message
function addSourcesToMessage(contentDiv, sources, webResults, indexingInfo) {
    if (indexingInfo) {
        const indexDiv = document.createElement('div');
        indexDiv.style.cssText = 'padding: 12px; background: rgba(201, 42, 42, 0.1); border-radius: 8px; margin-top: 12px; border-left: 3px solid var(--accent);';
        
        let indexHtml = '<strong style="color: var(--accent); display: flex; align-items: center; gap: 8px; margin-bottom: 8px;"><i class="fas fa-database"></i>Search Engine Activity</strong>';
        
        if (indexingInfo.newly_indexed_domains && indexingInfo.newly_indexed_domains.length > 0) {
            indexHtml += `
                <div style="color: var(--success); margin-bottom: 4px;">
                    <i class="fas fa-check-circle"></i> Indexed ${indexingInfo.newly_indexed_domains.length} new domain(s): ${indexingInfo.newly_indexed_domains.join(', ')}
                </div>
            `;
        }
        
        indexHtml += `
            <div style="color: var(--text-secondary); font-size: 0.85rem;">
                Total indexed: ${indexingInfo.total_indexed_domains || 0} domains, ${indexingInfo.total_indexed_pages || 0} pages
            </div>
        `;
        
        indexDiv.innerHTML = indexHtml;
        contentDiv.appendChild(indexDiv);
    }
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.style.cssText = 'margin-top: 12px;';
        sourcesDiv.innerHTML = '<strong style="color: var(--accent);">Sources:</strong>';
        sources.forEach((source, i) => {
            const sourceItem = document.createElement('div');
            sourceItem.style.cssText = 'padding: 8px; background: var(--bg-tertiary); border-radius: 8px; margin-top: 4px; font-size: 0.85rem;';
            sourceItem.textContent = `[${i + 1}] ${source.source}`;
            sourcesDiv.appendChild(sourceItem);
        });
        contentDiv.appendChild(sourcesDiv);
    }

    if (webResults && webResults.length > 0) {
        const webDiv = document.createElement('div');
        webDiv.style.cssText = 'margin-top: 12px;';
        webDiv.innerHTML = '<strong style="color: var(--accent);">Web Sources:</strong>';
        webResults.forEach((result, i) => {
            const webItem = document.createElement('div');
            webItem.style.cssText = 'padding: 12px; background: var(--bg-tertiary); border-radius: 8px; margin-top: 8px; border-left: 3px solid var(--accent);';
            
            const sourceIcon = result.source === 'google' ? '🌐' : result.source === 'local_index' ? '📚' : '🔍';
            const sourceLabel = result.source === 'google' ? 'Current Web' : result.source === 'local_index' ? 'Indexed' : 'Web';
            
            webItem.innerHTML = `
                <div style="font-weight: 600; color: var(--accent); margin-bottom: 4px;">
                    ${sourceIcon} [${i + 1}] ${result.title}
                    <span style="font-size: 0.75rem; color: var(--text-muted); margin-left: 8px;">(${sourceLabel})</span>
                </div>
                <div style="color: var(--text-muted); font-size: 0.85rem; margin-bottom: 4px;">${result.url}</div>
                <div style="color: var(--text-secondary);">${result.snippet}</div>
            `;
            webDiv.appendChild(webItem);
        });
        contentDiv.appendChild(webDiv);
    }
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `toast ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);

    setTimeout(() => notification.remove(), 5000);
}

// Load stats
async function loadStats() {
    try {
        const response = await fetch('/health');
        const result = await response.json();
        const vectorCount = document.getElementById('docCount');
        const webCount = document.getElementById('webCount');
        if (vectorCount) vectorCount.textContent = result.vector_count || 0;
        if (webCount) webCount.textContent = result.indexed_pages || 0;
    } catch (error) {
        console.error('Stats load error:', error);
    }
}

// Check training data status
async function checkTrainingDataStatus() {
    // deprecated functionality removed
}

async function fetchCustomModelInfo() {
    // deprecated functionality removed
}

// Search Engine Functions
function renderSearchResults(results, query) {
    const container = document.getElementById('searchEngineResults');
    if (!container) return;

    if (!results || results.length === 0) {
        container.innerHTML = `<div class="empty-state">No results found for "${query}".</div>`;
        return;
    }

    container.innerHTML = results.map((result, idx) => `
        <div class="search-result">
            <div class="search-result-header">
                <span class="search-index">[${idx + 1}]</span>
                <a href="${result.url}" target="_blank">${result.title}</a>
            </div>
            <div class="search-result-snippet">${result.snippet}</div>
            <div class="search-result-meta">Score: ${result.score?.toFixed(2) || 'N/A'} | ${result.timestamp || ''}</div>
        </div>
    `).join('');
}

async function performEnhancedSearch() {
    const queryInput = document.getElementById('enhancedSearchQuery');
    if (!queryInput) return;

    const query = queryInput.value.trim();
    if (!query) {
        showNotification('Enter a query to search.', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/search/local', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        const result = await response.json();
        const items = result.success ? result.results : [];
        renderSearchResults(items, query);
        showNotification(`Found ${items.length} results`, 'success');
    } catch (error) {
        console.error('Local search error:', error);
        showNotification(`Search error: ${error.message}`, 'error');
    }
}

async function performLocalSearch() {
    performEnhancedSearch();
}

async function performManualIndexing() {
    const urlInput = document.getElementById('manualIndexUrl');
    if (!urlInput) return;

    const url = urlInput.value.trim();
    if (!url) {
        showNotification('Enter a URL to index.', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/search/crawl', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ urls: [url], max_depth: 1 })
        });

        const result = await response.json();
        if (result.success) {
            showNotification(`Indexed ${result.indexed} pages (total ${result.total}).`, 'success');
            loadStats();
        } else {
            showNotification(`Indexing failed: ${result.error || 'Unknown error'}`, 'error');
        }
    } catch (error) {
        console.error('Manual indexing error:', error);
        showNotification(`Indexing error: ${error.message}`, 'error');
    }
}

async function clearSearchIndex() {
    showNotification('Clearing index not supported in this build.', 'info');
}

// Copy message text to clipboard
function copyMessageText(text, button) {
    // Strip HTML tags for plain text copy
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = text;
    const plainText = tempDiv.textContent || tempDiv.innerText || '';
    
    navigator.clipboard.writeText(plainText).then(() => {
        // Change icon to checkmark
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.classList.add('copied');
        
        // Show notification
        showNotification('Message copied to clipboard!', 'success');
        
        // Reset icon after 2 seconds
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
        showNotification('Failed to copy message', 'error');
    });
}
