/**
 * Conversation History Manager for Research Site
 * Replaces the notes feature with conversation history tracking
 */

// Global conversation history storage
let savedConversations = [];
let currentConversationId = null;
let autoSaveEnabled = true;

// Initialize conversation history on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeConversationHistory();
});

function initializeConversationHistory() {
    // Load saved conversations from localStorage
    loadConversationsFromStorage();
    
    // Set up event listeners
    const historyToggleBtn = document.getElementById('historyToggleBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const deleteConversationBtn = document.getElementById('deleteConversationBtn');
    const exportConversationBtn = document.getElementById('exportConversationBtn');
    const sidebarToggleBtn = document.getElementById('sidebarToggleBtn');
    
    if (historyToggleBtn) {
        historyToggleBtn.addEventListener('click', toggleHistoryPanel);
    }
    
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', clearAllHistory);
    }
    
    if (deleteConversationBtn) {
        deleteConversationBtn.addEventListener('click', deleteCurrentConversation);
    }
    
    if (exportConversationBtn) {
        exportConversationBtn.addEventListener('click', exportCurrentConversation);
    }
    
    if (sidebarToggleBtn) {
        sidebarToggleBtn.addEventListener('click', toggleHistorySidebar);
    }
    
    // Auto-save current conversation when new messages are added
    setupAutoSave();
    
    console.log('✅ Conversation history initialized');
}

function toggleHistoryPanel() {
    const historyContainer = document.getElementById('historyContainer');
    const chatContainer = document.getElementById('chatContainer');
    const mainContainer = document.getElementById('mainContainer');
    
    if (historyContainer && chatContainer && mainContainer) {
        const isHidden = historyContainer.classList.contains('hidden');
        
        if (isHidden) {
            // Show history panel
            historyContainer.classList.remove('hidden');
            chatContainer.style.display = 'flex';
            chatContainer.style.width = '50%';
            mainContainer.classList.add('split-view');
            
            // Refresh history list
            renderHistoryList();
        } else {
            // Hide history panel
            historyContainer.classList.add('hidden');
            chatContainer.style.width = '100%';
            mainContainer.classList.remove('split-view');
        }
    }
}

function toggleHistorySidebar() {
    const sidebar = document.querySelector('.notes-sidebar');
    if (sidebar) {
        sidebar.classList.toggle('collapsed');
    }
}

function loadConversationsFromStorage() {
    try {
        const stored = localStorage.getItem('research_conversations');
        if (stored) {
            savedConversations = JSON.parse(stored);
            console.log(`📚 Loaded ${savedConversations.length} conversations from storage`);
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        savedConversations = [];
    }
}

function saveConversationsToStorage() {
    try {
        localStorage.setItem('research_conversations', JSON.stringify(savedConversations));
        updateHistoryStats();
    } catch (error) {
        console.error('Error saving conversations:', error);
    }
}

function setupAutoSave() {
    // Monitor the chat messages div for changes
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Use MutationObserver to detect new messages
    const observer = new MutationObserver((mutations) => {
        if (autoSaveEnabled && conversationHistory && conversationHistory.length > 0) {
            // Debounce the save operation
            clearTimeout(window.autoSaveTimeout);
            window.autoSaveTimeout = setTimeout(() => {
                saveCurrentConversation();
            }, 2000); // Save 2 seconds after last message
        }
    });
    
    observer.observe(chatMessages, { childList: true, subtree: true });
}

function saveCurrentConversation() {
    if (!conversationHistory || conversationHistory.length === 0) return;
    
    const conversation = {
        id: currentConversationId || generateConversationId(),
        timestamp: Date.now(),
        messages: [...conversationHistory],
        preview: generatePreview(conversationHistory),
        messageCount: conversationHistory.length
    };
    
    // Update existing or add new
    const existingIndex = savedConversations.findIndex(c => c.id === conversation.id);
    if (existingIndex >= 0) {
        savedConversations[existingIndex] = conversation;
    } else {
        savedConversations.unshift(conversation); // Add to beginning
        currentConversationId = conversation.id;
    }
    
    // Keep only last 50 conversations
    if (savedConversations.length > 50) {
        savedConversations = savedConversations.slice(0, 50);
    }
    
    saveConversationsToStorage();
    renderHistoryList();
}

function generateConversationId() {
    return 'conv_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function generatePreview(messages) {
    if (!messages || messages.length === 0) return 'Empty conversation';
    
    // Get first user message as preview
    const firstUserMsg = messages.find(m => m.role === 'user');
    if (firstUserMsg) {
        const text = firstUserMsg.content.substring(0, 60);
        return text + (firstUserMsg.content.length > 60 ? '...' : '');
    }
    
    return messages[0].content.substring(0, 60) + '...';
}

function renderHistoryList() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    if (savedConversations.length === 0) {
        historyList.innerHTML = `
            <div class="notes-empty-state">
                <i class="fas fa-comments"></i>
                <p>No conversations yet</p>
                <p style="font-size: 0.85rem;">Start chatting to save your first conversation!</p>
            </div>
        `;
        return;
    }
    
    historyList.innerHTML = '';
    
    savedConversations.forEach((conv, index) => {
        const convDiv = document.createElement('div');
        convDiv.className = 'note-item';
        if (currentConversationId === conv.id) {
            convDiv.classList.add('active');
        }
        
        const date = new Date(conv.timestamp);
        const timeAgo = getTimeAgo(date);
        
        convDiv.innerHTML = `
            <div class="note-header">
                <i class="fas fa-comment"></i>
                <div class="note-info">
                    <div class="note-title">${escapeHtml(conv.preview)}</div>
                    <div class="note-meta">
                        ${conv.messageCount} messages • ${timeAgo}
                    </div>
                </div>
            </div>
        `;
        
        convDiv.addEventListener('click', () => viewConversation(conv.id));
        
        historyList.appendChild(convDiv);
    });
    
    updateHistoryStats();
}

function viewConversation(conversationId) {
    const conversation = savedConversations.find(c => c.id === conversationId);
    if (!conversation) return;
    
    const historyContent = document.getElementById('historyContent');
    const noHistorySelected = document.getElementById('noHistorySelected');
    const historyTitle = document.getElementById('historyTitle');
    const deleteBtn = document.getElementById('deleteConversationBtn');
    const exportBtn = document.getElementById('exportConversationBtn');
    
    if (!historyContent) return;
    
    // Update UI
    if (noHistorySelected) noHistorySelected.style.display = 'none';
    historyContent.style.display = 'block';
    
    if (historyTitle) {
        const date = new Date(conversation.timestamp);
        historyTitle.textContent = `Conversation from ${date.toLocaleString()}`;
    }
    
    if (deleteBtn) deleteBtn.disabled = false;
    if (exportBtn) exportBtn.disabled = false;
    
    // Render messages
    historyContent.innerHTML = '';
    conversation.messages.forEach(msg => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `history-message ${msg.role}`;
        msgDiv.style.cssText = `
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 8px;
            background: ${msg.role === 'user' ? 'var(--accent)' : 'var(--surface)'};
            color: ${msg.role === 'user' ? '#fff' : 'var(--text-primary)'};
        `;
        
        msgDiv.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 4px; font-size: 0.85rem; opacity: 0.8;">
                ${msg.role === 'user' ? '👤 You' : '🤖 Assistant'}
            </div>
            <div>${formatMessageContent(msg.content)}</div>
        `;
        
        historyContent.appendChild(msgDiv);
    });
    
    currentConversationId = conversationId;
    
    // Update active state in list
    document.querySelectorAll('.note-item').forEach(item => item.classList.remove('active'));
    event.target.closest('.note-item')?.classList.add('active');
}

function formatMessageContent(content) {
    // Basic markdown-like formatting
    let formatted = escapeHtml(content);
    
    // Convert markdown code blocks
    formatted = formatted.replace(/```([^`]+)```/g, '<pre style="background: var(--bg-tertiary); padding: 8px; border-radius: 4px; overflow-x: auto; margin: 8px 0;"><code>$1</code></pre>');
    
    // Convert inline code
    formatted = formatted.replace(/`([^`]+)`/g, '<code style="background: var(--bg-tertiary); padding: 2px 6px; border-radius: 3px;">$1</code>');
    
    // Convert bold
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    
    // Convert line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

function deleteCurrentConversation() {
    if (!currentConversationId) return;
    
    if (confirm('Delete this conversation? This cannot be undone.')) {
        savedConversations = savedConversations.filter(c => c.id !== currentConversationId);
        saveConversationsToStorage();
        renderHistoryList();
        
        // Clear viewer
        const historyContent = document.getElementById('historyContent');
        const noHistorySelected = document.getElementById('noHistorySelected');
        if (historyContent) historyContent.style.display = 'none';
        if (noHistorySelected) noHistorySelected.style.display = 'flex';
        
        currentConversationId = null;
        
        const deleteBtn = document.getElementById('deleteConversationBtn');
        const exportBtn = document.getElementById('exportConversationBtn');
        if (deleteBtn) deleteBtn.disabled = true;
        if (exportBtn) exportBtn.disabled = true;
    }
}

function exportCurrentConversation() {
    if (!currentConversationId) return;
    
    const conversation = savedConversations.find(c => c.id === currentConversationId);
    if (!conversation) return;
    
    let text = `Conversation Export\n`;
    text += `Date: ${new Date(conversation.timestamp).toLocaleString()}\n`;
    text += `Messages: ${conversation.messageCount}\n`;
    text += `\n${'='.repeat(60)}\n\n`;
    
    conversation.messages.forEach((msg, index) => {
        text += `${msg.role === 'user' ? 'YOU' : 'ASSISTANT'}:\n`;
        text += `${msg.content}\n\n`;
        text += `${'-'.repeat(60)}\n\n`;
    });
    
    // Create download
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${new Date(conversation.timestamp).toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

function clearAllHistory() {
    if (confirm('Delete ALL conversation history? This cannot be undone.')) {
        savedConversations = [];
        currentConversationId = null;
        saveConversationsToStorage();
        renderHistoryList();
        
        // Clear viewer
        const historyContent = document.getElementById('historyContent');
        const noHistorySelected = document.getElementById('noHistorySelected');
        if (historyContent) historyContent.style.display = 'none';
        if (noHistorySelected) noHistorySelected.style.display = 'flex';
        
        const deleteBtn = document.getElementById('deleteConversationBtn');
        const exportBtn = document.getElementById('exportConversationBtn');
        if (deleteBtn) deleteBtn.disabled = true;
        if (exportBtn) exportBtn.disabled = true;
    }
}

function updateHistoryStats() {
    const statsDiv = document.getElementById('historyStats');
    if (statsDiv) {
        const count = savedConversations.length;
        const totalMessages = savedConversations.reduce((sum, c) => sum + c.messageCount, 0);
        statsDiv.textContent = `${count} conversation${count !== 1 ? 's' : ''} • ${totalMessages} messages`;
    }
}

function getTimeAgo(date) {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    
    return date.toLocaleDateString();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Hook into the existing newChat function to save before clearing
const originalNewChat = window.newChat;
if (originalNewChat) {
    window.newChat = function() {
        // Save current conversation before starting new one
        if (conversationHistory && conversationHistory.length > 0) {
            saveCurrentConversation();
        }
        
        // Reset for new conversation
        currentConversationId = null;
        
        // Call original newChat
        originalNewChat();
    };
}

console.log('📝 Conversation history module loaded');



