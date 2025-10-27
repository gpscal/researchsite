// VoIP-LLM PCAP Analyzer JavaScript
class VoIPPCAPAnalyzer {
    constructor() {
        this.currentPcapId = null;
        this.currentModel = 'ollama';
        this.isProcessing = false;
        this.chatHistory = [];
        
        this.initializeElements();
        this.bindEvents();
        this.loadModels();
    }

    initializeElements() {
        // Header elements
        this.modelDropdown = document.getElementById('modelDropdown');
        this.newChatBtn = document.getElementById('newChatBtn');
        
        // Chat elements
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatInput = document.getElementById('chatInput');
        this.sendBtn = document.getElementById('sendBtn');
        
        // PCAP attachment elements
        this.attachPcapBtn = document.getElementById('attachPcapBtn');
        this.pcapFileInput = document.getElementById('pcapFileInput');
        this.attachedPcap = document.getElementById('attachedPcap');
        this.attachedPcapName = document.getElementById('attachedPcapName');
        this.attachedPcapStats = document.getElementById('attachedPcapStats');
        this.removePcapBtn = document.getElementById('removePcapBtn');
        
        // Chat options
        this.useVoIPTraining = document.getElementById('useVoIPTraining');
        this.useWebSearch = document.getElementById('useWebSearch');
        this.useConversationHistory = document.getElementById('useConversationHistory');
        this.topK = document.getElementById('topK');
        
        // Loading indicator
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.loadingText = document.getElementById('loadingText');
    }

    bindEvents() {
        // Model selection
        this.modelDropdown.addEventListener('change', () => {
            this.currentModel = this.modelDropdown.value;
            console.log('Model switched to:', this.currentModel);
        });

        // New chat
        this.newChatBtn.addEventListener('click', () => this.startNewChat());

        // PCAP attachment
        this.attachPcapBtn.addEventListener('click', () => this.pcapFileInput.click());
        this.pcapFileInput.addEventListener('change', (e) => this.handlePcapUpload(e));
        this.removePcapBtn.addEventListener('click', () => this.removePcapAttachment());

        // Chat input
        this.chatInput.addEventListener('input', () => {
            this.updateSendButton();
            this.autoResizeTextarea();
        });
        this.chatInput.addEventListener('keydown', (e) => this.handleKeyPress(e));
        
        // Send button
        this.sendBtn.addEventListener('click', () => this.sendMessage());
    }

    async loadModels() {
        try {
            const response = await fetch('/api/pcap/models');
            const data = await response.json();
            
            if (data.success) {
                // Update dropdown with available models
                this.modelDropdown.innerHTML = '';
                data.models.forEach(model => {
                    if (model.available) {
                        const option = document.createElement('option');
                        option.value = model.id;
                        option.textContent = model.name;
                        this.modelDropdown.appendChild(option);
                    }
                });
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }

    async handlePcapUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            this.showLoading('Uploading and parsing PCAP file...');

            const response = await fetch('/api/pcap/upload-attach', {
                method: 'POST',
                body: formData
            });

            // Check if response is OK before parsing JSON
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} - ${errorText.substring(0, 100)}`);
            }

            const result = await response.json();
            
            this.hideLoading();

            if (result.success) {
                this.currentPcapId = result.file_id;
                this.showPcapAttachment(file.name, result.metadata, result.summary);
                
                // Show welcome message about PCAP if chat is empty
                if (this.chatHistory.length === 0) {
                    this.hideWelcomeScreen();
                    this.addMessage(`I've analyzed your PCAP file "${file.name}". Here's a summary:\n\n${result.summary}\n\nFeel free to ask me questions about this capture!`, 'assistant');
                }
            } else {
                this.showError(result.error || 'Failed to upload PCAP file');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error uploading PCAP file: ' + error.message);
        }

        // Reset file input
        this.pcapFileInput.value = '';
    }

    showPcapAttachment(filename, metadata, summary) {
        this.attachedPcapName.textContent = filename;
        this.attachedPcapStats.textContent = `${metadata.packet_count} packets, ${metadata.protocols.join(', ')}`;
        this.attachedPcap.style.display = 'block';
    }

    removePcapAttachment() {
        this.currentPcapId = null;
        this.attachedPcap.style.display = 'none';
        this.attachedPcapName.textContent = '';
        this.attachedPcapStats.textContent = '';
    }

    startNewChat() {
        this.chatHistory = [];
        this.chatMessages.innerHTML = '';
        this.chatMessages.style.display = 'none';
        this.welcomeScreen.style.display = 'flex';
        this.currentPcapId = null;
        this.attachedPcap.style.display = 'none';
    }

    hideWelcomeScreen() {
        this.welcomeScreen.style.display = 'none';
        this.chatMessages.style.display = 'flex';
    }

    updateSendButton() {
        const hasText = this.chatInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText || this.isProcessing;
    }

    autoResizeTextarea() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
    }

    handleKeyPress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isProcessing) return;

        // Hide welcome screen if visible
        if (this.welcomeScreen.style.display !== 'none') {
            this.hideWelcomeScreen();
        }

        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.updateSendButton();
        this.autoResizeTextarea();

        // Show typing indicator
        const typingElement = this.showTypingIndicator();
        this.isProcessing = true;

        try {
            // Get chat options
            const useTraining = this.useVoIPTraining.checked;
            const useWeb = this.useWebSearch.checked;
            const useHistory = this.useConversationHistory.checked;
            const topK = parseInt(this.topK.value);

            // Send query
            const response = await fetch('/api/pcap/voip-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: message,
                    pcap_id: this.currentPcapId,
                    use_training_data: useTraining,
                    use_web: useWeb,
                    use_conversation_history: useHistory,
                    top_k: topK,
                    model_type: this.currentModel,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Remove typing indicator
            this.removeTypingIndicator(typingElement);

            // Handle streaming response
            await this.handleStreamingResponse(response);

            // Update chat history
            this.chatHistory.push({
                user: message,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            console.error('Error sending message:', error);
            this.removeTypingIndicator(typingElement);
            this.addMessage('Sorry, I encountered an error processing your request. Please try again.', 'assistant');
        } finally {
            this.isProcessing = false;
            this.updateSendButton();
        }
    }

    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentMessageElement = null;
        let fullResponse = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') break;

                    try {
                        const parsed = JSON.parse(data);

                        if (parsed.content) {
                            if (!currentMessageElement) {
                                currentMessageElement = this.addMessage('', 'assistant', true);
                            }
                            fullResponse += parsed.content;
                            this.updateMessageContent(currentMessageElement, fullResponse);
                        } else if (parsed.pending_commands) {
                            // Handle command suggestions from AI
                            this.renderCommandCards(parsed.pending_commands);
                        } else if (parsed.done) {
                            break;
                        } else if (parsed.error) {
                            this.showError(parsed.error);
                            break;
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e);
                    }
                }
            }
        }
    }

    addMessage(content, type, isStreaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        if (content) {
            bubble.innerHTML = this.formatMessage(content);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        return isStreaming ? bubble : messageDiv;
    }

    updateMessageContent(element, content) {
        element.innerHTML = this.formatMessage(content);
        this.scrollToBottom();
    }

    formatMessage(content) {
        // Basic markdown-like formatting
        let formatted = content
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');

        return formatted;
    }

    showTypingIndicator() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = `
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        `;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        return messageDiv;
    }

    removeTypingIndicator(element) {
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showLoading(text) {
        this.loadingText.textContent = text;
        this.loadingIndicator.style.display = 'flex';
    }

    hideLoading() {
        this.loadingIndicator.style.display = 'none';
    }

    showError(message) {
        // Create error toast notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-toast';
        errorDiv.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--error);
            color: white;
            padding: 16px 20px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
            max-width: 400px;
            display: flex;
            align-items: center;
            gap: 12px;
        `;
        
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(errorDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => errorDiv.remove(), 300);
            }
        }, 5000);
    }

    renderCommandCards(commands) {
        /**
         * Render command cards in the chat for user approval
         */
        if (!commands || commands.length === 0) return;

        commands.forEach(cmd => {
            const commandCard = document.createElement('div');
            commandCard.className = 'command-card';
            commandCard.dataset.commandId = cmd.id;

            commandCard.innerHTML = `
                <div class="command-card-header">
                    <i class="fas fa-terminal"></i>
                    <span>AI Suggests Running Analysis Command</span>
                </div>
                <div class="command-card-body">
                    <div class="command-reason">
                        <strong>Purpose:</strong> ${cmd.explanation}
                    </div>
                    <div class="command-display">
                        <code>${cmd.command}</code>
                    </div>
                    <div class="command-actions">
                        <button class="btn-execute" data-command-id="${cmd.id}">
                            <i class="fas fa-play"></i> Run Command
                        </button>
                        <button class="btn-skip" data-command-id="${cmd.id}">
                            <i class="fas fa-times"></i> Skip
                        </button>
                    </div>
                    <div class="command-status" style="display: none;"></div>
                </div>
            `;

            this.chatMessages.appendChild(commandCard);
            this.scrollToBottom();

            // Bind events
            const executeBtn = commandCard.querySelector('.btn-execute');
            const skipBtn = commandCard.querySelector('.btn-skip');

            executeBtn.addEventListener('click', () => this.executeCommand(cmd, commandCard));
            skipBtn.addEventListener('click', () => this.skipCommand(commandCard));
        });
    }

    async executeCommand(cmd, commandCard) {
        /**
         * Execute a user-approved command
         */
        const statusDiv = commandCard.querySelector('.command-status');
        const actionsDiv = commandCard.querySelector('.command-actions');

        try {
            // Update UI to show executing state
            actionsDiv.style.display = 'none';
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = `
                <i class="fas fa-spinner fa-spin"></i>
                Executing command...
            `;
            statusDiv.className = 'command-status executing';

            // Execute command via API
            const response = await fetch('/api/pcap/execute-command', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    command: cmd.command,
                    pcap_id: cmd.pcap_id,
                    approved: true
                })
            });

            const result = await response.json();

            if (result.success) {
                // Show success
                statusDiv.innerHTML = `
                    <i class="fas fa-check-circle"></i>
                    Command executed successfully (${result.execution_time.toFixed(2)}s)
                `;
                statusDiv.className = 'command-status success';

                // Add output to chat
                this.addCommandOutput(result.output, result.command);

                // Auto-send results back to AI
                setTimeout(() => {
                    const outputMessage = `Command executed successfully. Output:\n\`\`\`\n${result.output}\n\`\`\``;
                    this.sendMessageWithText(outputMessage);
                }, 500);

            } else {
                // Show error
                statusDiv.innerHTML = `
                    <i class="fas fa-exclamation-triangle"></i>
                    Error: ${result.error}
                `;
                statusDiv.className = 'command-status error';

                // Restore actions
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                    actionsDiv.style.display = 'flex';
                }, 3000);
            }

        } catch (error) {
            console.error('Error executing command:', error);
            statusDiv.innerHTML = `
                <i class="fas fa-exclamation-triangle"></i>
                Failed to execute: ${error.message}
            `;
            statusDiv.className = 'command-status error';

            // Restore actions
            setTimeout(() => {
                statusDiv.style.display = 'none';
                actionsDiv.style.display = 'flex';
            }, 3000);
        }
    }

    skipCommand(commandCard) {
        /**
         * Skip/dismiss a command card
         */
        commandCard.style.opacity = '0.5';
        const statusDiv = commandCard.querySelector('.command-status');
        const actionsDiv = commandCard.querySelector('.command-actions');

        actionsDiv.style.display = 'none';
        statusDiv.style.display = 'block';
        statusDiv.innerHTML = `
            <i class="fas fa-times-circle"></i>
            Skipped
        `;
        statusDiv.className = 'command-status skipped';
    }

    addCommandOutput(output, command) {
        /**
         * Add command output as a special message in chat
         */
        const outputDiv = document.createElement('div');
        outputDiv.className = 'message command-output';

        outputDiv.innerHTML = `
            <div class="command-output-header">
                <i class="fas fa-terminal"></i>
                <span>Command Output</span>
                <button class="toggle-output">
                    <i class="fas fa-chevron-down"></i>
                </button>
            </div>
            <div class="command-output-command">
                <code>${command}</code>
            </div>
            <div class="command-output-content">
                <pre>${output}</pre>
            </div>
        `;

        this.chatMessages.appendChild(outputDiv);
        this.scrollToBottom();

        // Toggle expand/collapse
        const toggleBtn = outputDiv.querySelector('.toggle-output');
        const content = outputDiv.querySelector('.command-output-content');
        
        toggleBtn.addEventListener('click', () => {
            const isExpanded = content.style.display !== 'none';
            content.style.display = isExpanded ? 'none' : 'block';
            toggleBtn.querySelector('i').className = isExpanded ? 'fas fa-chevron-down' : 'fas fa-chevron-up';
        });
    }

    sendMessageWithText(text) {
        /**
         * Programmatically send a message (used for sending command results to AI)
         */
        this.chatInput.value = text;
        this.sendMessage();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VoIPPCAPAnalyzer();
    console.log('VoIP-LLM PCAP Analyzer initialized');
});

// Add additional styles for animations
const additionalStyles = document.createElement('style');
additionalStyles.textContent = `
@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes fadeOut {
    from {
        opacity: 1;
    }
    to {
        opacity: 0;
    }
}
`;
document.head.appendChild(additionalStyles);
