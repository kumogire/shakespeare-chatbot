/**
 * Shakespeare Chatbot - Frontend JavaScript
 * Handles chat interface, API communication, and user interactions
 */

class ShakespeareChatbot {
    constructor() {
        this.isConnected = false;
        this.isTyping = false;
        this.settings = {
            temperature: 0.8,
            maxLength: 100
        };
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkConnection();
        this.loadSettings();
        this.setWelcomeTime();
    }

    initializeElements() {
        // Chat elements
        this.messagesContainer = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.charCount = document.getElementById('charCount');

        // Status elements
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.statusDot = this.statusIndicator.querySelector('.status-dot');

        // Settings elements
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsPanel = document.getElementById('settingsPanel');
        this.temperatureSlider = document.getElementById('temperatureSlider');
        this.temperatureValue = document.getElementById('temperatureValue');
        this.maxLengthSlider = document.getElementById('maxLengthSlider');
        this.maxLengthValue = document.getElementById('maxLengthValue');
        this.resetSettings = document.getElementById('resetSettings');
        this.saveSettings = document.getElementById('saveSettings');

        // Action buttons
        this.sampleBtn = document.getElementById('sampleBtn');
        this.clearBtn = document.getElementById('clearBtn');

        // Modal elements
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        this.modalClose = document.getElementById('modalClose');
        this.modalOk = document.getElementById('modalOk');

        // Info panel
        this.infoToggle = document.getElementById('infoToggle');
        this.infoContent = document.getElementById('infoContent');

        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    setupEventListeners() {
        // Chat input events
        this.messageInput.addEventListener('input', () => this.handleInput());
        this.messageInput.addEventListener('keypress', (e) => this.handleKeyPress(e));
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Settings events
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        this.temperatureSlider.addEventListener('input', () => this.updateTemperature());
        this.maxLengthSlider.addEventListener('input', () => this.updateMaxLength());
        this.resetSettings.addEventListener('click', () => this.resetSettingsToDefault());
        this.saveSettings.addEventListener('click', () => this.saveSettingsToServer());

        // Action button events
        this.sampleBtn.addEventListener('click', () => this.generateSample());
        this.clearBtn.addEventListener('click', () => this.clearConversation());

        // Modal events
        this.modalClose.addEventListener('click', () => this.hideModal());
        this.modalOk.addEventListener('click', () => this.hideModal());

        // Info panel events
        this.infoToggle.addEventListener('click', () => this.toggleInfo());

        // Click outside to close panels
        document.addEventListener('click', (e) => this.handleOutsideClick(e));

        // Prevent form submission
        document.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.target !== this.messageInput) {
                e.preventDefault();
            }
        });
    }

    handleInput() {
        const value = this.messageInput.value;
        const length = value.length;
        
        // Update character count
        this.charCount.textContent = length;
        
        // Update send button state
        this.sendButton.disabled = length === 0 || this.isTyping;
        
        // Auto-resize input (optional)
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    handleKeyPress(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendButton.disabled) {
                this.sendMessage();
            }
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;

        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        this.handleInput();

        // Show typing indicator
        this.showTyping();

        try {
            const response = await this.callChatAPI(message);
            this.hideTyping();
            
            if (response.response) {
                this.addMessage(response.response, 'bot');
            } else {
                throw new Error('No response received');
            }
        } catch (error) {
            this.hideTyping();
            console.error('Chat error:', error);
            this.showError('Failed to get response from Shakespeare. Please try again.');
            this.addMessage('*The Bard seems to have lost his words momentarily. Pray, try again.*', 'bot', true);
        }
    }

    async callChatAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                temperature: this.settings.temperature,
                max_length: this.settings.maxLength
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    addMessage(text, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        if (isError) {
            messageDiv.classList.add('error-message');
        }

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-feather-alt"></i>';

        const content = document.createElement('div');
        content.className = 'message-content';

        const textDiv = document.createElement('div');
        textDiv.className = 'message-text';
        
        // Format text with proper paragraphs
        const paragraphs = text.split('\n').filter(p => p.trim());
        paragraphs.forEach(paragraph => {
            const p = document.createElement('p');
            p.textContent = paragraph;
            textDiv.appendChild(p);
        });

        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = this.getCurrentTime();

        content.appendChild(textDiv);
        content.appendChild(timeDiv);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTyping() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.sendButton.disabled = true;
        this.scrollToBottom();
    }

    hideTyping() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
        this.sendButton.disabled = this.messageInput.value.trim() === '';
    }

    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    setWelcomeTime() {
        const welcomeTime = document.getElementById('welcomeTime');
        if (welcomeTime) {
            welcomeTime.textContent = this.getCurrentTime();
        }
    }

    // Settings Management
    toggleSettings() {
        this.settingsPanel.classList.toggle('show');
    }

    updateTemperature() {
        const value = parseFloat(this.temperatureSlider.value);
        this.temperatureValue.textContent = value.toFixed(1);
        this.settings.temperature = value;
    }

    updateMaxLength() {
        const value = parseInt(this.maxLengthSlider.value);
        this.maxLengthValue.textContent = value;
        this.settings.maxLength = value;
    }

    resetSettingsToDefault() {
        this.settings = {
            temperature: 0.8,
            maxLength: 100
        };
        this.updateSettingsUI();
        this.saveSettings();
    }

    updateSettingsUI() {
        this.temperatureSlider.value = this.settings.temperature;
        this.temperatureValue.textContent = this.settings.temperature.toFixed(1);
        this.maxLengthSlider.value = this.settings.maxLength;
        this.maxLengthValue.textContent = this.settings.maxLength;
    }

    saveSettings() {
        localStorage.setItem('shakespeareSettings', JSON.stringify(this.settings));
    }

    loadSettings() {
        const saved = localStorage.getItem('shakespeareSettings');
        if (saved) {
            try {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
                this.updateSettingsUI();
            } catch (error) {
                console.warn('Failed to load settings:', error);
            }
        }
    }

    async saveSettingsToServer() {
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(this.settings)
            });

            if (response.ok) {
                this.saveSettings();
                this.showNotification('Settings saved successfully!');
                this.settingsPanel.classList.remove('show');
            } else {
                throw new Error('Failed to save settings to server');
            }
        } catch (error) {
            console.error('Settings save error:', error);
            this.saveSettings(); // Save locally anyway
            this.showNotification('Settings saved locally');
        }
    }

    // Sample Generation
    async generateSample() {
        if (this.isTyping) return;

        this.showTyping();
        
        try {
            const response = await fetch('/api/sample');
            this.hideTyping();
            
            if (response.ok) {
                const data = await response.json();
                this.addMessage(data.sample, 'bot');
            } else {
                throw new Error('Failed to generate sample');
            }
        } catch (error) {
            this.hideTyping();
            console.error('Sample generation error:', error);
            this.showError('Failed to generate sample text. Please try again.');
        }
    }

    // Clear Conversation
    clearConversation() {
        // Keep only the welcome message
        const welcomeMessage = this.messagesContainer.querySelector('.welcome-message').parentElement;
        this.messagesContainer.innerHTML = '';
        this.messagesContainer.appendChild(welcomeMessage);
    }

    // Connection Status
    async checkConnection() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                this.updateConnectionStatus(data.model_loaded, data);
            } else {
                this.updateConnectionStatus(false);
            }
        } catch (error) {
            console.error('Connection check failed:', error);
            this.updateConnectionStatus(false);
        }

        // Check again in 30 seconds
        setTimeout(() => this.checkConnection(), 30000);
    }

    updateConnectionStatus(connected, statusData = null) {
        this.isConnected = connected;
        
        if (connected) {
            this.statusDot.className = 'status-dot connected';
            this.statusText.textContent = 'Connected';
            this.loadingOverlay.classList.remove('show');
        } else {
            this.statusDot.className = 'status-dot error';
            this.statusText.textContent = 'Disconnected';
        }

        // Update tooltip with additional info
        if (statusData) {
            const device = statusData.cuda_available ? 'GPU' : 'CPU';
            this.statusIndicator.title = `Running on ${device} - ${statusData.torch_version}`;
        }
    }

    // Info Panel
    toggleInfo() {
        this.infoContent.classList.toggle('show');
    }

    // Outside Click Handler
    handleOutsideClick(e) {
        // Close settings panel
        if (!this.settingsPanel.contains(e.target) && !this.settingsBtn.contains(e.target)) {
            this.settingsPanel.classList.remove('show');
        }

        // Close info panel
        if (!this.infoContent.contains(e.target) && !this.infoToggle.contains(e.target)) {
            this.infoContent.classList.remove('show');
        }
    }

    // Error Handling
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.classList.add('show');
    }

    hideModal() {
        this.errorModal.classList.remove('show');
    }

    // Notifications
    showNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '2rem',
            right: '2rem',
            background: type === 'success' ? '#4CAF50' : '#f44336',
            color: 'white',
            padding: '1rem 1.5rem',
            borderRadius: '12px',
            boxShadow: '0 5px 20px rgba(0,0,0,0.3)',
            zIndex: '2000',
            transform: 'translateX(100%)',
            transition: 'transform 0.3s ease-out'
        });

        document.body.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Remove after delay
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    // Utility Methods
    formatText(text) {
        // Add basic text formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    // Keyboard shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to send message
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (!this.sendButton.disabled) {
                    this.sendMessage();
                }
            }
            
            // Escape to close panels
            if (e.key === 'Escape') {
                this.settingsPanel.classList.remove('show');
                this.infoContent.classList.remove('show');
                this.hideModal();
            }
            
            // Ctrl/Cmd + K to clear chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.clearConversation();
            }
        });
    }
}

// Initialize the chatbot when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.shakespeareBot = new ShakespeareChatbot();
    
    // Add some easter eggs
    console.log('🎭 "All the world\'s a stage, and all the men and women merely players" - Shakespeare (via nanoGPT)');
    console.log('💡 Tip: Try Ctrl+K to clear the chat, or Ctrl+Enter to send a message!');
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.shakespeareBot) {
        window.shakespeareBot.checkConnection();
    }
});

// Handle errors gracefully
window.addEventListener('error', (e) => {
    console.error('JavaScript Error:', e.error);
    if (window.shakespeareBot) {
        window.shakespeareBot.showError('An unexpected error occurred. Please refresh the page.');
    }
});

// Service worker registration (optional, for PWA features)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        // Uncomment if you add a service worker
        // navigator.serviceWorker.register('/sw.js');
    });
}