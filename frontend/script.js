// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;
// Bumped on every new chat so responses to queries from a previous chat are discarded
let requestGeneration = 0;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    
    setupEventListeners();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // New chat
    document.getElementById('newChatButton').addEventListener('click', newChat);

    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Remember which chat this request belongs to
    const generation = requestGeneration;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        if (!response.ok) throw new Error('Query failed');

        const data = await response.json();

        // The user started a new chat while this request was in flight:
        // drop the answer and delete the session it may have created
        if (generation !== requestGeneration) {
            if (data.session_id) deleteSessionOnServer(data.session_id);
            return;
        }

        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources);

    } catch (error) {
        if (generation !== requestGeneration) return;

        // Replace loading message with error
        loadingMessage.remove();
        addMessage(`Error: ${error.message}`, 'assistant');
    } finally {
        // Only the chat that is still on screen should re-enable the input
        if (generation === requestGeneration) {
            chatInput.disabled = false;
            sendButton.disabled = false;
            chatInput.focus();
        }
    }
}

// Start a fresh conversation without reloading the page
function newChat() {
    const oldSessionId = currentSessionId;

    // Invalidate any query still in flight for the previous chat
    requestGeneration++;

    // Free the old session on the server. The UI does not wait for this.
    if (oldSessionId) deleteSessionOnServer(oldSessionId);

    // Clear the chat window and show the welcome message (also resets currentSessionId)
    createNewSession();

    // Unlock input in case a query from the old chat was still pending
    chatInput.value = '';
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
}

// Best-effort deletion of a backend session; failures are only logged
function deleteSessionOnServer(sessionId) {
    fetch(`${API_URL}/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
        .catch(error => console.error('Failed to delete session:', error));
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    const displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    messageDiv.innerHTML = html;
    if (sources && sources.length > 0) {
        messageDiv.appendChild(renderSources(sources));
    }
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

// Build the collapsible Sources list as DOM nodes so labels and URLs are never parsed as HTML
function renderSources(sources) {
    const details = document.createElement('details');
    details.className = 'sources-collapsible';

    const summary = document.createElement('summary');
    summary.className = 'sources-header';
    summary.textContent = 'Sources';
    details.appendChild(summary);

    const content = document.createElement('div');
    content.className = 'sources-content';

    sources.forEach((source, index) => {
        if (index > 0) content.appendChild(document.createTextNode(', '));
        if (source.url) {
            const link = document.createElement('a');
            link.href = source.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = source.label;
            content.appendChild(link);
        } else {
            content.appendChild(document.createTextNode(source.label));
        }
    });

    details.appendChild(content);
    return details;
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    currentSessionId = null;
    chatMessages.innerHTML = '';
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, true);
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}