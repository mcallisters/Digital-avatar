import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const messagesEndRef = useRef(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();
      
      // Update session ID if new
      if (!sessionId) {
        setSessionId(data.session_id);
      }

      // Add assistant message to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.response,
        category: data.category 
      }]);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { 
        role: 'error', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="app">
      <div className="chat-container">
        {/* Header */}
        <div className="header">
          <h1>Hi, I am Sean's Digital Avatar, how can I help you today?</h1>
          <button 
            className="info-button"
            onClick={() => setShowInfo(!showInfo)}
          >
            <span className="info-icon">ⓘ</span> What can you do?
          </button>
        </div>

        {/* Info Panel */}
        {showInfo && (
          <div className="info-panel">
            <h3>I can help you with:</h3>
            <ul>
              <li><strong>General:</strong> Questions about Sean's work experience and background</li>
              <li><strong>Projects:</strong> Information about Sean's personal and community projects</li>
              <li><strong>Publications:</strong> Details about Sean's research papers and publications</li>
              <li><strong>Calendar:</strong> Scheduling meetings and checking availability</li>
              <li><strong>Hobbies:</strong> Sean's interests outside of professional work</li>
            </ul>
          </div>
        )}

        {/* Messages */}
        <div className="messages">
          {messages.length === 0 && !showInfo && (
            <div className="welcome-message">
              <p>Ask me anything about Sean's work, projects, publications, or interests!</p>
            </div>
          )}
          
          {messages.map((msg, index) => (
            <div 
              key={index} 
              className={`message ${msg.role}`}
            >
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="message assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="input-container">
          <div className="input-wrapper">
            <button 
              className="add-button"
              title="Add attachment (coming soon)"
            >
              +
            </button>
            <input
              type="text"
              placeholder="Ask anything..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={loading}
            />
            <button 
              className="send-button"
              onClick={sendMessage}
              disabled={!input.trim() || loading}
            >
              ↑
            </button>
          </div>
        </div>

        {/* Clear Chat Button */}
        {messages.length > 0 && (
          <button className="clear-button" onClick={clearChat}>
            Clear Chat
          </button>
        )}
      </div>
    </div>
  );
}

export default App;
