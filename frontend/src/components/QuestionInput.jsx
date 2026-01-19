import React, { useState } from 'react';
import { MessageSquare, Send } from 'lucide-react';

const QuestionInput = ({ onQuestionSubmit, loading }) => {
    const [question, setQuestion] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (question.trim() && !loading) {
            onQuestionSubmit(question);
        }
    };

    return (
        <div className="card">
            <h2 className="card-title">
                <MessageSquare size={24} />
                Ask Legal Question
            </h2>

            <form onSubmit={handleSubmit} className="question-form">
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Enter your legal question here... (e.g., What are the requirements for forming a contract?)"
                    className="question-textarea"
                    rows={4}
                    disabled={loading}
                />

                <button
                    type="submit"
                    disabled={!question.trim() || loading}
                    className="btn btn-primary"
                >
                    <Send size={20} />
                    {loading ? 'Processing...' : 'Ask Legal Question'}
                </button>
            </form>

            {loading && (
                <div className="loading-indicator">
                    <div className="spinner"></div>
                    <p>Agent is analyzing your question...</p>
                </div>
            )}
        </div>
    );
};

export default QuestionInput;
