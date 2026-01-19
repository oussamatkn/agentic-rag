import React, { useState } from 'react';
import Header from './components/Header';
import DocumentUpload from './components/DocumentUpload';
import QuestionInput from './components/QuestionInput';
import AnswerDisplay from './components/AnswerDisplay';
import AgentTransparency from './components/AgentTransparency';
import { askQuestion } from './services/api';

function App() {
    const [answer, setAnswer] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleQuestionSubmit = async (question) => {
        setLoading(true);
        setError(null);
        setAnswer(null);

        try {
            const response = await askQuestion(question);
            setAnswer(response);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to process question. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleUploadSuccess = () => {
        // Optional: Could show a notification or update UI state
        console.log('Document uploaded successfully');
    };

    return (
        <div className="app">
            <Header />

            <main className="main-content">
                <div className="container">
                    {/* Agent Transparency Panel */}
                    <AgentTransparency isVisible={true} />

                    {/* Document Upload Section */}
                    <DocumentUpload onUploadSuccess={handleUploadSuccess} />

                    {/* Question Input Section */}
                    <QuestionInput
                        onQuestionSubmit={handleQuestionSubmit}
                        loading={loading}
                    />

                    {/* Error Display */}
                    {error && (
                        <div className="error-banner">
                            <p>{error}</p>
                        </div>
                    )}

                    {/* Answer Display Section */}
                    {answer && <AnswerDisplay answer={answer} />}
                </div>
            </main>

            <footer className="footer">
                <p>© Agentic AI | Legal Assistant</p>
            </footer>
        </div>
    );
}

export default App;
