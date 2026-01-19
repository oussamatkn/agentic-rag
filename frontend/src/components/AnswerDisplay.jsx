import React from 'react';
import { FileText, BookOpen, AlertTriangle, TrendingUp, Clock } from 'lucide-react';

const AnswerDisplay = ({ answer }) => {
    if (!answer) return null;

    return (
        <div className="answer-container">
            <h2 className="section-title">Legal Analysis Results</h2>

            {/* Summary Card */}
            <div className="card answer-card">
                <h3 className="answer-section-title">
                    <FileText size={20} />
                    Summary
                </h3>
                <p className="answer-text">{answer.summary}</p>
            </div>

            {/* Legal Reasoning Card */}
            <div className="card answer-card">
                <h3 className="answer-section-title">
                    <BookOpen size={20} />
                    Legal Reasoning
                </h3>
                <p className="answer-text">{answer.legal_reasoning}</p>
            </div>

            {/* Cited Sources Card */}
            <div className="card answer-card">
                <h3 className="answer-section-title">
                    <FileText size={20} />
                    Cited Sources
                </h3>
                <div className="sources-list">
                    {answer.cited_sources && answer.cited_sources.length > 0 ? (
                        answer.cited_sources.map((source, index) => (
                            <div key={index} className="source-item">
                                <div className="source-header">
                                    <span className="source-id">{source.document_id}</span>
                                    <span className="source-score">
                                        Relevance: {(source.relevance_score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                {source.metadata && (
                                    <div className="source-metadata">
                                        {source.metadata.filename && (
                                            <span>File: {source.metadata.filename}</span>
                                        )}
                                        {source.metadata.chunk_index !== undefined && (
                                            <span>Chunk: {source.metadata.chunk_index + 1}</span>
                                        )}
                                    </div>
                                )}
                            </div>
                        ))
                    ) : (
                        <p className="no-sources">No sources cited</p>
                    )}
                </div>
            </div>

            {/* Metadata Card */}
            <div className="metadata-grid">
                <div className="metadata-item">
                    <TrendingUp size={18} />
                    <div>
                        <span className="metadata-label">Confidence Score</span>
                        <span className="metadata-value">
                            {(answer.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>

                <div className="metadata-item">
                    <Clock size={18} />
                    <div>
                        <span className="metadata-label">Timestamp</span>
                        <span className="metadata-value">
                            {new Date(answer.timestamp).toLocaleString()}
                        </span>
                    </div>
                </div>
            </div>

            {/* Legal Disclaimer */}
            <div className="disclaimer-card">
                <div className="disclaimer-header">
                    <AlertTriangle size={20} />
                    <strong>Legal Disclaimer</strong>
                </div>
                <p>{answer.disclaimer}</p>
            </div>
        </div>
    );
};

export default AnswerDisplay;
