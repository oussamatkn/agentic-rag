import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Lightbulb, Search, CheckSquare, RotateCw } from 'lucide-react';

const AgentTransparency = ({ isVisible = true }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    const agentSteps = [
        {
            phase: 'Planning',
            icon: <Lightbulb size={20} />,
            description: 'Question decomposition into sub-queries',
            details: 'The agent analyzes the legal question and breaks it down into focused sub-questions for targeted retrieval.'
        },
        {
            phase: 'Retrieval',
            icon: <Search size={20} />,
            description: 'Semantic search across legal documents',
            details: 'Vector embeddings are used to find the most relevant legal document chunks based on semantic similarity.'
        },
        {
            phase: 'Evaluation',
            icon: <CheckSquare size={20} />,
            description: 'Sufficiency assessment of retrieved information',
            details: 'The agent evaluates whether the retrieved information is sufficient to answer the question comprehensively.'
        },
        {
            phase: 'Iteration',
            icon: <RotateCw size={20} />,
            description: 'Adaptive refinement (up to 3 iterations)',
            details: 'If information is insufficient, the agent refines its search strategy and retrieves additional relevant content.'
        }
    ];

    if (!isVisible) return null;

    return (
        <div className="card transparency-card">
            <div
                className="transparency-header"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <h3 className="card-title">
                    <Lightbulb size={24} />
                    Agentic Process Transparency
                </h3>
                <button className="expand-btn">
                    {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                </button>
            </div>

            {isExpanded && (
                <div className="transparency-content">
                    <p className="transparency-intro">
                        This system uses an <strong>Agentic RAG (Retrieval-Augmented Generation)</strong> approach
                        with autonomous planning, retrieval, and evaluation cycles.
                    </p>

                    <div className="agent-steps">
                        {agentSteps.map((step, index) => (
                            <div key={index} className="agent-step">
                                <div className="step-header">
                                    <div className="step-icon">{step.icon}</div>
                                    <div className="step-info">
                                        <h4>{step.phase}</h4>
                                        <p className="step-description">{step.description}</p>
                                    </div>
                                </div>
                                <p className="step-details">{step.details}</p>
                            </div>
                        ))}
                    </div>

                    <div className="transparency-note">
                        <strong>Note:</strong> The agent autonomously determines when sufficient information
                        has been gathered, typically completing within 1-3 iterations.
                    </div>
                </div>
            )}
        </div>
    );
};

export default AgentTransparency;
