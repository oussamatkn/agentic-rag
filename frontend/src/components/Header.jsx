import React from 'react';
import { Scale } from 'lucide-react';

const Header = () => {
    return (
        <header className="header">
            <div className="header-content">
                <div className="header-icon">
                    <Scale size={48} />
                </div>
                <div className="header-text">
                    <h1>Agentic AI Legal Assistant</h1>
                    <p className="subtitle">Educational Legal Research Tool</p>
                </div>
            </div>
        </header>
    );
};

export default Header;
