import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';

const DocumentUpload = ({ onUploadSuccess }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            const validTypes = ['application/pdf', 'text/plain'];
            if (validTypes.includes(file.type) || file.name.endsWith('.pdf') || file.name.endsWith('.txt')) {
                setSelectedFile(file);
                setError(null);
                setResult(null);
            } else {
                setError('Please select a PDF or TXT file');
                setSelectedFile(null);
            }
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select a file first');
            return;
        }

        setUploading(true);
        setError(null);
        setResult(null);

        try {
            const { ingestDocument } = await import('../services/api');
            const response = await ingestDocument(selectedFile);
            setResult(response);
            if (onUploadSuccess) {
                onUploadSuccess(response);
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Upload failed. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="card">
            <h2 className="card-title">
                <Upload size={24} />
                Document Upload
            </h2>

            <div className="upload-section">
                <div className="file-input-wrapper">
                    <input
                        type="file"
                        id="file-upload"
                        accept=".pdf,.txt"
                        onChange={handleFileChange}
                        className="file-input"
                    />
                    <label htmlFor="file-upload" className="file-label">
                        <FileText size={20} />
                        {selectedFile ? selectedFile.name : 'Choose PDF or TXT file'}
                    </label>
                </div>

                <button
                    onClick={handleUpload}
                    disabled={!selectedFile || uploading}
                    className="btn btn-primary"
                >
                    {uploading ? 'Ingesting...' : 'Ingest Document'}
                </button>
            </div>

            {result && (
                <div className="result-box success">
                    <CheckCircle size={20} />
                    <div>
                        <p><strong>Success!</strong> {result.message}</p>
                        <p className="result-details">
                            Filename: <strong>{result.filename}</strong> |
                            Chunks created: <strong>{result.chunks_created}</strong>
                        </p>
                    </div>
                </div>
            )}

            {error && (
                <div className="result-box error">
                    <AlertCircle size={20} />
                    <p>{error}</p>
                </div>
            )}
        </div>
    );
};

export default DocumentUpload;
