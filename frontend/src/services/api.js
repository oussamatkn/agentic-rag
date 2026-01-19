import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const ingestDocument = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/ingest', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const askQuestion = async (question) => {
    const response = await api.post('/ask', { question });
    return response.data;
};

export const getHealth = async () => {
    const response = await api.get('/health');
    return response.data;
};

export default api;
