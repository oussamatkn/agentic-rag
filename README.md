# 🎓 Agentic AI Legal Assistant

A professional web application combining **Agentic RAG (Retrieval-Augmented Generation)** with a modern React frontend for educational legal research.

![Educational Use](https://img.shields.io/badge/Purpose-Educational-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal)

---

## 📋 Overview

This system demonstrates an **agentic approach to legal research** using:
- **Planning**: Autonomous question decomposition
- **Retrieval**: Semantic search across legal documents
- **Evaluation**: Sufficiency assessment
- **Iteration**: Adaptive refinement (up to 3 cycles)

### Key Features

✅ **Document Ingestion**: Upload PDF/TXT legal documents  
✅ **Intelligent Q&A**: Ask legal questions with structured answers  
✅ **Source Citations**: Full transparency with relevance scores  
✅ **Agent Transparency**: See how the AI agent works  
✅ **Professional UI**: Clean, academic, responsive design  
✅ **Educational Focus**: Clear legal disclaimers  

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** ([Download](https://nodejs.org/))

### Installation & Run

**1. Install Backend Dependencies**
```bash
cd "D:\AI Legal Assistant"
pip install -r requirements.txt
```

**2. Install Frontend Dependencies**
```bash
cd frontend
npm install
```

**3. Start Backend** (Terminal 1)
```bash
cd "D:\AI Legal Assistant"
python main.py
```
✅ Backend runs at: `http://localhost:8000`

**4. Start Frontend** (Terminal 2)
```bash
cd "D:\AI Legal Assistant\frontend"
npm run dev
```
✅ Frontend runs at: `http://localhost:5173`

**5. Open Browser**
Navigate to: **http://localhost:5173**

---

## 📖 Usage

### 1. Upload Documents
- Click "Choose PDF or TXT file"
- Select a legal document
- Click "Ingest Document"
- View confirmation with chunk count

### 2. Ask Questions
- Type your legal question
- Click "Ask Legal Question"
- Wait for agent processing
- View structured answer

### 3. Review Results
The answer includes:
- **Summary**: Quick overview
- **Legal Reasoning**: Detailed analysis
- **Cited Sources**: Document references with scores
- **Confidence**: AI confidence level
- **Disclaimer**: Educational use notice

### 4. Understand the Process
Expand the "Agentic Process Transparency" panel to see:
- Planning phase
- Retrieval strategy
- Evaluation criteria
- Iteration count

---

## 🏗️ Architecture

### Backend (FastAPI)
```
main.py
├── LegalAgent (Orchestrator)
├── Planner (Question decomposition)
├── Retriever (Vector search)
├── Evaluator (Sufficiency check)
└── AnswerGenerator (Response synthesis)
```

### Frontend (React + Vite)
```
frontend/src/
├── App.jsx (Main application)
├── components/
│   ├── Header.jsx
│   ├── DocumentUpload.jsx
│   ├── QuestionInput.jsx
│   ├── AnswerDisplay.jsx
│   └── AgentTransparency.jsx
└── services/
    └── api.js (Backend integration)
```

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF document processing
- **Uvicorn**: ASGI server

### Frontend
- **React 18**: UI framework
- **Vite 5**: Build tool
- **Axios**: HTTP client
- **Lucide React**: Icon library
- **Vanilla CSS**: Professional styling

---

## 📁 Project Structure

```
D:\AI Legal Assistant\
│
├── main.py                    # FastAPI backend
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── RUN_INSTRUCTIONS.md        # Detailed setup guide
├── QUICKSTART.md              # Quick reference
│
└── frontend/
    ├── package.json           # Node dependencies
    ├── vite.config.js         # Vite configuration
    ├── index.html             # HTML template
    └── src/
        ├── main.jsx           # React entry
        ├── App.jsx            # Main component
        ├── App.css            # Styles
        ├── components/        # UI components
        └── services/          # API integration
```

---

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload legal documents |
| `/ask` | POST | Submit legal questions |
| `/health` | GET | System health check |
| `/docs` | GET | Interactive API documentation |

---

## ⚠️ Important Notes

### Educational Use Only

This system is designed for **educational and research purposes only**. It does not provide legal advice and should not be used as a substitute for professional legal counsel.

### Legal Disclaimer

All responses include a prominent disclaimer stating that the information is for educational purposes only and does not constitute legal advice.

### Data Privacy

- Documents are processed in-memory
- No persistent storage (data cleared on restart)
- For production, implement proper database storage

---

## 📚 Documentation

- **[RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)**: Comprehensive setup and troubleshooting guide
- **[QUICKSTART.md](QUICKSTART.md)**: Quick reference for starting the application
- **API Docs**: Visit `http://localhost:8000/docs` when backend is running

---

## 🎯 Demo Workflow

Perfect for PFE presentations:

1. **Start both servers** (backend + frontend)
2. **Show the UI** - Clean, professional design
3. **Upload a document** - Demonstrate ingestion
4. **Ask a question** - Show agent processing
5. **Explain the answer** - Highlight structured output
6. **Open transparency panel** - Explain agentic workflow
7. **Discuss architecture** - Planning → Retrieval → Evaluation

---

## 🔄 Future Enhancements

### Production Readiness
- [ ] Integrate real LLM (OpenAI, Anthropic)
- [ ] Add user authentication
- [ ] Implement persistent storage
- [ ] Add conversation history
- [ ] Export answers to PDF

### Advanced Features
- [ ] Multi-language support
- [ ] Document management UI
- [ ] Advanced search filters
- [ ] Collaborative annotations
- [ ] API rate limiting

---

## 🤝 Contributing

This is an educational project. For improvements:
1. Test thoroughly
2. Maintain code quality
3. Update documentation
4. Follow existing patterns

---

## 📄 License

Educational project - for academic use.

---

## 🎓 Academic Context

This project demonstrates:
- Modern web development practices
- Agentic AI architecture
- RAG (Retrieval-Augmented Generation)
- Full-stack integration
- Professional UI/UX design

Perfect for:
- PFE (Projet de Fin d'Études) demonstrations
- AI/ML coursework
- Web development portfolios
- Legal tech research

---

## 📞 Support

For issues or questions:
1. Check [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)
2. Verify prerequisites are installed
3. Ensure both servers are running
4. Check browser console for errors

---

## ✨ Credits

Built with modern web technologies and best practices for educational excellence.

**Technologies**: React, Vite, FastAPI, FAISS, PyPDF2  
**Purpose**: Educational Legal Research  
**Status**: Ready for Demo ✅

---

**🎉 Ready to explore agentic AI for legal research!**

Visit **http://localhost:5173** after starting both servers.
