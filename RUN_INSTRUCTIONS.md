# 🚀 Agentic AI Legal Assistant - Run Instructions

## 📋 Project Overview

This project consists of:
- **Backend**: FastAPI-based Agentic RAG system (`main.py`)
- **Frontend**: React + Vite professional web interface (`frontend/`)

---

## ⚙️ Prerequisites

### Required Software

1. **Python 3.8+**
   - Check version: `python --version`
   - Download: https://www.python.org/downloads/

2. **Node.js 16+**
   - Check version: `node --version`
   - Download: https://nodejs.org/

3. **pip** (Python package manager)
   - Usually included with Python
   - Check: `pip --version`

---

## 🔧 Backend Setup

### 1. Navigate to Project Directory

```bash
cd "D:\AI Legal Assistant"
```

### 2. Install Python Dependencies

```bash
pip install fastapi uvicorn python-multipart PyPDF2 faiss-cpu numpy
```

**Required packages:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `PyPDF2` - PDF parsing
- `faiss-cpu` - Vector similarity search
- `numpy` - Numerical operations

### 3. Start the Backend Server

```bash
python main.py
```

**Expected output:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Backend is now running at:** `http://localhost:8000`

---

## 🎨 Frontend Setup

### 1. Navigate to Frontend Directory

```bash
cd "D:\AI Legal Assistant\frontend"
```

### 2. Install Dependencies (Already Done)

Dependencies are already installed. If needed, run:

```bash
npm install
```

### 3. Start the Development Server

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

✅ **Frontend is now running at:** `http://localhost:5173`

---

## 🌐 Access the Application

### Open in Browser

Navigate to: **http://localhost:5173**

The application will automatically open in your default browser.

---

## 📖 How to Use

### 1. Upload Legal Documents

1. Click **"Choose PDF or TXT file"** in the Document Upload section
2. Select a legal document (PDF or TXT format)
3. Click **"Ingest Document"**
4. Wait for confirmation showing the number of chunks created

### 2. Ask Legal Questions

1. Type your legal question in the textarea
2. Click **"Ask Legal Question"**
3. Wait for the agent to process (shows loading indicator)
4. View the structured answer with:
   - Summary
   - Legal Reasoning
   - Cited Sources
   - Confidence Score
   - Timestamp
   - Legal Disclaimer

### 3. View Agent Process

Click on the **"Agentic Process Transparency"** panel to see:
- Planning phase
- Retrieval phase
- Evaluation phase
- Iteration details

---

## 🛑 Stopping the Application

### Stop Backend
In the terminal running `python main.py`:
- Press `Ctrl + C`

### Stop Frontend
In the terminal running `npm run dev`:
- Press `Ctrl + C`

---

## 🔍 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError`
```bash
# Solution: Install missing packages
pip install fastapi uvicorn python-multipart PyPDF2 faiss-cpu numpy
```

**Problem:** Port 8000 already in use
```bash
# Solution: Kill the process or change port in main.py (line 469)
# Change: uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Frontend Issues

**Problem:** `npm: command not found`
```bash
# Solution: Install Node.js from https://nodejs.org/
```

**Problem:** Port 5173 already in use
```bash
# Solution: The dev server will automatically use the next available port
# Or manually specify in vite.config.js
```

**Problem:** CORS errors in browser console
```bash
# Solution: Ensure backend is running and CORS is configured
# Check that main.py has CORSMiddleware configured for http://localhost:5173
```

### Connection Issues

**Problem:** Frontend can't connect to backend
- ✅ Ensure backend is running on port 8000
- ✅ Check browser console for errors
- ✅ Verify CORS configuration in `main.py`

---

## 📁 Project Structure

```
D:\AI Legal Assistant\
│
├── main.py                          # FastAPI backend
│
└── frontend/
    ├── package.json                 # Dependencies
    ├── vite.config.js              # Vite configuration
    ├── index.html                  # HTML template
    │
    └── src/
        ├── main.jsx                # React entry point
        ├── App.jsx                 # Main app component
        ├── App.css                 # Global styles
        │
        ├── components/
        │   ├── Header.jsx          # Header component
        │   ├── DocumentUpload.jsx  # Upload interface
        │   ├── QuestionInput.jsx   # Q&A interface
        │   ├── AnswerDisplay.jsx   # Answer display
        │   └── AgentTransparency.jsx # Process transparency
        │
        └── services/
            └── api.js              # API integration
```

---

## 🎯 API Endpoints

### Backend API (http://localhost:8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Upload and process legal documents |
| `/ask` | POST | Submit legal questions |
| `/health` | GET | Check system status |

### API Documentation

Visit: **http://localhost:8000/docs** (Swagger UI)

---

## 🎓 Demo Workflow

### Complete Demo Steps

1. **Start Backend**
   ```bash
   cd "D:\AI Legal Assistant"
   python main.py
   ```

2. **Start Frontend** (in new terminal)
   ```bash
   cd "D:\AI Legal Assistant\frontend"
   npm run dev
   ```

3. **Open Browser**
   - Navigate to `http://localhost:5173`

4. **Upload Sample Document**
   - Prepare a legal PDF or TXT file
   - Upload via the interface
   - Verify chunks created

5. **Ask Sample Question**
   - Example: "What are the requirements for contract formation?"
   - Submit and observe agent processing
   - Review structured answer

6. **Show Agent Transparency**
   - Expand the transparency panel
   - Explain the agentic workflow

---

## ⚠️ Important Notes

### Educational Use Only

This system is designed for **educational and research purposes only**. It does not provide legal advice and should not be used as a substitute for professional legal counsel.

### Legal Disclaimer

All responses include a legal disclaimer stating that the information is for educational purposes only and does not constitute legal advice.

### Data Privacy

- Documents are processed in-memory
- No persistent storage (data is lost when server stops)
- For production use, implement proper database storage

---

## 🎨 UI Features

- ✅ Clean, professional, academic design
- ✅ Neutral color palette (grays, blues)
- ✅ Responsive layout (mobile-friendly)
- ✅ Loading indicators
- ✅ Error handling
- ✅ Success/error messages
- ✅ Structured answer display
- ✅ Agent process transparency

---

## 🔄 Development Commands

### Frontend

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend

```bash
# Run server
python main.py

# Run with auto-reload (for development)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## ✅ Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Backend dependencies installed
- [ ] Frontend dependencies installed
- [ ] Backend running on port 8000
- [ ] Frontend running on port 5173
- [ ] Browser opened to http://localhost:5173
- [ ] Sample document ready for upload
- [ ] Sample legal question prepared

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure both servers are running
4. Check browser console for errors

---

**🎉 You're ready to demo the Agentic AI Legal Assistant!**
