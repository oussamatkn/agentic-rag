# ⚡ EXACT RUN COMMANDS - Copy & Paste Ready

## 🎯 For Your PFE Demo

### Step 1: Install Backend Dependencies

Open PowerShell/Terminal and run:

```powershell
cd "D:\AI Legal Assistant"
pip install -r requirements.txt
```

**Expected packages installed:**
- fastapi
- uvicorn
- python-multipart
- PyPDF2
- faiss-cpu
- numpy
- pydantic

---

### Step 2: Start Backend Server

**In Terminal 1:**

```powershell
cd "D:\AI Legal Assistant"
python main.py
```

**Wait for this output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

✅ **Backend is ready!**

---

### Step 3: Start Frontend Server

**In Terminal 2 (new terminal):**

```powershell
cd "D:\AI Legal Assistant\frontend"
npm run dev
```

**Wait for this output:**
```
➜  Local:   http://localhost:5173/
```

✅ **Frontend is ready!**

---

### Step 4: Open Browser

Navigate to: **http://localhost:5173**

The application will load automatically.

---

## 🎬 Demo Script

### 1. Show the UI (30 seconds)
- Point out the professional, clean design
- Highlight the header: "Agentic AI Legal Assistant"
- Mention the educational focus

### 2. Explain Agent Transparency (1 minute)
- Click to expand the "Agentic Process Transparency" panel
- Explain the 4 phases:
  - **Planning**: Question decomposition
  - **Retrieval**: Semantic search
  - **Evaluation**: Sufficiency check
  - **Iteration**: Up to 3 refinement cycles

### 3. Upload a Document (1 minute)
- Click "Choose PDF or TXT file"
- Select a sample legal document
- Click "Ingest Document"
- Show the success message with chunk count

### 4. Ask a Question (2 minutes)
- Type: "What are the requirements for forming a valid contract?"
- Click "Ask Legal Question"
- Point out the loading indicator
- Wait for results

### 5. Review the Answer (2 minutes)
Walk through each section:
- **Summary**: Quick overview
- **Legal Reasoning**: Detailed analysis
- **Cited Sources**: Show document IDs and relevance scores
- **Confidence Score**: AI's confidence level
- **Timestamp**: When processed
- **Legal Disclaimer**: Highlight educational use

### 6. Explain the Architecture (2 minutes)
- **Backend**: FastAPI with agentic RAG
- **Frontend**: React + Vite
- **Vector Store**: FAISS for similarity search
- **Process**: Planning → Retrieval → Evaluation → Synthesis

---

## 🛑 To Stop the Application

### Stop Backend
In Terminal 1: Press `Ctrl + C`

### Stop Frontend
In Terminal 2: Press `Ctrl + C`

---

## 📊 Key Talking Points

### Technical Excellence
- ✅ Modern tech stack (React 18, Vite 5, FastAPI)
- ✅ Agentic AI architecture (not just simple RAG)
- ✅ Professional UI/UX design
- ✅ Full-stack integration
- ✅ Comprehensive documentation

### Educational Value
- ✅ Demonstrates autonomous AI agents
- ✅ Shows retrieval-augmented generation
- ✅ Transparent AI decision-making
- ✅ Real-world application (legal research)
- ✅ Production-ready code quality

### Innovation
- ✅ Multi-phase agent workflow
- ✅ Iterative refinement
- ✅ Source citation with confidence scores
- ✅ User transparency into AI process
- ✅ Structured, professional output

---

## 🎯 Sample Questions to Demo

1. **Contract Law**
   - "What are the requirements for forming a valid contract?"
   - "What constitutes breach of contract?"

2. **Property Law**
   - "What are the types of property ownership?"
   - "What are tenant rights in lease agreements?"

3. **Corporate Law**
   - "What are the duties of corporate directors?"
   - "What is required to form a corporation?"

---

## ✅ Pre-Demo Checklist

- [ ] Python 3.8+ installed and working
- [ ] Node.js 16+ installed and working
- [ ] Backend dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (already done via `npm install`)
- [ ] Sample legal documents prepared (PDF or TXT)
- [ ] Sample questions prepared
- [ ] Both terminals ready
- [ ] Browser ready (Chrome/Firefox/Edge recommended)

---

## 🚨 Quick Troubleshooting

### Backend won't start
```powershell
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Frontend won't start
```powershell
# Reinstall dependencies
cd "D:\AI Legal Assistant\frontend"
rm -r node_modules
npm install
```

### Can't connect
- ✅ Check backend is running on port 8000
- ✅ Check frontend is running on port 5173
- ✅ Check no firewall blocking localhost
- ✅ Try refreshing browser

---

## 📁 Project Files Summary

**Created:**
- ✅ Complete React frontend (12 files)
- ✅ Professional CSS styling (600+ lines)
- ✅ API integration layer
- ✅ 5 React components
- ✅ Comprehensive documentation (4 files)
- ✅ Requirements file

**Modified:**
- ✅ main.py (added CORS middleware only)

**Total Lines of Code:**
- Frontend: ~1,500 lines
- Documentation: ~500 lines
- Backend changes: 10 lines

---

## 🎉 You're Ready!

Everything is set up and ready for your PFE demo.

**Just run the two commands above and you're live!**

Good luck with your presentation! 🚀
