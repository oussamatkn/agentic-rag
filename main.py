# ============================================================================
# IMPORTS SECTION
# ============================================================================
# Standard library imports for file operations, JSON handling, and utilities
import os
import json
import hashlib  # Used for generating deterministic embeddings in mock mode
from dotenv import load_dotenv # Load environment variables from .env file
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict  # For creating structured data classes
from datetime import datetime  # For timestamping operations
from pathlib import Path

# FastAPI framework imports for building the REST API
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # For handling cross-origin requests
from pydantic import BaseModel  # For request/response validation
import uvicorn  # ASGI server for running the FastAPI application

# Optional dependency: FAISS for efficient vector similarity search
# If not installed, the system will fall back to basic search
try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

# Optional dependency: PyPDF2 for extracting text from PDF documents
# If not installed, PDF processing will be disabled
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Optional dependency: Google Generative AI for embeddings and LLM
# If not installed, the system will use mock implementations
try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ============================================================================
# DATA CLASSES SECTION
# ============================================================================
# These classes define the core data structures used throughout the application

@dataclass
class Document:
    """
    Represents a single document chunk stored in the vector database.
    
    Attributes:
        id: Unique identifier for the document chunk (e.g., "file.pdf::chunk_0")
        content: The actual text content of the chunk
        metadata: Additional information (filename, chunk index, timestamp, etc.)
        embedding: Vector representation of the content for similarity search
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RetrievalResult:
    """
    Represents a search result returned from the vector database.
    
    Attributes:
        document_id: ID of the retrieved document chunk
        content: Text content of the retrieved chunk
        score: Similarity score (higher = more relevant)
        metadata: Additional information about the document
    """
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


@dataclass
class PlanStep:
    """
    Represents a single step in the agent's query decomposition plan.
    
    The agent breaks down complex questions into simpler sub-questions.
    Each sub-question becomes a PlanStep that guides the retrieval process.
    
    Attributes:
        step_id: Sequential identifier for this step
        query: The sub-question to be answered
        rationale: Explanation of why this step is needed
        completed: Whether this step has been executed
    """
    step_id: int
    query: str
    rationale: str
    completed: bool = False


@dataclass
class AgentState:
    """
    Tracks the complete state of the agent during question processing.
    
    This is the "memory" of the agent's reasoning process, storing:
    - The original user question
    - The decomposed plan (sub-questions)
    - All retrieved information chunks
    - Number of iterations performed
    - Evaluation results from each iteration
    - The final synthesized answer
    
    Attributes:
        original_question: The user's original legal question
        plan: List of sub-questions the agent will answer
        retrieved_chunks: All document chunks retrieved so far
        iterations: Number of retrieval-evaluation cycles completed
        evaluation_results: Sufficiency assessments from each iteration
        final_answer: The synthesized answer (JSON string)
    """
    original_question: str
    plan: List[PlanStep]
    retrieved_chunks: List[RetrievalResult]
    iterations: int
    evaluation_results: List[Dict[str, Any]]
    final_answer: Optional[str] = None


class MockEmbedder:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        seed = int.from_bytes(hash_bytes[:4], 'little')
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype('float32')
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]



class GeminiLLM:
    """
    Wrapper for Google's Gemini language model API.
    
    This class handles all interactions with the Gemini LLM for:
    - Query decomposition (breaking questions into sub-questions)
    - Sufficiency evaluation (determining if retrieved info is enough)
    - Answer synthesis (generating final legal responses)
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the Gemini LLM with API credentials.
        
        Args:
            api_key: Google API key for Gemini access
            model_name: Specific Gemini model to use (default: gemini-2.5-flash)
        """
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        # Configure the API with credentials
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Default generation parameters for balanced, coherent responses
        self.generation_config = {
            "temperature": 0.7,      # Controls randomness (0=deterministic, 1=creative)
            "top_p": 0.95,           # Nucleus sampling threshold
            "top_k": 40,             # Limits vocabulary to top K tokens
            "max_output_tokens": 2048,  # Maximum response length
        }
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate text completion from the LLM.
        
        Args:
            prompt: The input prompt/question for the LLM
            max_tokens: Maximum length of the generated response
            
        Returns:
            Generated text response (or error JSON if failed)
        """
        try:
            # Override max tokens for this specific request
            config = self.generation_config.copy()
            config["max_output_tokens"] = max_tokens
            
            # Call the Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config=config
            )
            
            return response.text
        except Exception as e:
            print(f"[ERROR] Gemini API call failed: {e}")
            # Return error as JSON for graceful degradation
            return json.dumps({"error": str(e)})


class GeminiEmbedder:
    """
    Wrapper for Google's Gemini text embedding API.
    
    Converts text into high-dimensional vectors (embeddings) that capture
    semantic meaning. Similar texts will have similar embeddings, enabling
    semantic search in the vector database.
    """
    def __init__(self, dimension: int = 768):
        """
        Initialize the Gemini embedder.
        
        Args:
            dimension: Target embedding dimension (default: 768 for Gemini)
        """
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        self.dimension = dimension
        self.model_name = "models/text-embedding-004"  # Gemini's embedding model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text into a semantic embedding vector.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding (length = self.dimension)
        """
        try:
            # Call Gemini embedding API
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"  # Optimized for document retrieval
            )
            embedding = result['embedding']
            
            # Ensure embedding matches expected dimension
            # Pad with zeros if too short, truncate if too long
            if len(embedding) < self.dimension:
                embedding = embedding + [0.0] * (self.dimension - len(embedding))
            elif len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            
            return embedding
        except Exception as e:
            print(f"[ERROR] Embedding failed: {e}")
            # Fallback to zero vector (will match poorly with everything)
            return [0.0] * self.dimension
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (processes one at a time)."""
        return [self.embed_text(t) for t in texts]



# Keep MockEmbedder as fallback (duplicate definition for compatibility)
class MockEmbedder:
    """
    Fallback embedder for when Gemini API is unavailable.
    
    This is a duplicate definition that serves as a fallback implementation.
    Creates deterministic embeddings using MD5 hashing and random number generation.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash."""
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        seed = int.from_bytes(hash_bytes[:4], 'little')
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dimension).astype('float32')
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


# Fallback MockLLM for when Gemini API is unavailable
class MockLLM:
    """
    Mock language model for testing and fallback scenarios.
    
    Provides hardcoded responses for different types of prompts:
    - Decomposition prompts: Returns sample sub-questions
    - Evaluation prompts: Returns sufficiency assessment
    - Synthesis prompts: Returns generic legal analysis
    
    This ensures the system continues to function even without API access.
    """
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate mock responses based on prompt keywords.
        
        Args:
            prompt: Input prompt (analyzed for keywords)
            max_tokens: Ignored in mock implementation
            
        Returns:
            JSON string or text response appropriate to the prompt type
        """
        # Handle query decomposition requests
        if "decompose" in prompt.lower() or "sub-questions" in prompt.lower():
            return json.dumps({
                "sub_questions": [
                    "What are the key legal definitions?",
                    "What are the applicable regulations?",
                    "What are the legal requirements?"
                ]
            })
        # Handle sufficiency evaluation requests
        elif "evaluate" in prompt.lower() or "sufficient" in prompt.lower():
            return json.dumps({
                "is_sufficient": True,
                "confidence": 0.85,
                "missing_aspects": []
            })
        # Handle answer synthesis requests
        elif "synthesize" in prompt.lower() or "final answer" in prompt.lower():
            return "Based on the retrieved legal documents, the answer involves multiple legal considerations including regulatory compliance and contractual obligations."
        # Default fallback response
        else:
            return "Legal analysis response based on provided context."



# ============================================================================
# VECTOR STORE CLASS
# ============================================================================
# Manages document storage and similarity search

class VectorStore:
    """
    In-memory vector database for storing and searching document embeddings.
    
    Uses FAISS (Facebook AI Similarity Search) for efficient similarity search.
    If FAISS is unavailable, falls back to returning first K documents.
    
    The vector store maintains:
    - A list of Document objects with their content and metadata
    - A FAISS index for fast similarity search on embeddings
    """
    def __init__(self, dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimensionality of embedding vectors (must match embedder)
        """
        self.dimension = dimension
        self.documents: List[Document] = []
        self.index = None
        # Create FAISS index if available (IndexFlatIP = Inner Product search)
        if faiss and np:
            self.index = faiss.IndexFlatIP(dimension)
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of Document objects with embeddings
        """
        self.documents.extend(documents)
        # Add embeddings to FAISS index for fast search
        if self.index and faiss and np:
            embeddings = np.array([doc.embedding for doc in documents], dtype='float32')
            self.index.add(embeddings)
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """
        Search for documents most similar to the query embedding.
        
        Args:
            query_embedding: Vector representation of the search query
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (highest first)
        """
        if not self.documents:
            return []
        
        # Use FAISS for efficient similarity search
        if self.index and faiss and np:
            query_vec = np.array([query_embedding], dtype='float32')
            # Search returns (scores, indices) of top_k most similar vectors
            scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append(RetrievalResult(
                        document_id=doc.id,
                        content=doc.content,
                        score=float(score),
                        metadata=doc.metadata
                    ))
            return results
        # Fallback: return first K documents with dummy scores
        else:
            return [
                RetrievalResult(
                    document_id=doc.id,
                    content=doc.content,
                    score=0.8,
                    metadata=doc.metadata
                )
                for doc in self.documents[:top_k]
            ]


# ============================================================================
# AGENT COMPONENT CLASSES
# ============================================================================
# These classes implement the different stages of the agentic RAG pipeline

class Planner:
    """
    Decomposes complex questions into simpler sub-questions.
    
    This is the first stage of the agentic RAG pipeline. The planner uses
    the LLM to break down a complex legal question into multiple focused
    sub-questions that can be answered independently.
    
    Example:
        Question: "What are my rights as a tenant?"
        Sub-questions:
            1. What are the key legal definitions for tenants?
            2. What are the applicable tenant protection regulations?
            3. What are the legal requirements for landlords?
    """
    def __init__(self, llm: MockLLM):
        """Initialize with a language model."""
        self.llm = llm
    
    def create_plan(self, question: str) -> List[PlanStep]:
        """
        Decompose a question into sub-questions.
        
        Args:
            question: The user's original legal question
            
        Returns:
            List of PlanStep objects, each representing a sub-question
        """
        # Prompt the LLM to decompose the question
        prompt = f"""Analyze this legal question and decompose it into sub-questions:
Question: {question}

Provide a JSON response with 'sub_questions' array."""
        
        response = self.llm.generate(prompt)
        try:
            # Parse LLM response as JSON
            parsed = json.loads(response)
            sub_questions = parsed.get("sub_questions", [])
        except:
            # Fallback to generic sub-questions if parsing fails
            sub_questions = [
                "What are the relevant legal provisions?",
                "What are the applicable interpretations?",
                "What are the practical implications?"
            ]
        
        # Convert sub-questions into PlanStep objects
        plan = []
        for idx, sq in enumerate(sub_questions):
            plan.append(PlanStep(
                step_id=idx + 1,
                query=sq,
                rationale=f"Legal analysis component {idx + 1}"
            ))
        return plan



class Retriever:
    """
    Retrieves relevant documents from the vector store based on queries.
    
    This is the second stage of the agentic RAG pipeline. The retriever:
    1. Converts text queries into embeddings
    2. Searches the vector store for similar documents
    3. Returns the most relevant chunks
    """
    def __init__(self, vector_store: VectorStore, embedder: MockEmbedder):
        """Initialize with a vector store and embedder."""
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Text query (e.g., a sub-question from the planner)
            top_k: Number of top results to return
            
        Returns:
            List of RetrievalResult objects with relevant document chunks
        """
        # Convert query text to embedding vector
        query_embedding = self.embedder.embed_text(query)
        # Search vector store for similar documents
        results = self.vector_store.search(query_embedding, top_k)
        return results



class Evaluator:
    """
    Evaluates whether retrieved information is sufficient to answer the question.
    
    This is the third stage of the agentic RAG pipeline. The evaluator:
    1. Analyzes the retrieved document chunks
    2. Determines if they contain enough information
    3. Identifies missing aspects if information is insufficient
    
    This enables the agent to iteratively refine its retrieval strategy.
    """
    def __init__(self, llm: MockLLM):
        """Initialize with a language model."""
        self.llm = llm
    
    def evaluate_sufficiency(self, question: str, retrieved_chunks: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Evaluate if retrieved information is sufficient to answer the question.
        
        Args:
            question: The original user question
            retrieved_chunks: Documents retrieved so far
            
        Returns:
            Dictionary with:
                - is_sufficient (bool): Whether info is adequate
                - confidence (float): Confidence score (0-1)
                - missing_aspects (list): What information is still needed
        """
        # Create context summary from retrieved chunks (truncated for efficiency)
        context = "\n".join([f"[{r.document_id}] {r.content[:200]}" for r in retrieved_chunks])
        
        # Prompt LLM to evaluate sufficiency
        prompt = f"""Evaluate if the retrieved information is sufficient to answer the legal question.

Question: {question}
Retrieved Context:
{context}

Provide JSON with: is_sufficient (bool), confidence (float), missing_aspects (list)."""
        
        response = self.llm.generate(prompt)
        try:
            # Parse LLM evaluation response
            result = json.loads(response)
        except:
            # Fallback: assume sufficient if we have at least 3 chunks
            result = {
                "is_sufficient": len(retrieved_chunks) >= 3,
                "confidence": 0.75,
                "missing_aspects": []
            }
        
        return result



class Memory:
    """
    Stores the agent's processing history and state across iterations.
    
    Maintains a record of all agent states, allowing the system to:
    - Track the agent's reasoning process
    - Debug issues
    - Analyze performance
    - Potentially resume interrupted sessions
    """
    def __init__(self):
        """Initialize empty memory."""
        self.states: List[AgentState] = []
    
    def save_state(self, state: AgentState):
        """Save an agent state to memory."""
        self.states.append(state)
    
    def get_last_state(self) -> Optional[AgentState]:
        """Retrieve the most recent agent state."""
        return self.states[-1] if self.states else None



class AnswerGenerator:
    """
    Synthesizes a final answer from retrieved documents.
    
    This is the final stage of the agentic RAG pipeline. The generator:
    1. Takes all retrieved document chunks
    2. Uses the LLM to synthesize a coherent answer
    3. Formats the response with citations and metadata
    4. Adds appropriate legal disclaimers
    """
    def __init__(self, llm: MockLLM):
        """Initialize with a language model."""
        self.llm = llm
    
    def generate_answer(self, question: str, retrieved_chunks: List[RetrievalResult], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive answer from retrieved documents.
        
        Args:
            question: The original user question
            retrieved_chunks: All relevant document chunks
            evaluation: Sufficiency evaluation results
            
        Returns:
            Dictionary containing:
                - summary: Brief answer summary
                - legal_reasoning: Full detailed analysis
                - cited_sources: List of source documents with metadata
                - disclaimer: Legal disclaimer text
                - confidence: Confidence score
                - timestamp: When the answer was generated
        """
        # Combine all retrieved chunks into context for the LLM
        context = "\n\n".join([
            f"Source [{r.document_id}]:\n{r.content}"
            for r in retrieved_chunks
        ])
        
        # Prompt LLM to synthesize answer from context
        prompt = f"""Synthesize a comprehensive legal answer based on the retrieved documents.

Question: {question}

Retrieved Legal Context:
{context}

Provide a structured legal analysis."""
        
        # Generate answer text
        answer_text = self.llm.generate(prompt, max_tokens=1000)
        
        # Prepare source citations with metadata
        sources = [
            {
                "document_id": r.document_id,
                "relevance_score": r.score,
                "metadata": r.metadata
            }
            for r in retrieved_chunks
        ]
        
        # Standard legal disclaimer
        legal_disclaimer = """LEGAL DISCLAIMER: This response is generated for educational and informational purposes only. It does not constitute legal advice and should not be relied upon as such. For specific legal matters, please consult a qualified attorney licensed in your jurisdiction."""
        
        # Return structured response
        return {
            "summary": answer_text[:300] + "..." if len(answer_text) > 300 else answer_text,
            "legal_reasoning": answer_text,
            "cited_sources": sources,
            "disclaimer": legal_disclaimer,
            "confidence": evaluation.get("confidence", 0.0),
            "timestamp": datetime.utcnow().isoformat()
        }



# ============================================================================
# LEGAL AGENT CLASS - Main Orchestrator
# ============================================================================
# This is the core of the agentic RAG system, coordinating all components

class LegalAgent:
    """
    Main orchestrator for the agentic RAG (Retrieval-Augmented Generation) system.
    
    The LegalAgent implements an iterative question-answering workflow:
    
    1. PLANNING: Decompose complex question into sub-questions
    2. RETRIEVAL: Retrieve relevant documents for each sub-question
    3. EVALUATION: Assess if retrieved information is sufficient
    4. RE-PLANNING: If insufficient, identify missing info and retrieve more
    5. SYNTHESIS: Generate final answer from all retrieved information
    
    This iterative approach allows the agent to refine its search strategy
    and gather comprehensive information before generating an answer.
    """
    def __init__(self, retriever: Retriever, planner: Planner, evaluator: Evaluator, 
                 answer_generator: AnswerGenerator, memory: Memory, max_iterations: int = 3):
        """
        Initialize the legal agent with all necessary components.
        
        Args:
            retriever: Handles document retrieval from vector store
            planner: Decomposes questions into sub-questions
            evaluator: Assesses information sufficiency
            answer_generator: Synthesizes final answers
            memory: Stores processing history
            max_iterations: Maximum retrieval-evaluation cycles (prevents infinite loops)
        """
        self.retriever = retriever
        self.planner = planner
        self.evaluator = evaluator
        self.answer_generator = answer_generator
        self.memory = memory
        self.max_iterations = max_iterations
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a legal question through the complete agentic RAG pipeline.
        
        Workflow:
        1. Plan: Break question into sub-questions
        2. Loop (up to max_iterations):
           a. Retrieve: Get relevant docs for each sub-question
           b. Evaluate: Check if information is sufficient
           c. If sufficient or max iterations reached: break
           d. Otherwise: Re-plan to address missing information
        3. Generate: Synthesize final answer from all retrieved docs
        
        Args:
            question: The user's legal question
            
        Returns:
            Dictionary with answer, reasoning, sources, and metadata
        """
        print(f"\n[AGENT] Processing question: {question}")
        
        # ========== STEP 1: PLANNING ==========
        print("[AGENT] Step 1: Planning - Decomposing question into sub-queries")
        plan = self.planner.create_plan(question)
        
        # Initialize agent state to track the entire process
        state = AgentState(
            original_question=question,
            plan=plan,
            retrieved_chunks=[],
            iterations=0,
            evaluation_results=[]
        )
        
        # Accumulate all retrieved documents across iterations
        all_retrieved: List[RetrievalResult] = []
        
        # ========== ITERATIVE RETRIEVAL-EVALUATION LOOP ==========
        for iteration in range(self.max_iterations):
            state.iterations = iteration + 1
            print(f"\n[AGENT] Iteration {iteration + 1}/{self.max_iterations}")
            
            # ========== STEP 2: RETRIEVAL ==========
            print("[AGENT] Step 2: Retrieval - Executing retrieval for plan steps")
            for step in plan:
                if not step.completed:
                    print(f"[AGENT]   Retrieving for: {step.query}")
                    # Retrieve top 3 documents for this sub-question
                    results = self.retriever.retrieve(step.query, top_k=3)
                    all_retrieved.extend(results)
                    step.completed = True
            
            # Update state with newly retrieved documents
            state.retrieved_chunks = all_retrieved
            
            # ========== STEP 3: EVALUATION ==========
            print("[AGENT] Step 3: Evaluation - Assessing information sufficiency")
            evaluation = self.evaluator.evaluate_sufficiency(question, all_retrieved)
            state.evaluation_results.append(evaluation)
            
            print(f"[AGENT]   Sufficient: {evaluation['is_sufficient']}, Confidence: {evaluation['confidence']}")
            
            # Check if we have enough information or reached max iterations
            if evaluation["is_sufficient"] or iteration == self.max_iterations - 1:
                print("[AGENT] Information deemed sufficient or max iterations reached")
                break
            else:
                # ========== STEP 4: RE-PLANNING ==========
                print("[AGENT] Step 4: Re-planning for missing information")
                missing = evaluation.get("missing_aspects", [])
                if missing:
                    # Create new plan steps for missing information
                    additional_steps = [
                        PlanStep(
                            step_id=len(plan) + i + 1,
                            query=aspect,
                            rationale="Addressing missing information"
                        )
                        for i, aspect in enumerate(missing)
                    ]
                    plan.extend(additional_steps)
        
        # ========== STEP 5: ANSWER GENERATION ==========
        print("[AGENT] Step 5: Answer Generation - Synthesizing final response")
        final_answer_obj = self.answer_generator.generate_answer(
            question, 
            all_retrieved,
            state.evaluation_results[-1] if state.evaluation_results else {}
        )
        state.final_answer = json.dumps(final_answer_obj)
        
        # Save the complete processing state to memory
        self.memory.save_state(state)
        
        return final_answer_obj



# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================
# Handles file ingestion, text extraction, and chunking for the vector store

class DocumentProcessor:
    """
    Processes uploaded documents for ingestion into the vector store.
    
    Responsibilities:
    1. Extract text from various file formats (PDF, TXT, etc.)
    2. Split long documents into manageable chunks
    3. Generate embeddings for each chunk
    4. Create Document objects with metadata
    
    Chunking Strategy:
    - Documents are split into overlapping chunks
    - Overlap ensures context isn't lost at chunk boundaries
    - Each chunk is small enough for efficient retrieval and LLM processing
    """
    def __init__(self, embedder: MockEmbedder):
        """
        Initialize the document processor.
        
        Args:
            embedder: Embedder instance for converting text to vectors
        """
        self.embedder = embedder
        self.chunk_size = 500       # Characters per chunk
        self.chunk_overlap = 100    # Overlapping characters between chunks
    
    def process_file(self, file_path: str, file_content: bytes, filename: str) -> List[Document]:
        """
        Process an uploaded file into Document objects.
        
        Args:
            file_path: Path to the file (currently unused, for future use)
            file_content: Raw bytes of the file
            filename: Name of the file (used to determine file type)
            
        Returns:
            List of Document objects, one per chunk
        """
        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = self._extract_pdf(file_content)
        elif filename.endswith('.txt'):
            text = file_content.decode('utf-8', errors='ignore')
        else:
            # Default: treat as text file
            text = file_content.decode('utf-8', errors='ignore')
        
        # Split text into chunks
        chunks = self._chunk_text(text)
        documents = []
        
        # Create a Document object for each chunk
        for idx, chunk in enumerate(chunks):
            doc_id = f"{filename}::chunk_{idx}"
            # Generate embedding for this chunk
            embedding = self.embedder.embed_text(chunk)
            doc = Document(
                id=doc_id,
                content=chunk,
                metadata={
                    "filename": filename,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.utcnow().isoformat()
                },
                embedding=embedding
            )
            documents.append(doc)
        
        return documents
    
    def _extract_pdf(self, content: bytes) -> str:
        """
        Extract text from PDF file bytes.
        
        Args:
            content: Raw PDF file bytes
            
        Returns:
            Extracted text (or error message if extraction fails)
        """
        if PdfReader is None:
            return "PDF extraction not available. Install PyPDF2."
        
        try:
            from io import BytesIO
            pdf_file = BytesIO(content)
            reader = PdfReader(pdf_file)
            text = ""
            # Extract text from each page
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except:
            return "PDF extraction failed."
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Chunking with overlap ensures that information spanning chunk
        boundaries is not lost. For example, a sentence split across
        chunks will appear complete in at least one chunk.
        
        Args:
            text: Full text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move forward by (chunk_size - overlap) to create overlap
            start += self.chunk_size - self.chunk_overlap
        # Return chunks, or full text if no chunks were created
        return chunks if chunks else [text]



# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================
# REST API for the Legal Assistant system

app = FastAPI(title="Agentic RAG Legal Assistant")

# Configure CORS (Cross-Origin Resource Sharing) to allow frontend requests
# This allows the React frontend running on localhost:5173 to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)


# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================
# Initialize all components of the agentic RAG system

# Configuration
# Configuration
load_dotenv() # Load env vars
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_DIMENSION = 768  # Gemini embeddings are 768-dimensional

# Initialize with real Gemini API (with fallback to mock implementations)
try:
    print("[INIT] Initializing with Gemini 2.0 Flash API...")
    llm = GeminiLLM(api_key=GEMINI_API_KEY, model_name="gemini-2.5-flash")
    embedder = GeminiEmbedder(dimension=EMBEDDING_DIMENSION)
    print("[INIT] ✅ Gemini API initialized successfully")
except Exception as e:
    # Fallback to mock implementations if Gemini API is unavailable
    print(f"[INIT] ⚠️ Failed to initialize Gemini API: {e}")
    print("[INIT] Falling back to mock implementations...")
    llm = MockLLM()
    embedder = MockEmbedder(dimension=384)
    EMBEDDING_DIMENSION = 384

# Initialize all system components
vector_store = VectorStore(dimension=EMBEDDING_DIMENSION)
retriever = Retriever(vector_store, embedder)
planner = Planner(llm)
evaluator = Evaluator(llm)
answer_generator = AnswerGenerator(llm)
memory = Memory()
agent = LegalAgent(retriever, planner, evaluator, answer_generator, memory, max_iterations=3)
doc_processor = DocumentProcessor(embedder)


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================
# Pydantic models for request validation and response serialization

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    summary: str
    legal_reasoning: str
    cited_sources: List[Dict[str, Any]]
    disclaimer: str
    confidence: float
    timestamp: str


# ============================================================================
# API ENDPOINTS
# ============================================================================
# REST API endpoints for document ingestion and question answering

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document into the vector store.
    
    This endpoint:
    1. Receives an uploaded file (PDF, TXT, etc.)
    2. Extracts and chunks the text
    3. Generates embeddings for each chunk
    4. Stores the chunks in the vector database
    
    Args:
        file: Uploaded file from the frontend
        
    Returns:
        JSON with status, filename, and number of chunks created
        
    Raises:
        HTTPException: If file processing fails
    """
    try:
        # Read file content
        content = await file.read()
        # Process file into document chunks with embeddings
        documents = doc_processor.process_file("", content, file.filename)
        # Add documents to vector store
        vector_store.add_documents(documents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(documents),
            "message": f"Processed {len(documents)} chunks from {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Answer a legal question using the agentic RAG system.
    
    This endpoint:
    1. Receives a legal question from the user
    2. Runs the question through the agentic RAG pipeline:
       - Planning: Decompose into sub-questions
       - Retrieval: Find relevant document chunks
       - Evaluation: Assess information sufficiency
       - Re-planning: Retrieve more if needed
       - Synthesis: Generate final answer
    3. Returns structured answer with sources and metadata
    
    Args:
        request: AskRequest containing the user's question
        
    Returns:
        AskResponse with answer, reasoning, sources, and disclaimer
        
    Raises:
        HTTPException: If question processing fails
    """
    try:
        # Process question through the agentic RAG pipeline
        result = agent.process_question(request.question)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring system status.
    
    Returns:
        JSON with system health metrics:
        - status: "healthy" if system is running
        - documents_indexed: Number of document chunks in vector store
        - queries_processed: Number of questions answered
        - timestamp: Current server time
    """
    return {
        "status": "healthy",
        "documents_indexed": len(vector_store.documents),
        "queries_processed": len(memory.states),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================
# Start the FastAPI server when running this file directly

if __name__ == "__main__":
    # Run the server on all network interfaces (0.0.0.0) on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)