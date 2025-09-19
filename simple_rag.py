import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # ✅ Fix Keras compatibility

from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
from typing import List, Tuple
import re

class SimpleRAG:
    def __init__(self):
        try:
            # Load models with error handling
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.qa_model = pipeline('question-answering',
                                   model='distilbert-base-cased-distilled-squad',
                                   device=-1,  # Force CPU usage
                                   model_kwargs={"low_cpu_mem_usage": True})
            
            self.index = None
            self.documents = []
            print("SimpleRAG initialized successfully")
            
        except Exception as e:
            print(f"Error initializing SimpleRAG: {e}")
            raise
    
    def add_documents(self, docs: List[str]):
        """Add documents to knowledge base with improved validation"""
        if not docs:
            print("Warning: No documents provided")
            return
        
        # Filter and validate documents
        valid_docs = []
        for doc in docs:
            if self._validate_document(doc):
                valid_docs.append(doc)
        
        if not valid_docs:
            print("Error: No valid documents found after filtering")
            return
        
        self.documents = valid_docs
        
        try:
            # Create embeddings
            print(f"Creating embeddings for {len(valid_docs)} documents...")
            embeddings = self.encoder.encode(valid_docs, show_progress_bar=True)
            
            # Build FAISS index
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            print(f"Successfully added {len(valid_docs)} documents to RAG system")
            
        except Exception as e:
            print(f"Error adding documents to RAG: {e}")
            raise
    
    def _validate_document(self, doc: str) -> bool:
        """Validate document content quality"""
        if not doc or len(doc.strip()) < 50:
            return False
        
        # Check for metadata contamination
        if doc.strip().startswith("Chunk") or "energy_storage_research" in doc.lower():
            return False
        
        # Ensure meaningful content
        words = doc.split()
        return len(words) >= 20 and len([w for w in words if len(w) > 2]) >= 15
    
    def _clean_answer(self, answer_text: str) -> str:
        """Clean answer to remove metadata contamination"""
        if not answer_text:
            return "Information not found"
        
        # Remove problematic patterns
        problematic_patterns = [
            r'^energy[_\s]storage[_\s]research',
            r'^Chunk\s+\d+',
            r'^\[Source:.*\]$',
            r'^Page\s+\d+$'
        ]
        
        for pattern in problematic_patterns:
            if re.match(pattern, answer_text.strip(), re.IGNORECASE):
                return "Information not found in document"
        
        # Fix common formatting issues
        answer_text = re.sub(r'\$(\d+)\s+(\d+)', r'$\1.\2', answer_text)  # Fix "$0 10" -> "$0.10"
        answer_text = re.sub(r'(\d+)\s+-\s*(\d+)', r'\1-\2', answer_text)  # Fix "10 - 20" -> "10-20"
        
        return answer_text.strip()
    
    def answer_question(self, question: str) -> dict:
        """Answer question using RAG with improved processing"""
        if not self.documents:
            return {"answer": "No documents available", "confidence": 0}
        
        if not question.strip():
            return {"answer": "Please provide a valid question", "confidence": 0}
        
        try:
            print(f"RAG DEBUG - Processing question: '{question}'")
            
            # Retrieve relevant documents
            query_embedding = self.encoder.encode([question])
            faiss.normalize_L2(query_embedding)
            
            # Search for top 3 most similar documents
            scores, indices = self.index.search(query_embedding.astype('float32'), k=3)
            
            # Check if any documents were retrieved
            if len(indices[0]) == 0:
                return {"answer": "No relevant documents found", "confidence": 0}
            
            # Get top documents with bounds checking and score filtering
            valid_docs = []
            valid_scores = []
            
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.documents) and score > 0.1:  # Filter low-similarity docs
                    valid_docs.append(self.documents[idx])
                    valid_scores.append(score)
            
            if not valid_docs:
                return {"answer": "No sufficiently relevant documents found", "confidence": 0}
            
            # Create combined context with better formatting
            contexts = []
            for doc in valid_docs[:2]:  # Use top 2 documents
                # Extract meaningful content, skip source tags
                content = doc
                if ']' in content:
                    content = content.split(']', 1)[1].strip()  # Remove source tag
                contexts.append(content)
            
            combined_context = " ".join(contexts)[:1500]  # Limit context length
            
            print(f"RAG DEBUG - Combined context preview: {combined_context[:200]}...")
            
            # Generate answer using QA model
            result = self.qa_model(question=question, context=combined_context)
            
            print(f"RAG DEBUG - Raw QA result: {result}")
            
            # Clean the answer
            cleaned_answer = self._clean_answer(result['answer'])
            
            return {
                "answer": cleaned_answer,
                "confidence": result['score'],
                "retrieved_docs": valid_docs,
                "retrieval_scores": valid_scores
            }
            
        except Exception as e:
            print(f"Error in RAG answer_question: {e}")
            return {"answer": f"Error generating answer: {str(e)}", "confidence": 0}
    
    def debug_retrieval(self, question: str) -> dict:
        """Debug method to see what documents are being retrieved"""
        if not self.documents:
            return {"error": "No documents loaded"}
        
        query_embedding = self.encoder.encode([question])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding.astype('float32'), k=5)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append({
                    "index": idx,
                    "score": score,
                    "content_preview": self.documents[idx][:200]
                })
        
        return {"retrieved_documents": results}
