import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # ✅ Fix Keras compatibility

from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
from typing import List
import re

class SimpleREALM:
    def __init__(self):
        try:
            # Set device and initialize models with memory optimization
            self.device = torch.device('cpu')  # Use CPU to avoid memory issues
            
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased', 
                                                  low_cpu_mem_usage=True).to(self.device)
            
            # Add QA pipeline for actual answer generation
            self.qa_pipeline = pipeline('question-answering', 
                                       model='distilbert-base-cased-distilled-squad',
                                       device=-1,  # Force CPU
                                       model_kwargs={"low_cpu_mem_usage": True})
            
            self.documents = []
            self.doc_embeddings = None
            
            print("SimpleREALM initialized successfully")
        except Exception as e:
            print(f"Error initializing SimpleREALM: {e}")
            raise
    
    def add_documents(self, docs: List[str]):
        """Add documents and create embeddings with improved validation"""
        if not docs:
            print("Warning: No documents provided")
            return
        
        # Filter and validate documents
        valid_docs = []
        for i, doc in enumerate(docs):
            if self._validate_document(doc):
                valid_docs.append(doc)
        
        if not valid_docs:
            print("Error: No valid documents found after filtering")
            return
        
        self.documents = valid_docs
        
        try:
            # Debug: Print document chunks overview (first 3 only)
            print(f"📄 REALM Document Chunks Preview (showing first 3):")
            for i, doc in enumerate(valid_docs[:3]):
                preview = doc[:150].replace('\n', ' ')
                print(f"Chunk {i}: {preview}...")
            
            # Batch processing for memory efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(valid_docs), batch_size):
                batch_docs = valid_docs[i:i+batch_size]
                
                # Process documents for embeddings (remove source tags)
                clean_docs = []
                for doc in batch_docs:
                    clean_doc = doc
                    if ']' in clean_doc:
                        clean_doc = clean_doc.split(']', 1)[1].strip()
                    clean_docs.append(clean_doc)
                
                # Encode documents (simulating REALM's document encoder)
                doc_inputs = self.tokenizer(clean_docs, return_tensors='pt', 
                                           padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    doc_outputs = self.model(**doc_inputs.to(self.device))
                    # Use CLS token as document representation
                    batch_embeddings = doc_outputs.last_hidden_state[:, 0, :].cpu()
                    all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            self.doc_embeddings = torch.cat(all_embeddings, dim=0)
            
            print(f"Successfully added {len(valid_docs)} documents to REALM system")
            
        except Exception as e:
            print(f"Error adding documents to REALM: {e}")
            raise
    
    def _validate_document(self, doc: str) -> bool:
        """Validate document content quality"""
        if not doc or len(doc.strip()) < 50:
            return False
        
        # Allow documents with source tags, but ensure they have real content
        content = doc
        if ']' in content:
            content = content.split(']', 1)[1].strip()
        
        if not content or len(content.strip()) < 30:
            return False
        
        # Ensure meaningful content
        words = content.split()
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
        """Answer question using REALM-style approach with comprehensive improvements"""
        print(f"🔍 DEBUG 1 - Question received: '{question}'")
        
        if len(self.documents) == 0:
            return {"answer": "No documents available", "confidence": 0}
        
        if not question.strip():
            return {"answer": "Please provide a valid question", "confidence": 0}
        
        try:
            # Encode query (simulating REALM's query encoder)
            query_inputs = self.tokenizer([question], return_tensors='pt', 
                                        padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                query_outputs = self.model(**query_inputs.to(self.device))
                query_embedding = query_outputs.last_hidden_state[:, 0, :].cpu()
            
            # Compute similarities (simulating REALM retrieval)
            similarities = torch.cosine_similarity(query_embedding, self.doc_embeddings)
            
            # Try multiple documents for better results
            top_k = min(3, len(self.documents))
            top_scores, top_indices = torch.topk(similarities, k=top_k)
            
            best_result = None
            best_confidence = 0
            
            for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
                doc_idx = idx.item()
                doc = self.documents[doc_idx]
                retrieval_score = score.item()
                
                print(f"🔍 DEBUG 2.{i} - Testing document {doc_idx} (score: {retrieval_score:.3f})")
                print(f"🔍 DEBUG 3.{i} - Doc preview: {doc[:150]}...")
                
                # Prepare context for QA (remove source tag)
                context_for_qa = doc
                if ']' in context_for_qa:
                    context_for_qa = context_for_qa.split(']', 1)[1].strip()
                
                context_for_qa = context_for_qa[:1500]
                print(f"🔍 DEBUG 4.{i} - Context length: {len(context_for_qa)} characters")
                
                try:
                    # Generate answer using QA pipeline
                    qa_result = self.qa_pipeline(question=question, context=context_for_qa)
                    print(f"🔍 DEBUG 5.{i} - QA result: {qa_result}")
                    
                    # Select best result based on QA confidence
                    if qa_result['score'] > best_confidence:
                        best_confidence = qa_result['score']
                        best_result = {
                            'qa_result': qa_result,
                            'doc': doc,
                            'doc_idx': doc_idx,
                            'retrieval_score': retrieval_score
                        }
                        print(f"✅ New best result with confidence: {qa_result['score']:.3f}")
                
                except Exception as e:
                    print(f"❌ Error with document {doc_idx}: {e}")
                    continue
            
            if not best_result or best_confidence < 0.05:
                return {
                    "answer": "Could not find relevant information in the documents",
                    "confidence": 0,
                    "retrieved_doc": "",
                    "retrieval_score": 0
                }
            
            # Extract source information
            source_info = ""
            if ']' in best_result['doc']:
                source_info = best_result['doc'].split(']')[0] + ']'
            else:
                source_info = f"[Document {best_result['doc_idx']+1}]"
            
            print(f"🔍 DEBUG 6 - Selected source: {source_info}")
            
            # Clean the answer
            cleaned_answer = self._clean_answer(best_result['qa_result']['answer'])
            
            # Add confidence indicator for transparency
            confidence_indicator = ""
            if best_confidence < 0.3:
                confidence_indicator = " [Low Confidence]"
            elif best_confidence > 0.7:
                confidence_indicator = " [High Confidence]"
            
            # Format the answer with source attribution
            if cleaned_answer == "Information not found in document":
                final_answer = cleaned_answer
            else:
                final_answer = f"Based on the retrieved context: {source_info} {cleaned_answer}{confidence_indicator}"
            
            print(f"🔍 DEBUG 7 - Final answer: {final_answer}")
            
            result = {
                "answer": final_answer,
                "confidence": best_confidence,
                "retrieved_doc": best_result['doc'],
                "retrieval_score": best_result['retrieval_score']
            }
            
            print(f"🔍 DEBUG 8 - Returning result with confidence: {best_confidence:.3f}")
            return result
            
        except Exception as e:
            error_msg = f"Error in REALM answer_question: {e}"
            print(f"❌ DEBUG ERROR - {error_msg}")
            import traceback
            print(f"❌ Full traceback: {traceback.format_exc()}")
            return {"answer": f"Error generating answer: {str(e)}", "confidence": 0}
    
    def __del__(self):
        """Cleanup method to free memory"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'doc_embeddings'):
                del self.doc_embeddings
            torch.cuda.empty_cache()  # Clear any GPU cache if used
        except:
            pass
