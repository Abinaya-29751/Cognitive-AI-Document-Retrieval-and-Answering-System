import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # ✅ Set BEFORE any imports

import streamlit as st
from simple_rag import SimpleRAG
from simple_realm import SimpleREALM
from utils import (extract_text_from_pdf, chunk_text, process_multiple_files, 
                   process_website_urls, extract_text_from_website)

def main():
    st.title("🤖 RAG vs REALM with Multiple Document Types")
    st.markdown("Upload PDFs, text files, CSV, Word docs, or provide website URLs!")

    # Initialize systems
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAG()
        st.session_state.realm_system = SimpleREALM()
        st.session_state.documents_loaded = False

    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # Input method selection
        input_method = st.selectbox(
            "Choose input method:",
            [
                "📁 Upload Files (PDF, TXT, CSV, DOCX)",
                "🌐 Website URLs", 
                "📚 Use Sample PDFs",
                "🔗 Mixed Sources"
            ]
        )
        
        documents = []
        
        if input_method == "📁 Upload Files (PDF, TXT, CSV, DOCX)":
            st.subheader("File Upload")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=['pdf', 'txt', 'csv', 'docx'],
                accept_multiple_files=True,
                help="Supported: PDF, TXT, CSV, DOCX files"
            )
            
            if uploaded_files:
                with st.spinner("Processing uploaded files..."):
                    documents = process_multiple_files(uploaded_files)
                if documents:
                    st.success(f"✅ Loaded {len(documents)} chunks from {len(uploaded_files)} files")
                    
        elif input_method == "🌐 Website URLs":
            st.subheader("Website URLs")
            st.info("Enter one URL per line")
            
            urls_text = st.text_area(
                "Enter website URLs:",
                placeholder="https://example.com\nhttps://another-site.com",
                height=100
            )
            
            if st.button("Process URLs") and urls_text.strip():
                urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
                
                if urls:
                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        documents = process_website_urls(urls)
                    if documents:
                        st.success(f"✅ Loaded {len(documents)} chunks from {len(urls)} URLs")
                        
        elif input_method == "🔗 Mixed Sources":
            st.subheader("Mixed Sources")
            
            # File upload section
            st.write("**Upload Files:**")
            uploaded_files = st.file_uploader(
                "Files",
                type=['pdf', 'txt', 'csv', 'docx'],
                accept_multiple_files=True,
                key="mixed_files"
            )
            
            # URL section
            st.write("**Website URLs:**")
            urls_text = st.text_area(
                "URLs (one per line):",
                placeholder="https://example.com",
                height=80,
                key="mixed_urls"
            )
            
            if st.button("Process All Sources"):
                all_documents = []
                
                # Process files
                if uploaded_files:
                    with st.spinner("Processing files..."):
                        file_docs = process_multiple_files(uploaded_files)
                        all_documents.extend(file_docs)
                
                # Process URLs
                if urls_text.strip():
                    urls = [url.strip() for url in urls_text.strip().split('\n') if url.strip()]
                    if urls:
                        with st.spinner("Processing URLs..."):
                            url_docs = process_website_urls(urls)
                            all_documents.extend(url_docs)
                
                documents = all_documents
                if documents:
                    st.success(f"✅ Loaded {len(documents)} total chunks from mixed sources")
                    
        else: # Sample PDFs
            st.info("Using sample PDF documents")
            sample_dir = "sample_pdfs"
            
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.pdf')]
                if sample_files:
                    st.write(f"Found {len(sample_files)} sample PDFs:")
                    for file in sample_files:
                        st.write(f"• {file}")
                        
                    if st.button("Load Sample PDFs"):
                        documents = []
                        for filename in sample_files:
                            filepath = os.path.join(sample_dir, filename)
                            with open(filepath, 'rb') as file:
                                text = extract_text_from_pdf(file)
                                if text:
                                    chunks = chunk_text(text)
                                    labeled_chunks = [f"[Source: {filename}]\n{chunk}" for chunk in chunks]
                                    documents.extend(labeled_chunks)
                        
                        if documents:
                            st.success(f"✅ Loaded {len(documents)} chunks from sample PDFs")
                else:
                    st.warning("No sample PDFs found")
            else:
                st.warning("sample_pdfs folder not found")

    # Load documents into systems
    if documents and not st.session_state.documents_loaded:
        with st.spinner("Loading documents into RAG and REALM systems..."):
            st.session_state.rag_system.add_documents(documents)
            st.session_state.realm_system.add_documents(documents)
            st.session_state.documents_loaded = True
            st.session_state.current_documents = documents
        st.success("🎉 Both systems ready!")

    # Document statistics
    if st.session_state.documents_loaded:
        st.subheader("📊 Document Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        total_words = sum(len(doc.split()) for doc in st.session_state.current_documents)
        avg_chunk_size = total_words / len(st.session_state.current_documents)
        
        with col1:
            st.metric("Total Chunks", len(st.session_state.current_documents))
        with col2:
            st.metric("Total Words", f"{total_words:,}")
        with col3:
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} words")

    # Main Q&A interface
    if st.session_state.documents_loaded:
        st.header("❓ Ask Questions About Your Documents")
        
        # # Quick suggestions
        # st.subheader("💡 Quick Question Ideas:")
        # col1, col2, col3, col4 = st.columns(4)
        
        # with col1:
        #     if st.button("📋 Summary"):
        #         st.session_state.suggested_q = "What are the main points discussed?"
        # with col2:
        #     if st.button("📊 Data/Stats"):
        #         st.session_state.suggested_q = "What numbers or statistics are mentioned?"
        # with col3:
        #     if st.button("👥 People"):
        #         st.session_state.suggested_q = "Who are the key people mentioned?"
        # with col4:
        #     if st.button("🏢 Organizations"):
        #         st.session_state.suggested_q = "What organizations or companies are discussed?"
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What is the main topic discussed in the documents?",
            value=st.session_state.get('suggested_q', '')
        )
        
        if st.button("Clear Suggestion") and 'suggested_q' in st.session_state:
            del st.session_state.suggested_q
            st.experimental_rerun()

        # Get answers
        if st.button("🚀 Get Answers", type="primary") and question:
            with st.spinner("Generating answers..."):
                # Baseline answer
                baseline_answer = "I don't have specific information about this topic without accessing the documents."
                
                # RAG answer
                rag_result = st.session_state.rag_system.answer_question(question)
                
                # REALM answer
                realm_result = st.session_state.realm_system.answer_question(question)

            # Display results
            st.header("📋 Results Comparison")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🔍 Side-by-Side", "📊 Detailed Analysis", "📚 Sources"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🔍 Baseline (No Retrieval)")
                    st.write(baseline_answer)
                    st.caption("Generic response without document access")
                
                with col2:
                    st.subheader("🎯 RAG Answer")
                    st.write(rag_result['answer'])
                    st.metric("Confidence", f"{rag_result['confidence']:.3f}")
                
                with col3:
                    st.subheader("🧠 REALM Answer")
                    st.write(realm_result['answer'])
                    st.metric("Confidence", f"{realm_result['confidence']:.3f}")
            
            with tab2:
                st.subheader("📊 Quality Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Baseline Length", f"{len(baseline_answer.split())} words")
                with col2:
                    rag_improvement = len(rag_result['answer'].split()) - len(baseline_answer.split())
                    st.metric("RAG Length", f"{len(rag_result['answer'].split())} words", 
                             delta=f"{rag_improvement:+d} words")
                with col3:
                    realm_improvement = len(realm_result['answer'].split()) - len(baseline_answer.split())
                    st.metric("REALM Length", f"{len(realm_result['answer'].split())} words",
                             delta=f"{realm_improvement:+d} words")
                with col4:
                    st.metric("Sources Retrieved", len(rag_result.get('retrieved_docs', [])))
            
            with tab3:
                st.subheader("📚 Retrieved Sources")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**RAG Retrieved Documents:**")
                    for i, doc in enumerate(rag_result.get('retrieved_docs', [])):
                        score = rag_result.get('retrieval_scores', [0])[i]
                        st.expander(f"RAG Source {i+1} (Score: {score:.3f})").write(doc[:300] + "...")
                
                with col2:
                    st.write("**REALM Retrieved Document:**")
                    if 'retrieved_doc' in realm_result:
                        score = realm_result.get('retrieval_score', 0)
                        st.expander(f"REALM Source (Score: {score:.3f})").write(realm_result['retrieved_doc'][:300] + "...")

    else:
        # Instructions when no documents loaded
        st.info("👆 Please upload documents, provide URLs, or load sample PDFs to start asking questions")
        
        st.header("🚀 How to Use This Enhanced System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Supported Input Types")
            st.markdown("""
            - **📄 PDF Files**: Research papers, reports, books
            - **📝 Text Files**: Plain text documents  
            - **📊 CSV Files**: Data tables and spreadsheets
            - **📃 Word Documents**: DOCX format documents
            - **🌐 Websites**: Any public webpage content
            - **🔗 Mixed Sources**: Combine multiple types
            """)
        
        with col2:
            st.subheader("❓ Question Examples")
            st.markdown("""
            - **Summary**: "What are the main findings?"
            - **Data**: "What statistics are mentioned?"
            - **People**: "Who are the key authors?"
            - **Comparison**: "How do the methods differ?"
            - **Specific**: "What is the accuracy rate?"
            - **Context**: "What problem does this solve?"
            """)

if __name__ == "__main__":
    main()
