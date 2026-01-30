"""
Complete RAG (Retrieval Augmented Generation) Pipeline Example
Demonstrates: Ingestion, Chunking, Embeddings, Retrieval, Re-Ranking, 
Prompting, Guardrails, and Evaluation
"""

import numpy as np
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
import os


# ==============================================================================
# 1. DOCUMENT INGESTION
# ==============================================================================

class DocumentIngestion:
    """Handles loading and preprocessing of documents from various sources"""
    
    @staticmethod
    def load_documents(sources: List[str]) -> List[Dict[str, str]]:
        """
        Load documents from different sources (files, APIs, databases, etc.)
        In production: PDF parsing, OCR, web scraping, API calls, etc.
        """
        documents = []
        for idx, source in enumerate(sources):
            documents.append({
                'id': f'doc_{idx}',
                'content': source,
                'metadata': {'source': f'source_{idx}', 'type': 'text'}
            })
        return documents


# ==============================================================================
# 2. CHUNKING STRATEGIES
# ==============================================================================

class Chunker:
    """Different chunking strategies for splitting documents"""
    
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 500, 
                           overlap: int = 50) -> List[str]:
        """
        Simple fixed-size chunking with overlap
        - chunk_size: characters per chunk
        - overlap: characters to overlap between chunks (preserves context)
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
    
    @staticmethod
    def semantic_chunking(text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Chunk by semantic boundaries (paragraphs, sentences)
        Better for maintaining context coherence
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def sentence_window_chunking(text: str, sentences_per_chunk: int = 3) -> List[Dict]:
        """
        Chunk by sentence windows, but store larger context
        Good for: retrieving precise info while maintaining context
        """
        # Simple sentence splitting (use spacy/nltk in production)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            # Store small chunk for embedding, but keep larger context
            context_start = max(0, i - 1)
            context_end = min(len(sentences), i + sentences_per_chunk + 1)
            
            chunks.append({
                'chunk': ' '.join(chunk_sentences),
                'context': ' '.join(sentences[context_start:context_end])
            })
        
        return chunks


# ==============================================================================
# 3. EMBEDDINGS
# ==============================================================================

class EmbeddingModel:
    """
    Handles text-to-vector conversion
    In production: Use OpenAI embeddings, sentence-transformers, or Cohere
    """
    
    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        self.vocab = {}
        self.vocab_size = 0
        self.embedding_dim = 100  # Fixed embedding dimension
    
    def simple_bow_embedding(self, text: str) -> np.ndarray:
        """
        Simple Bag-of-Words embedding (for demonstration)
        In production: Use proper embedding models like:
        - OpenAI text-embedding-3-large
        - sentence-transformers/all-mpnet-base-v2
        - Cohere embed-english-v3.0
        """
        # Tokenize
        words = text.lower().split()
        
        # Build vocabulary
        for word in words:
            if word not in self.vocab and len(self.vocab) < self.embedding_dim:
                self.vocab[word] = len(self.vocab)
        
        # Create vector with fixed dimension
        vector = np.zeros(self.embedding_dim)
        for word in words:
            if word in self.vocab:
                vector[self.vocab[word]] += 1
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently"""
        return np.array([self.simple_bow_embedding(text) for text in texts])


# ==============================================================================
# 4. VECTOR STORAGE & RETRIEVAL
# ==============================================================================

@dataclass
class Document:
    """Document with embedding"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict


class VectorStore:
    """
    Simple in-memory vector store
    In production: Use Pinecone, Weaviate, Qdrant, ChromaDB, FAISS, etc.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the store"""
        self.documents.extend(documents)
    
    def similarity_search(self, query_embedding: np.ndarray, 
                         top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k most similar documents using cosine similarity
        
        In production, consider:
        - Approximate Nearest Neighbor (ANN) algorithms for scale
        - Hybrid search (combining dense + sparse retrieval)
        - Metadata filtering
        """
        if not self.documents:
            return []
        
        # Get all embeddings
        doc_embeddings = np.array([doc.embedding for doc in self.documents])
        
        # Compute similarities
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            doc_embeddings
        )[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return documents with scores
        results = [
            (self.documents[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results


# ==============================================================================
# 5. RE-RANKING
# ==============================================================================

class ReRanker:
    """
    Re-rank retrieved documents for better relevance
    """
    
    @staticmethod
    def cross_encoder_rerank(query: str, documents: List[Tuple[Document, float]], 
                            top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Re-rank using cross-encoder approach
        In production: Use models like cross-encoder/ms-marco-MiniLM-L-12-v2
        
        This is a dummy implementation - real cross-encoders jointly encode
        query + document for better relevance scoring
        """
        # Simple re-ranking based on keyword overlap (demonstration only)
        query_words = set(query.lower().split())
        
        scored_docs = []
        for doc, initial_score in documents:
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words & doc_words)
            # Combine initial retrieval score with overlap score
            rerank_score = initial_score * 0.7 + (overlap / len(query_words)) * 0.3
            scored_docs.append((doc, rerank_score))
        
        # Sort by new scores
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]
    
    @staticmethod
    def diversity_rerank(documents: List[Tuple[Document, float]], 
                        lambda_param: float = 0.5) -> List[Tuple[Document, float]]:
        """
        Maximal Marginal Relevance (MMR) for diversity
        Balances relevance with diversity to avoid redundant results
        """
        if len(documents) <= 1:
            return documents
        
        selected = [documents[0]]
        remaining = documents[1:]
        
        while remaining and len(selected) < len(documents):
            mmr_scores = []
            
            for doc, rel_score in remaining:
                # Compute similarity to already selected documents
                max_sim = max(
                    cosine_similarity(
                        doc.embedding.reshape(1, -1),
                        sel_doc.embedding.reshape(1, -1)
                    )[0][0]
                    for sel_doc, _ in selected
                )
                
                # MMR score = relevance - lambda * max_similarity
                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim
                mmr_scores.append((doc, mmr_score))
            
            # Select document with highest MMR score
            best_doc = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_doc)
            remaining = [(d, s) for d, s in remaining if d.id != best_doc[0].id]
        
        return selected


# ==============================================================================
# 6. PROMPT CONSTRUCTION
# ==============================================================================

class PromptBuilder:
    """Build prompts with retrieved context"""
    
    @staticmethod
    def build_rag_prompt(query: str, contexts: List[str], 
                        system_prompt: str = None) -> Tuple[str, str]:
        """
        Construct RAG prompt with retrieved context
        
        Key considerations:
        - Context ordering (most relevant first)
        - Token limits
        - Clear separation of context from query
        - Attribution instructions
        """
        
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which context passages you used."""
        
        # Build context section
        context_text = "\n\n".join([
            f"[Context {i+1}]:\n{ctx}" 
            for i, ctx in enumerate(contexts)
        ])
        
        # Build user prompt
        user_prompt = f"""Based on the following context, please answer the question.

{context_text}

Question: {query}

Please provide a clear answer based on the context above."""
        
        return system_prompt, user_prompt


# ==============================================================================
# 7. GUARDRAILS
# ==============================================================================

class Guardrails:
    """
    Safety and quality checks for RAG systems
    """
    
    @staticmethod
    def check_input_safety(query: str) -> Tuple[bool, str]:
        """
        Check if input query is safe
        In production: Use moderation APIs, custom classifiers
        """
        # Simple keyword-based check (use ML models in production)
        unsafe_patterns = ['ignore previous', 'disregard instructions', 'system prompt']
        
        query_lower = query.lower()
        for pattern in unsafe_patterns:
            if pattern in query_lower:
                return False, f"Query contains potentially unsafe pattern: {pattern}"
        
        return True, "Safe"
    
    @staticmethod
    def check_output_quality(response: str, contexts: List[str]) -> Dict[str, any]:
        """
        Verify output quality
        - Relevance to context
        - No hallucination
        - Appropriate length
        """
        checks = {
            'has_content': len(response.strip()) > 0,
            'reasonable_length': 10 < len(response) < 5000,
            'uses_context': any(
                snippet in response 
                for context in contexts 
                for snippet in context.split()[:10]  # Check first words
            )
        }
        
        checks['passed'] = all(checks.values())
        return checks
    
    @staticmethod
    def check_citation_grounding(response: str, contexts: List[str]) -> bool:
        """
        Verify that claims are grounded in provided context
        In production: Use NLI models or LLM-as-judge
        """
        # Simple check: does response reference context numbers?
        has_citations = bool(re.search(r'\[Context \d+\]', response))
        return has_citations


# ==============================================================================
# 8. EVALUATION METRICS
# ==============================================================================

class RAGEvaluator:
    """
    Evaluate RAG system performance
    """
    
    @staticmethod
    def evaluate_retrieval(retrieved_docs: List[Document], 
                          ground_truth_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        - Precision@k
        - Recall@k
        - MRR (Mean Reciprocal Rank)
        """
        retrieved_ids = [doc.id for doc in retrieved_docs]
        relevant_retrieved = set(retrieved_ids) & set(ground_truth_ids)
        
        metrics = {
            'precision': len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0,
            'recall': len(relevant_retrieved) / len(ground_truth_ids) if ground_truth_ids else 0,
        }
        
        # MRR: position of first relevant document
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in ground_truth_ids:
                metrics['mrr'] = 1.0 / (i + 1)
                break
        else:
            metrics['mrr'] = 0.0
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        return metrics
    
    @staticmethod
    def evaluate_generation(response: str, reference: str) -> Dict[str, float]:
        """
        Evaluate generation quality
        In production: Use ROUGE, BLEU, BERTScore, or LLM-as-judge
        """
        # Simple word overlap metric (use proper metrics in production)
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        
        overlap = len(response_words & reference_words)
        
        metrics = {
            'word_overlap': overlap / len(reference_words) if reference_words else 0,
            'length_ratio': len(response) / len(reference) if reference else 0
        }
        
        return metrics


# ==============================================================================
# 9. COMPLETE RAG PIPELINE
# ==============================================================================

class RAGPipeline:
    """
    Complete RAG pipeline integrating all components
    """
    
    def __init__(self, anthropic_api_key: str = None):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.chunker = Chunker()
        self.reranker = ReRanker()
        self.prompt_builder = PromptBuilder()
        self.guardrails = Guardrails()
        self.evaluator = RAGEvaluator()
        
        # Initialize Anthropic client if API key provided
        self.client = None
        if anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
    
    def index_documents(self, documents: List[str], chunk_strategy: str = 'semantic'):
        """
        Index documents: chunk, embed, store
        """
        print("🔧 Indexing documents...")
        
        # 1. Chunk documents
        all_chunks = []
        for doc_id, doc in enumerate(documents):
            if chunk_strategy == 'semantic':
                chunks = self.chunker.semantic_chunking(doc)
            elif chunk_strategy == 'fixed':
                chunks = self.chunker.fixed_size_chunking(doc)
            else:
                chunks = [doc]
            
            for chunk_id, chunk in enumerate(chunks):
                all_chunks.append({
                    'id': f'doc{doc_id}_chunk{chunk_id}',
                    'content': chunk,
                    'metadata': {'doc_id': doc_id, 'chunk_id': chunk_id}
                })
        
        print(f"  ✓ Created {len(all_chunks)} chunks")
        
        # 2. Generate embeddings
        chunk_texts = [c['content'] for c in all_chunks]
        embeddings = self.embedding_model.embed_batch(chunk_texts)
        print(f"  ✓ Generated embeddings")
        
        # 3. Store in vector database
        docs = [
            Document(
                id=chunk['id'],
                content=chunk['content'],
                embedding=emb,
                metadata=chunk['metadata']
            )
            for chunk, emb in zip(all_chunks, embeddings)
        ]
        self.vector_store.add_documents(docs)
        print(f"  ✓ Stored {len(docs)} documents in vector store\n")
    
    def query(self, query: str, top_k: int = 3, use_reranking: bool = True) -> Dict:
        """
        Execute RAG query pipeline
        """
        print(f"🔍 Query: {query}\n")
        
        # 1. Input guardrails
        is_safe, safety_msg = self.guardrails.check_input_safety(query)
        if not is_safe:
            return {'error': f'Query rejected: {safety_msg}'}
        
        # 2. Embed query
        query_embedding = self.embedding_model.simple_bow_embedding(query)
        print(f"  ✓ Generated query embedding")
        
        # 3. Retrieve relevant documents
        retrieved = self.vector_store.similarity_search(query_embedding, top_k=top_k*2)
        print(f"  ✓ Retrieved {len(retrieved)} documents (scores: {[f'{s:.3f}' for _, s in retrieved[:3]]})")
        
        # 4. Re-rank (optional)
        if use_reranking and len(retrieved) > top_k:
            retrieved = self.reranker.cross_encoder_rerank(query, retrieved, top_k=top_k)
            print(f"  ✓ Re-ranked to top {top_k} documents")
        
        # 5. Build prompt
        contexts = [doc.content for doc, _ in retrieved[:top_k]]
        system_prompt, user_prompt = self.prompt_builder.build_rag_prompt(query, contexts)
        
        # 6. Generate response (using Claude if available, otherwise return context)
        if self.client:
            print(f"  ✓ Generating response with Claude...\n")
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            response = message.content[0].text
        else:
            response = f"[No LLM configured. Retrieved contexts would be used here]\n\n{user_prompt}"
        
        # 7. Output guardrails
        quality_checks = self.guardrails.check_output_quality(response, contexts)
        print(f"  ✓ Quality checks: {quality_checks}\n")
        
        return {
            'query': query,
            'response': response,
            'contexts': contexts,
            'retrieved_docs': [(doc.id, score) for doc, score in retrieved[:top_k]],
            'quality_checks': quality_checks
        }


# ==============================================================================
# 10. DEMONSTRATION
# ==============================================================================

def main():
    """
    Demonstrate the complete RAG pipeline
    """
    print("=" * 80)
    print("RAG PIPELINE DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Sample documents (knowledge base)
    documents = [
        """
        Actuarial pricing is the process of determining insurance premiums. 
        It involves analyzing historical loss data, expense ratios, and profit margins.
        Actuaries use statistical models to predict future claims and set appropriate prices.
        The key components are: loss costs, expenses, profit margin, and risk loading.
        """,
        
        """
        Reinsurance is insurance for insurance companies. It allows primary insurers 
        to transfer risk to reinsurers. Common types include quota share, surplus share,
        and excess of loss treaties. Reinsurance pricing requires sophisticated modeling
        of catastrophic events and accumulation scenarios. Swiss Re and Munich Re are 
        major global reinsurers based in Switzerland.
        """,
        
        """
        Machine learning in actuarial science has grown significantly. Techniques like
        gradient boosting, neural networks, and GLMs are used for pricing and reserving.
        However, interpretability remains crucial for regulatory compliance. Actuaries
        must balance model performance with explainability. Common tools include Python,
        R, and specialized actuarial software.
        """,
        
        """
        The Swiss insurance market is highly developed and regulated by FINMA.
        Switzerland is home to major insurance and reinsurance companies. The Solvency
        requirements are strict, requiring companies to hold adequate capital. Swiss Re
        is one of the world's largest reinsurers, headquartered in Zurich.
        """
    ]
    
    # Initialize RAG pipeline
    # Note: Set ANTHROPIC_API_KEY environment variable to use actual Claude responses
    api_key = os.getenv('ANTHROPIC_API_KEY')
    rag = RAGPipeline(anthropic_api_key=api_key)
    
    # Index documents
    rag.index_documents(documents, chunk_strategy='semantic')
    
    # Query examples
    queries = [
        "What is reinsurance and who are the major players in Switzerland?",
        "How is machine learning used in actuarial pricing?",
        "What are the key components of insurance pricing?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"EXAMPLE QUERY {i}")
        print('─' * 80)
        
        result = rag.query(query, top_k=3, use_reranking=True)
        
        print(f"📝 RESPONSE:")
        print(result['response'])
        print(f"\n📚 RETRIEVED CONTEXTS:")
        for j, ctx in enumerate(result['contexts'], 1):
            print(f"  [{j}] {ctx[:100]}...")
        print()
    
    # Evaluation example
    print("\n" + "=" * 80)
    print("EVALUATION EXAMPLE")
    print("=" * 80 + "\n")
    
    # Simulate evaluation with ground truth
    retrieved_doc_ids = [doc_id for doc_id, _ in result['retrieved_docs']]
    ground_truth_ids = ['doc1_chunk0', 'doc3_chunk0']  # Assuming these are relevant
    
    retrieval_metrics = rag.evaluator.evaluate_retrieval(
        [doc for doc in rag.vector_store.documents if doc.id in retrieved_doc_ids],
        ground_truth_ids
    )
    
    print("📊 Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. CHUNKING: Critical for context window - balance granularity vs. coherence
2. EMBEDDINGS: Quality determines retrieval accuracy - use good models
3. RETRIEVAL: Fast similarity search - use ANN algorithms at scale
4. RE-RANKING: Improves precision - cross-encoders or MMR for diversity
5. PROMPTING: Clear instructions and context separation crucial
6. GUARDRAILS: Essential for production - input/output validation
7. EVALUATION: Continuous monitoring - track retrieval and generation quality

Production considerations:
- Vector databases: Pinecone, Weaviate, Qdrant for scale
- Embedding models: OpenAI, Cohere, or open-source sentence-transformers
- Chunking: Experiment with different strategies for your domain
- Hybrid search: Combine dense + sparse (BM25) retrieval
- Monitoring: Log queries, track metrics, iterate on prompts
- Caching: Cache embeddings and frequent queries
    """)


if __name__ == "__main__":
    main()
