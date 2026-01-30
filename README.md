# RAG Architecture - Detailed Explanation

## Overview
RAG (Retrieval Augmented Generation) combines information retrieval with large language models to provide accurate, grounded responses based on external knowledge bases.

## 1. Document Ingestion

**Purpose:** Load and preprocess documents from various sources into a standardized format.

**Key Considerations:**
- **Format handling:** PDFs, Word docs, HTML, databases, APIs
- **Preprocessing:** OCR for scanned documents, HTML cleaning, metadata extraction
- **Scalability:** Batch processing, parallel ingestion
- **Data validation:** Check for corrupted files, encoding issues

**Production Tools:**
- PyPDF2, pdfplumber (PDF)
- Beautiful Soup (HTML)
- python-docx (Word)
- Apache Tika (multi-format)

**Example:**
```python
def ingest_pdf(file_path: str) -> str:
    import PyPDF2
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
```

---

## 2. Chunking Strategies

**Purpose:** Split documents into smaller pieces that fit within context windows and improve retrieval precision.

### Strategy Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed-size** | Simple, predictable | May break semantic units | Uniform content |
| **Semantic** | Preserves meaning | Variable sizes | Articles, reports |
| **Sentence-window** | Precise retrieval | More complex | Q&A, fact extraction |
| **Recursive** | Hierarchical context | Overhead | Long documents |

### Key Parameters
- **Chunk size:** 200-1000 tokens (balance: precision vs context)
- **Overlap:** 10-20% (prevents information loss at boundaries)
- **Separators:** Paragraphs > sentences > words

**Practical Considerations:**
- Small chunks (200-400 tokens): Better for specific facts, Q&A
- Large chunks (800-1000 tokens): Better for summarization, context-heavy tasks
- Overlap prevents "orphaned" information at chunk boundaries

**Advanced Technique - Sentence Window:**
```python
# Store small chunk for embedding (precise retrieval)
chunk = "Reinsurance is insurance for insurance companies."

# But return larger context window to LLM (better understanding)
context = "The insurance market has evolved significantly. 
           Reinsurance is insurance for insurance companies. 
           It allows risk transfer across the industry."
```

---

## 3. Embeddings

**Purpose:** Convert text into dense vector representations that capture semantic meaning.

### Embedding Model Selection

**OpenAI (Proprietary):**
- `text-embedding-3-large`: 3072 dimensions, best quality
- `text-embedding-3-small`: 1536 dimensions, faster/cheaper
- Cost: ~$0.13 per 1M tokens (small)

**Open Source:**
- `all-mpnet-base-v2`: 768 dim, good general purpose
- `all-MiniLM-L6-v2`: 384 dim, fast and lightweight
- `e5-large-v2`: 1024 dim, strong performance
- Cost: Self-hosted (GPU costs)

**Domain-Specific:**
- `msmarco-distilbert`: Information retrieval
- `gte-large`: General text embeddings
- Fine-tune on your domain for best results

**Key Metrics:**
- **Dimensionality:** Higher = more expressive, but slower
- **Max sequence length:** Typically 512 tokens
- **Normalization:** L2 normalization for cosine similarity

**Production Code:**
```python
from sentence_transformers import SentenceTransformer

# Initialize model (cached after first load)
model = SentenceTransformer('all-mpnet-base-v2')

# Batch embedding (much faster than one-by-one)
chunks = ["text 1", "text 2", "text 3", ...]
embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True  # For cosine similarity
)
```

---

## 4. Vector Storage & Retrieval

**Purpose:** Efficiently store and search high-dimensional vectors.

### Vector Database Options

| Database | Type | Best For | Key Feature |
|----------|------|----------|-------------|
| **Pinecone** | Managed | Production, scale | Fully managed, serverless |
| **Weaviate** | Hybrid | Multi-modal | Built-in vectorization |
| **Qdrant** | Self-hosted | Privacy, control | Rust-based, very fast |
| **ChromaDB** | Embedded | Development | Easy setup, local-first |
| **FAISS** | Library | Research | Meta-developed, flexible |
| **Milvus** | Self-hosted | Large scale | Distributed architecture |

### Similarity Search

**Distance Metrics:**
- **Cosine Similarity:** Most common, measures angle between vectors
- **Euclidean (L2):** Actual distance in vector space
- **Dot Product:** When vectors are normalized, equivalent to cosine

**Search Types:**
- **k-NN (k-Nearest Neighbors):** Exact search, slow at scale
- **ANN (Approximate Nearest Neighbors):** Fast, 95%+ recall
  - HNSW (Hierarchical Navigable Small World): Popular, accurate
  - IVF (Inverted File Index): Memory efficient
  - Product Quantization: Compression for large datasets

**Hybrid Search:**
Combine dense (semantic) + sparse (keyword/BM25) retrieval:
```python
# Dense retrieval (semantic)
semantic_results = vector_search(query_embedding)

# Sparse retrieval (keyword)
keyword_results = bm25_search(query_text)

# Combine scores (e.g., RRF - Reciprocal Rank Fusion)
final_results = combine_results(semantic_results, keyword_results)
```

### Filtering & Metadata
```python
# Search with metadata filtering
results = vector_store.search(
    query_embedding,
    filter={
        "doc_type": "policy",
        "date": {"$gte": "2024-01-01"},
        "department": "actuarial"
    },
    top_k=10
)
```

---

## 5. Re-Ranking

**Purpose:** Improve relevance of retrieved documents using more sophisticated (but slower) models.

### Two-Stage Retrieval Architecture
1. **First stage:** Fast retrieval (get top 100-1000)
2. **Second stage:** Slow re-ranking (refine to top 3-10)

### Re-Ranking Methods

**1. Cross-Encoder Models:**
- Jointly encode query + document
- Much more accurate than bi-encoders
- Too slow for initial retrieval (quadratic complexity)

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Score each query-document pair
pairs = [[query, doc.text] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by score
reranked = sorted(zip(retrieved_docs, scores), 
                  key=lambda x: x[1], reverse=True)
```

**2. Maximal Marginal Relevance (MMR):**
- Balances relevance with diversity
- Prevents redundant results
- Formula: `MMR = λ * relevance - (1-λ) * max_similarity_to_selected`

**3. LLM-based Re-ranking:**
- Use LLM to score relevance: "Rate 0-10: How relevant is this document?"
- Most accurate but expensive
- Use for small result sets only

**When to Use:**
- Large retrieval pool (>20 documents)
- Quality is critical
- Acceptable latency trade-off (~100-500ms extra)

---

## 6. Prompt Engineering for RAG

**Purpose:** Structure retrieved context and query for optimal LLM performance.

### Prompt Template Structure

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions 
based on the provided context. 

IMPORTANT RULES:
1. Only use information from the provided context
2. If context lacks information, say "I don't have enough information"
3. Cite which context passages support your answer
4. Do not make up information or use prior knowledge"""

USER_PROMPT = """
Context Information:
-------------------
[Context 1]: {context_1}
[Context 2]: {context_2}
[Context 3]: {context_3}

Question:
---------
{user_query}

Instructions:
-------------
Answer the question using ONLY the context above. 
Cite specific context numbers [Context N] in your response.
"""
```

### Advanced Techniques

**1. Chain-of-Thought Prompting:**
```python
"Let's approach this step by step:
1. First, identify relevant information from context
2. Then, synthesize the information
3. Finally, provide a clear answer with citations"
```

**2. Few-Shot Examples:**
```python
"""Examples of good answers:

Q: What is reinsurance?
Context 1: Reinsurance is insurance for insurance companies...
A: According to Context 1, reinsurance is insurance for insurance companies, 
   allowing them to transfer risk.

Q: What is the capital of Atlantis?
Context 1: Paris is the capital of France...
A: I don't have information about Atlantis in the provided context.

Now answer this question: {query}"""
```

**3. Context Ordering:**
- Most relevant first (LLMs pay more attention to start)
- Or most relevant last (recency bias)
- Test both for your use case

**4. Token Budget Management:**
```python
def fit_context_to_budget(contexts, max_tokens=3000):
    """Truncate or select contexts to fit token budget"""
    from tiktoken import encoding_for_model
    
    enc = encoding_for_model("gpt-4")
    included = []
    token_count = 0
    
    for ctx in contexts:
        ctx_tokens = len(enc.encode(ctx))
        if token_count + ctx_tokens <= max_tokens:
            included.append(ctx)
            token_count += ctx_tokens
        else:
            break
    
    return included
```

---

## 7. Guardrails

**Purpose:** Ensure system safety, reliability, and quality.

### Input Guardrails

**1. Prompt Injection Detection:**
```python
def detect_injection(query: str) -> bool:
    """Detect attempts to manipulate the system"""
    injection_patterns = [
        r"ignore (previous|above) instructions",
        r"you are now",
        r"disregard (all|any) (previous|above)",
        r"new (instructions|rules|guidelines)",
        r"<\s*system\s*>",  # XML tag injection
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False
```

**2. Content Moderation:**
- Use OpenAI Moderation API
- Or custom classifiers for domain-specific content
- Block malicious, harmful, or inappropriate queries

**3. Rate Limiting:**
```python
from functools import wraps
from time import time

def rate_limit(max_per_minute=10):
    """Decorator to limit query rate"""
    calls = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            # Remove calls older than 1 minute
            calls[:] = [t for t in calls if now - t < 60]
            
            if len(calls) >= max_per_minute:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Output Guardrails

**1. Hallucination Detection:**
```python
def check_grounding(response: str, contexts: List[str]) -> float:
    """
    Use NLI (Natural Language Inference) to verify claims
    Returns: grounding score 0-1
    """
    from transformers import pipeline
    
    nli = pipeline("text-classification", 
                   model="microsoft/deberta-v3-base-mnli")
    
    sentences = sent_tokenize(response)
    grounded_count = 0
    
    for sent in sentences:
        # Check if sentence is entailed by any context
        max_score = 0
        for ctx in contexts:
            result = nli(f"{ctx} [SEP] {sent}")
            if result[0]['label'] == 'ENTAILMENT':
                max_score = max(max_score, result[0]['score'])
        
        if max_score > 0.8:
            grounded_count += 1
    
    return grounded_count / len(sentences) if sentences else 0
```

**2. Citation Validation:**
```python
def validate_citations(response: str, contexts: List[str]) -> bool:
    """Ensure citations point to actual contexts"""
    cited_indices = re.findall(r'\[Context (\d+)\]', response)
    return all(int(idx) <= len(contexts) for idx in cited_indices)
```

**3. Quality Checks:**
- Response length (too short = likely error)
- Coherence scoring
- Toxicity detection
- PII detection and masking

**4. Fallback Strategies:**
```python
def generate_with_fallback(query, contexts):
    """Try generation with fallback options"""
    try:
        # Primary: Full RAG pipeline
        response = rag_generate(query, contexts)
        
        if not passes_quality_checks(response):
            # Fallback 1: Retry with different prompt
            response = rag_generate(query, contexts, 
                                   prompt_template="conservative")
            
        if not passes_quality_checks(response):
            # Fallback 2: Return safe default
            response = "I couldn't generate a confident answer. 
                       Here are the relevant passages: ..."
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return fallback_response()
```

---

## 8. Evaluation

**Purpose:** Measure and continuously improve RAG system performance.

### Retrieval Metrics

**1. Precision@k:**
- Of top k retrieved docs, how many are relevant?
- Formula: `relevant_in_top_k / k`

**2. Recall@k:**
- Of all relevant docs, how many are in top k?
- Formula: `relevant_in_top_k / total_relevant`

**3. Mean Reciprocal Rank (MRR):**
- Average of 1/rank of first relevant document
- Formula: `1 / position_of_first_relevant`

**4. Normalized Discounted Cumulative Gain (NDCG):**
- Accounts for position and relevance grades
- Higher positions weighted more

**Example Evaluation:**
```python
def evaluate_retrieval(queries, ground_truth):
    """Evaluate retrieval quality"""
    metrics = {
        'precision@3': [],
        'recall@3': [],
        'mrr': [],
        'ndcg@5': []
    }
    
    for query, relevant_docs in zip(queries, ground_truth):
        retrieved = retriever.search(query, top_k=5)
        
        # Precision@3
        relevant_in_3 = sum(1 for d in retrieved[:3] 
                           if d.id in relevant_docs)
        metrics['precision@3'].append(relevant_in_3 / 3)
        
        # Recall@3
        metrics['recall@3'].append(
            relevant_in_3 / len(relevant_docs)
        )
        
        # MRR
        for i, doc in enumerate(retrieved):
            if doc.id in relevant_docs:
                metrics['mrr'].append(1 / (i + 1))
                break
        else:
            metrics['mrr'].append(0)
    
    # Average all metrics
    return {k: np.mean(v) for k, v in metrics.items()}
```

### Generation Metrics

**1. Automatic Metrics:**

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Measures n-gram overlap with reference
- ROUGE-1: unigram overlap
- ROUGE-L: longest common subsequence

**BLEU (Bilingual Evaluation Understudy):**
- Precision-oriented n-gram matching
- Better for translation, less for open-ended QA

**BERTScore:**
- Semantic similarity using BERT embeddings
- Better captures meaning vs exact words

```python
from bert_score import score as bert_score

def evaluate_generation(predictions, references):
    """Evaluate generated answers"""
    
    # BERTScore
    P, R, F1 = bert_score(predictions, references, 
                          lang="en", model_type="microsoft/deberta-xlarge-mnli")
    
    return {
        'bertscore_f1': F1.mean().item(),
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item()
    }
```

**2. LLM-as-Judge:**

Most accurate but expensive:

```python
def llm_judge_evaluation(question, context, answer, reference):
    """Use LLM to judge answer quality"""
    
    prompt = f"""
    Evaluate the quality of this answer on a scale of 1-10.
    
    Question: {question}
    Context: {context}
    Generated Answer: {answer}
    Reference Answer: {reference}
    
    Criteria:
    - Accuracy (compared to reference)
    - Relevance (addresses the question)
    - Grounding (uses only provided context)
    - Completeness (covers key points)
    
    Provide:
    1. Overall score (1-10)
    2. Brief justification
    3. Scores for each criterion
    """
    
    # Call LLM for evaluation
    evaluation = llm.generate(prompt)
    return parse_evaluation(evaluation)
```

**3. Human Evaluation:**

Gold standard but expensive:
- Relevance: Does it answer the question?
- Accuracy: Is the information correct?
- Fluency: Is it well-written?
- Groundedness: Is it supported by context?

Use sampling (evaluate 100-1000 examples, not all)

### End-to-End Metrics

**1. Context Relevance:**
- Are retrieved contexts actually relevant?
- Use LLM to judge: "On 1-5, how relevant is this context to the query?"

**2. Answer Relevance:**
- Does answer actually address the question?

**3. Faithfulness/Groundedness:**
- Is answer faithful to retrieved context?
- Critical for factual accuracy

**RAGAS Framework:**
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)
```

### Monitoring in Production

**1. Logging:**
```python
import logging
from datetime import datetime

def log_rag_query(query, retrieved, response, latency):
    """Log every query for analysis"""
    log_entry = {
        'timestamp': datetime.now(),
        'query': query,
        'retrieved_doc_ids': [d.id for d in retrieved],
        'response_length': len(response),
        'latency_ms': latency,
        'num_retrieved': len(retrieved)
    }
    
    logger.info(json.dumps(log_entry))
    
    # Also write to analytics database
    analytics_db.insert(log_entry)
```

**2. Dashboard Metrics:**
- Query volume over time
- Average latency (p50, p95, p99)
- Error rates
- Top queries
- Retrieval success rate
- User satisfaction (thumbs up/down)

**3. A/B Testing:**
```python
def ab_test_chunking_strategy(query):
    """Test different configurations"""
    user_id = hash(query) % 100
    
    if user_id < 50:  # Group A
        docs = retriever_v1.search(query)
        group = 'baseline'
    else:  # Group B
        docs = retriever_v2.search(query)
        group = 'new_chunking'
    
    log_ab_test(query, group, docs)
    return docs
```

**4. Continuous Evaluation:**
- Run automated eval suite nightly
- Track metric trends
- Alert on degradation
- Use synthetic queries for regression testing

---

## Complete Production Checklist

### Data Pipeline
- [ ] Document ingestion automated
- [ ] Chunking strategy validated
- [ ] Embeddings cached/versioned
- [ ] Vector DB indexed and optimized

### Query Pipeline
- [ ] Retrieval latency < 100ms
- [ ] Re-ranking improves relevance
- [ ] Prompts tested and versioned
- [ ] Input/output guardrails active

### Monitoring
- [ ] All queries logged
- [ ] Metrics dashboard live
- [ ] Error alerting configured
- [ ] User feedback collection

### Quality
- [ ] Evaluation suite automated
- [ ] Hallucination detection active
- [ ] Citation validation working
- [ ] Human eval process established

### Scale & Reliability
- [ ] Load testing completed
- [ ] Failover/redundancy configured
- [ ] Rate limiting implemented
- [ ] Cost monitoring active

---

## Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| Poor retrieval quality | Try hybrid search, better embeddings, tune chunk size |
| Slow queries | Use ANN algorithms, cache embeddings, batch processing |
| Hallucinations | Stricter prompts, NLI verification, citation requirements |
| High costs | Cache results, use smaller models for re-ranking, optimize chunk size |
| Context overflow | Dynamic truncation, summarize long contexts, hierarchical retrieval |
| Stale data | Incremental updates, versioned indexes, rebuild strategy |

---

## Further Reading

**Papers:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Dense Passage Retrieval for Open-Domain Question Answering" (Karpukhin et al., 2020)
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)

**Tools & Libraries:**
- LangChain: RAG framework
- LlamaIndex: Data framework for LLM apps
- Haystack: NLP framework with RAG
- RAGAS: RAG evaluation framework

**Databases:**
- Pinecone docs: pinecone.io/learn
- Weaviate docs: weaviate.io/developers
- Qdrant docs: qdrant.tech/documentation
