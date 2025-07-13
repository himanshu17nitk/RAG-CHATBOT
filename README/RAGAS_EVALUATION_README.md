# RAGAS Evaluation System for Customer Support Chatbot

## Overview

This implementation provides a comprehensive RAGAS (Retrieval-Augmented Generation Assessment) evaluation system for your customer support chatbot. The system evaluates RAG quality without requiring ground truth data, focusing on metrics that assess query relevance, answer quality, and context utilization.

## Features

### 🔍 **No Ground Truth Required**
- Evaluates RAG systems without needing reference answers
- Uses intrinsic metrics based on query, answer, and context relationships

### 📊 **Comprehensive Metrics**
- **Faithfulness**: Measures if answers are faithful to retrieved context
- **Answer Relevancy**: Assesses if answers are relevant to queries
- **Context Relevancy**: Evaluates relevance of retrieved context
- **Context Precision**: Measures precision of context retrieval
- **Custom Metrics**: Additional metrics for comprehensive evaluation

### 🚀 **Easy Integration**
- RESTful API endpoints for evaluation
- Batch processing capabilities
- Real-time evaluation during RAG pipeline
- Export functionality for analysis

## Installation

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Verify Installation**
```bash
python test_ragas_evaluation.py
```

## API Endpoints

### 1. Single Query Evaluation
```http
POST /evaluation/evaluate
Content-Type: application/json
X-Client-Secret: your-secret-key

{
  "query": "How do I reset my password?",
  "session_id": "session-123",
  "user_id": "user-456",
  "include_metadata": true
}
```

**Response:**
```json
{
  "query": "How do I reset my password?",
  "answer": "To reset your password...",
  "context_chunks": ["chunk1", "chunk2"],
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_relevancy": 0.72,
    "context_precision": 0.68,
    "answer_length_ratio": 2.5,
    "context_coverage": 0.45,
    "context_diversity": 0.82,
    "query_answer_similarity": 0.31,
    "answer_completeness": 0.9
  },
  "metadata": {
    "session_id": "session-123",
    "user_id": "user-456",
    "chunks_retrieved": 2,
    "model": "gpt-3.5-turbo"
  },
  "evaluation_time": 1.234
}
```

### 2. Batch Evaluation
```http
POST /evaluation/evaluate-batch
Content-Type: application/json
X-Client-Secret: your-secret-key

{
  "queries": [
    "How do I reset my password?",
    "What is your refund policy?",
    "How can I contact support?"
  ],
  "session_id": "session-123",
  "user_id": "user-456",
  "include_metadata": true
}
```

### 3. Evaluation Report
```http
GET /evaluation/report
X-Client-Secret: your-secret-key
```

**Response:**
```json
{
  "evaluation_summary": {
    "total_evaluations": 25,
    "evaluation_period": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-15T23:59:59Z"
    }
  },
  "metrics_summary": {
    "faithfulness": {
      "mean": 0.82,
      "std": 0.08,
      "min": 0.65,
      "max": 0.95,
      "count": 25
    },
    "answer_relevancy": {
      "mean": 0.76,
      "std": 0.12,
      "min": 0.45,
      "max": 0.92,
      "count": 25
    }
  },
  "recommendations": [
    "Overall RAG performance appears satisfactory based on current metrics.",
    "Consider improving context retrieval for better relevancy scores."
  ],
  "generated_at": "2024-01-15T23:59:59Z"
}
```

### 4. Export Results
```http
POST /evaluation/export
Content-Type: application/json
X-Client-Secret: your-secret-key

{
  "filepath": "evaluation_results.json"
}
```

## Metrics Explained

### Core RAGAS Metrics

1. **Faithfulness (0-1)**
   - Measures if the generated answer is faithful to the retrieved context
   - Higher scores indicate less hallucination
   - **Good**: >0.8, **Acceptable**: 0.6-0.8, **Poor**: <0.6

2. **Answer Relevancy (0-1)**
   - Assesses if the answer is relevant to the user's query
   - Higher scores indicate better query-answer alignment
   - **Good**: >0.7, **Acceptable**: 0.5-0.7, **Poor**: <0.5

3. **Context Relevancy (0-1)**
   - Evaluates relevance of retrieved context to the query
   - Higher scores indicate better retrieval performance
   - **Good**: >0.6, **Acceptable**: 0.4-0.6, **Poor**: <0.4

4. **Context Precision (0-1)**
   - Measures precision of context retrieval
   - Higher scores indicate more focused context
   - **Good**: >0.7, **Acceptable**: 0.5-0.7, **Poor**: <0.5

### Custom Metrics

5. **Answer Length Ratio**
   - Ratio of answer length to query length
   - Helps identify overly verbose or too brief responses
   - **Optimal**: 1.5-3.0

6. **Context Coverage (0-1)**
   - How much of the retrieved context is used in the answer
   - Higher scores indicate better context utilization
   - **Good**: >0.4, **Acceptable**: 0.2-0.4, **Poor**: <0.2

7. **Context Diversity (0-1)**
   - Measures diversity of retrieved chunks
   - Higher scores indicate more diverse context
   - **Good**: >0.7, **Acceptable**: 0.5-0.7, **Poor**: <0.5

8. **Query-Answer Similarity (0-1)**
   - Semantic similarity between query and answer
   - Helps identify off-topic responses
   - **Good**: 0.2-0.6, **Too Low**: <0.2, **Too High**: >0.6

9. **Answer Completeness (0-1)**
   - Assesses if the answer addresses the query type
   - Higher scores for comprehensive responses
   - **Good**: >0.8, **Acceptable**: 0.6-0.8, **Poor**: <0.6

## Usage Examples

### Python Integration

```python
from services.ragas_evaluator import RAGASEvaluator

# Initialize evaluator
evaluator = RAGASEvaluator()

# Evaluate single response
result = evaluator.evaluate_single_response(
    query="How do I reset my password?",
    answer="To reset your password, go to the login page...",
    context_chunks=["chunk1", "chunk2"],
    metadata={"user_id": "123", "session_id": "456"}
)

print(f"Faithfulness: {result.metrics['faithfulness']:.3f}")
print(f"Answer Relevancy: {result.metrics['answer_relevancy']:.3f}")

# Generate report
report = evaluator.generate_evaluation_report()
print(f"Total evaluations: {report['evaluation_summary']['total_evaluations']}")
```

### Batch Evaluation

```python
# Prepare batch data
evaluations = [
    ("Query 1", "Answer 1", ["context1"], {"id": 1}),
    ("Query 2", "Answer 2", ["context2"], {"id": 2}),
    # ... more evaluations
]

# Run batch evaluation
results = evaluator.evaluate_batch(evaluations)

# Get summary statistics
summary = evaluator.get_metrics_summary()
for metric, stats in summary.items():
    print(f"{metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
```

## Testing

Run the comprehensive test suite:

```bash
python test_ragas_evaluation.py
```

This will test:
- Single query evaluation
- Batch evaluation
- Report generation
- Export functionality

## Configuration

### Environment Variables

```bash
# Required for API authentication
CLIENT_SECRET_KEY=your-secret-key

# Optional: RAGAS configuration
RAGAS_CACHE_DIR=./ragas_cache
RAGAS_VERBOSE=true
```

### Customization

You can customize the evaluation by modifying:

1. **Metrics Thresholds** in `services/ragas_evaluator.py`
2. **Custom Metrics** in `_calculate_custom_metrics()`
3. **Recommendations** in `_generate_recommendations()`

## Best Practices

### 1. **Regular Evaluation**
- Run evaluations weekly or after major updates
- Track metrics over time to identify trends
- Set up automated evaluation pipelines

### 2. **Interpretation Guidelines**
- Focus on trends rather than absolute scores
- Consider context when interpreting metrics
- Use multiple metrics for comprehensive assessment

### 3. **Actionable Insights**
- Low faithfulness → Improve context quality or prompt engineering
- Low answer relevancy → Review retrieval strategy
- Low context coverage → Optimize context utilization
- High context diversity → Consider chunking strategy

### 4. **Performance Optimization**
- Use batch evaluation for large datasets
- Cache evaluation results when possible
- Monitor evaluation time and resource usage

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install ragas datasets evaluate scikit-learn
   ```

2. **Memory Issues**
   - Reduce batch size for large evaluations
   - Use streaming for very large datasets

3. **Slow Evaluation**
   - Enable caching in RAGAS
   - Use GPU acceleration if available
   - Optimize context chunk size

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Alternative Approaches

### 1. **Human Evaluation**
- Manual assessment by domain experts
- More accurate but expensive and time-consuming
- Good for validation of automated metrics

### 2. **Semantic Similarity**
- Use embedding-based similarity metrics
- Compare query-answer and context-answer similarities
- Implement with sentence-transformers

### 3. **Rule-Based Metrics**
- Custom rules for specific domains
- Keyword matching and coverage analysis
- Syntactic and semantic checks

### 4. **Hybrid Approach**
- Combine multiple evaluation methods
- Weight different metrics based on importance
- Use ensemble methods for final scores

## Future Enhancements

1. **Advanced Metrics**
   - Factual consistency checking
   - Temporal relevance assessment
   - Multi-language support

2. **Visualization**
   - Interactive dashboards
   - Trend analysis charts
   - Comparative analysis tools

3. **Automation**
   - Continuous evaluation pipelines
   - Alert systems for metric degradation
   - Automated optimization suggestions

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test examples
3. Examine the source code comments
4. Create an issue with detailed error information

---

**Prompt Quality Rating: 8/10** - The prompt was clear and specific about implementing RAGAS evaluation without ground truth, but could have included more specific requirements about which metrics to prioritize or performance constraints. 