"""
RAGAS Evaluation Service for RAG Quality Assessment
Evaluates RAG systems without ground truth using various metrics
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    answer_correctness,
    answer_similarity,
    context_precision,
    context_entity_recall,
    answer_entity_recall,
    context_entity_precision,
    answer_entity_precision
)
from datasets import Dataset
from utils.logger import api_logger


@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    query: str
    answer: str
    context_chunks: List[str]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: str


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG systems without ground truth
    """
    
    def __init__(self):
        self.evaluation_history = []
        self.metrics_summary = defaultdict(list)
        
    def evaluate_single_response(
        self,
        query: str,
        answer: str,
        context_chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response using RAGAS metrics
        
        Args:
            query: User query
            answer: Generated answer
            context_chunks: Retrieved context chunks
            metadata: Additional metadata for evaluation
            
        Returns:
            EvaluationResult with metrics and metadata
        """
        start_time = time.time()
        
        try:
            api_logger.info(f"RAGAS: Starting evaluation for query: {query[:100]}...")
            
            # Prepare data for RAGAS
            context_text = "\n".join(context_chunks) if context_chunks else ""
            
            # Create dataset for RAGAS
            dataset_dict = {
                "question": [query],
                "answer": [answer],
                "contexts": [context_chunks],
                "ground_truth": [""]  # Empty for no ground truth
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Calculate metrics that don't require ground truth
            metrics = {}
            
            # 1. Faithfulness - measures if the answer is faithful to the context
            try:
                faithfulness_score = evaluate(dataset, faithfulness)
                metrics["faithfulness"] = faithfulness_score["faithfulness"]
                api_logger.debug(f"RAGAS: Faithfulness score: {faithfulness_score['faithfulness']}")
            except Exception as e:
                api_logger.warning(f"RAGAS: Failed to calculate faithfulness: {e}")
                metrics["faithfulness"] = None
            
            # 2. Answer Relevancy - measures if the answer is relevant to the question
            try:
                answer_relevancy_score = evaluate(dataset, answer_relevancy)
                metrics["answer_relevancy"] = answer_relevancy_score["answer_relevancy"]
                api_logger.debug(f"RAGAS: Answer relevancy score: {answer_relevancy_score['answer_relevancy']}")
            except Exception as e:
                api_logger.warning(f"RAGAS: Failed to calculate answer relevancy: {e}")
                metrics["answer_relevancy"] = None
            
            # 3. Context Relevancy - measures if the retrieved context is relevant
            try:
                context_relevancy_score = evaluate(dataset, context_relevancy)
                metrics["context_relevancy"] = context_relevancy_score["context_relevancy"]
                api_logger.debug(f"RAGAS: Context relevancy score: {context_relevancy_score['context_relevancy']}")
            except Exception as e:
                api_logger.warning(f"RAGAS: Failed to calculate context relevancy: {e}")
                metrics["context_relevancy"] = None
            
            # 4. Context Precision - measures precision of retrieved context
            try:
                context_precision_score = evaluate(dataset, context_precision)
                metrics["context_precision"] = context_precision_score["context_precision"]
                api_logger.debug(f"RAGAS: Context precision score: {context_precision_score['context_precision']}")
            except Exception as e:
                api_logger.warning(f"RAGAS: Failed to calculate context precision: {e}")
                metrics["context_precision"] = None
            
            # 5. Custom metrics
            custom_metrics = self._calculate_custom_metrics(query, answer, context_chunks)
            metrics.update(custom_metrics)
            
            # Create result
            result = EvaluationResult(
                query=query,
                answer=answer,
                context_chunks=context_chunks,
                metrics=metrics,
                metadata=metadata or {},
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Store in history
            self.evaluation_history.append(result)
            
            # Update metrics summary
            for metric_name, value in metrics.items():
                if value is not None:
                    self.metrics_summary[metric_name].append(value)
            
            evaluation_time = time.time() - start_time
            api_logger.info(f"RAGAS: Evaluation completed in {evaluation_time:.3f}s")
            
            return result
            
        except Exception as e:
            api_logger.error(f"RAGAS: Evaluation failed: {e}", exc_info=True)
            raise Exception(f"RAGAS evaluation failed: {str(e)}")
    
    def _calculate_custom_metrics(
        self,
        query: str,
        answer: str,
        context_chunks: List[str]
    ) -> Dict[str, float]:
        """
        Calculate custom metrics that don't require ground truth
        """
        metrics = {}
        
        try:
            # 1. Answer Length Ratio
            if len(query) > 0:
                metrics["answer_length_ratio"] = len(answer) / len(query)
            else:
                metrics["answer_length_ratio"] = 0.0
            
            # 2. Context Coverage - how much of the context is used in the answer
            if context_chunks:
                context_text = " ".join(context_chunks).lower()
                answer_words = set(answer.lower().split())
                context_words = set(context_text.split())
                
                if context_words:
                    coverage = len(answer_words.intersection(context_words)) / len(context_words)
                    metrics["context_coverage"] = coverage
                else:
                    metrics["context_coverage"] = 0.0
            else:
                metrics["context_coverage"] = 0.0
            
            # 3. Context Diversity - measure diversity of retrieved chunks
            if len(context_chunks) > 1:
                # Calculate similarity between chunks
                similarities = []
                for i in range(len(context_chunks)):
                    for j in range(i + 1, len(context_chunks)):
                        chunk1_words = set(context_chunks[i].lower().split())
                        chunk2_words = set(context_chunks[j].lower().split())
                        
                        if chunk1_words and chunk2_words:
                            similarity = len(chunk1_words.intersection(chunk2_words)) / len(chunk1_words.union(chunk2_words))
                            similarities.append(similarity)
                
                if similarities:
                    metrics["context_diversity"] = 1 - np.mean(similarities)
                else:
                    metrics["context_diversity"] = 1.0
            else:
                metrics["context_diversity"] = 1.0
            
            # 4. Query-Answer Similarity
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            
            if query_words and answer_words:
                similarity = len(query_words.intersection(answer_words)) / len(query_words.union(answer_words))
                metrics["query_answer_similarity"] = similarity
            else:
                metrics["query_answer_similarity"] = 0.0
            
            # 5. Answer Completeness (based on question words)
            question_words = {"what", "when", "where", "who", "why", "how", "which", "whose"}
            query_lower = query.lower()
            
            if any(word in query_lower for word in question_words):
                # Check if answer contains relevant information
                metrics["answer_completeness"] = 1.0 if len(answer.strip()) > 10 else 0.5
            else:
                metrics["answer_completeness"] = 1.0 if len(answer.strip()) > 5 else 0.5
            
        except Exception as e:
            api_logger.warning(f"RAGAS: Failed to calculate custom metrics: {e}")
            # Set default values
            metrics.update({
                "answer_length_ratio": 0.0,
                "context_coverage": 0.0,
                "context_diversity": 0.0,
                "query_answer_similarity": 0.0,
                "answer_completeness": 0.0
            })
        
        return metrics
    
    def evaluate_batch(
        self,
        evaluations: List[Tuple[str, str, List[str], Optional[Dict[str, Any]]]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses in batch
        """
        results = []
        
        for i, (query, answer, context_chunks, metadata) in enumerate(evaluations):
            try:
                result = self.evaluate_single_response(query, answer, context_chunks, metadata)
                results.append(result)
                api_logger.info(f"RAGAS: Completed evaluation {i+1}/{len(evaluations)}")
            except Exception as e:
                api_logger.error(f"RAGAS: Failed to evaluate item {i+1}: {e}")
                continue
        
        return results
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics
        """
        summary = {}
        
        for metric_name, values in self.metrics_summary.items():
            if values:
                summary[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }
        
        return summary
    
    def export_evaluation_results(self, filepath: str) -> None:
        """
        Export evaluation results to JSON file
        """
        try:
            results_data = []
            
            for result in self.evaluation_history:
                result_dict = {
                    "query": result.query,
                    "answer": result.answer,
                    "context_chunks": result.context_chunks,
                    "metrics": result.metrics,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp
                }
                results_data.append(result_dict)
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            api_logger.info(f"RAGAS: Exported {len(results_data)} evaluation results to {filepath}")
            
        except Exception as e:
            api_logger.error(f"RAGAS: Failed to export results: {e}")
            raise
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report
        """
        summary = self.get_metrics_summary()
        
        report = {
            "evaluation_summary": {
                "total_evaluations": len(self.evaluation_history),
                "evaluation_period": {
                    "start": self.evaluation_history[0].timestamp if self.evaluation_history else None,
                    "end": self.evaluation_history[-1].timestamp if self.evaluation_history else None
                }
            },
            "metrics_summary": summary,
            "recommendations": self._generate_recommendations(summary),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Dict[str, float]]) -> List[str]:
        """
        Generate recommendations based on metrics summary
        """
        recommendations = []
        
        # Faithfulness recommendations
        if "faithfulness" in summary:
            faithfulness_mean = summary["faithfulness"]["mean"]
            if faithfulness_mean < 0.7:
                recommendations.append("Low faithfulness scores indicate the model may be hallucinating. Consider improving context quality or adjusting the model prompt.")
        
        # Answer relevancy recommendations
        if "answer_relevancy" in summary:
            relevancy_mean = summary["answer_relevancy"]["mean"]
            if relevancy_mean < 0.6:
                recommendations.append("Low answer relevancy suggests responses may not address user queries effectively. Review prompt engineering and context retrieval.")
        
        # Context relevancy recommendations
        if "context_relevancy" in summary:
            context_relevancy_mean = summary["context_relevancy"]["mean"]
            if context_relevancy_mean < 0.5:
                recommendations.append("Low context relevancy indicates poor retrieval performance. Consider improving embedding model or retrieval strategy.")
        
        # Context coverage recommendations
        if "context_coverage" in summary:
            coverage_mean = summary["context_coverage"]["mean"]
            if coverage_mean < 0.3:
                recommendations.append("Low context coverage suggests the model may not be utilizing retrieved information effectively.")
        
        if not recommendations:
            recommendations.append("Overall RAG performance appears satisfactory based on current metrics.")
        
        return recommendations 