"""
Test script for RAGAS evaluation functionality
Demonstrates end-to-end evaluation without ground truth
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Mock the services for testing
class MockRetrieverService:
    def retrieve_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Mock retriever that returns sample context chunks"""
        sample_chunks = [
            "Customer support is available 24/7 through our help desk. Our team consists of trained professionals who can assist with technical issues, billing questions, and general inquiries.",
            "To reset your password, go to the login page and click on 'Forgot Password'. Enter your email address and follow the instructions sent to your inbox.",
            "Our refund policy allows customers to return products within 30 days of purchase. The item must be in original condition with all packaging intact.",
            "The API documentation provides detailed information about authentication, endpoints, and response formats. All API calls require a valid API key in the header.",
            "System maintenance is scheduled every Sunday from 2 AM to 6 AM EST. During this time, some services may be temporarily unavailable."
        ]
        return sample_chunks[:k]

class MockLLMClient:
    def chat(self, prompt: str) -> Dict[str, Any]:
        """Mock LLM that generates sample responses"""
        if "password" in prompt.lower():
            return {
                "response": "To reset your password, visit the login page and click 'Forgot Password'. Enter your email and check your inbox for reset instructions.",
                "model": "gpt-3.5-turbo",
                "input_tokens": 150,
                "output_tokens": 45
            }
        elif "refund" in prompt.lower():
            return {
                "response": "Our refund policy allows returns within 30 days. Items must be in original condition with packaging. Contact support for assistance.",
                "model": "gpt-3.5-turbo",
                "input_tokens": 120,
                "output_tokens": 35
            }
        else:
            return {
                "response": "I can help you with that. Please provide more details about your specific question or issue.",
                "model": "gpt-3.5-turbo",
                "input_tokens": 100,
                "output_tokens": 25
            }
    
    def get_response(self, json_response: Dict[str, Any]) -> Dict[str, Any]:
        return json_response

class MockMongoUtils:
    def get_chat_history(self, session_id: str) -> List[str]:
        """Mock chat history"""
        return [
            "User: How do I contact support?",
            "Assistant: You can reach our support team at support@company.com or call 1-800-HELP."
        ]

# Import the evaluator (we'll mock the dependencies)
try:
    from services.ragas_evaluator import RAGASEvaluator
    RAGAS_AVAILABLE = True
except ImportError:
    print("RAGAS not available, using mock evaluator")
    RAGAS_AVAILABLE = False

class MockRAGASEvaluator:
    """Mock evaluator for testing when RAGAS is not available"""
    
    def __init__(self):
        self.evaluation_history = []
        self.metrics_summary = {}
    
    def evaluate_single_response(self, query: str, answer: str, context_chunks: List[str], metadata: Optional[Dict[str, Any]] = None):
        """Mock evaluation that returns sample metrics"""
        from dataclasses import dataclass
        
        @dataclass
        class MockEvaluationResult:
            query: str
            answer: str
            context_chunks: List[str]
            metrics: Dict[str, float]
            metadata: Dict[str, Any]
            timestamp: str
        
        # Calculate mock metrics
        metrics = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
            "context_relevancy": 0.72,
            "context_precision": 0.68,
            "answer_length_ratio": len(answer) / max(len(query), 1),
            "context_coverage": 0.45,
            "context_diversity": 0.82,
            "query_answer_similarity": 0.31,
            "answer_completeness": 0.9
        }
        
        result = MockEvaluationResult(
            query=query,
            answer=answer,
            context_chunks=context_chunks,
            metrics=metrics,
            metadata=metadata or {},
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.evaluation_history.append(result)
        return result
    
    def evaluate_batch(self, evaluations):
        """Mock batch evaluation"""
        results = []
        for query, answer, context_chunks, metadata in evaluations:
            result = self.evaluate_single_response(query, answer, context_chunks, metadata)
            results.append(result)
        return results
    
    def get_metrics_summary(self):
        """Mock metrics summary"""
        return {
            "faithfulness": {"mean": 0.85, "std": 0.05, "min": 0.8, "max": 0.9, "count": len(self.evaluation_history)},
            "answer_relevancy": {"mean": 0.78, "std": 0.08, "min": 0.7, "max": 0.85, "count": len(self.evaluation_history)},
            "context_relevancy": {"mean": 0.72, "std": 0.06, "min": 0.65, "max": 0.8, "count": len(self.evaluation_history)}
        }
    
    def export_evaluation_results(self, filepath: str):
        """Mock export functionality"""
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
    
    def generate_evaluation_report(self):
        """Generate mock evaluation report"""
        return {
            "evaluation_summary": {
                "total_evaluations": len(self.evaluation_history),
                "evaluation_period": {
                    "start": self.evaluation_history[0].timestamp if self.evaluation_history else None,
                    "end": self.evaluation_history[-1].timestamp if self.evaluation_history else None
                }
            },
            "metrics_summary": {
                "faithfulness": {"mean": 0.85, "std": 0.05, "min": 0.8, "max": 0.9, "count": 1},
                "answer_relevancy": {"mean": 0.78, "std": 0.08, "min": 0.7, "max": 0.85, "count": 1},
                "context_relevancy": {"mean": 0.72, "std": 0.06, "min": 0.65, "max": 0.8, "count": 1}
            },
            "recommendations": [
                "Overall RAG performance appears satisfactory based on current metrics.",
                "Consider improving context retrieval for better relevancy scores."
            ],
            "generated_at": datetime.utcnow().isoformat()
        }

class RAGASEvaluationTester:
    """Test class for RAGAS evaluation functionality"""
    
    def __init__(self):
        self.retriever = MockRetrieverService()
        self.llm_client = MockLLMClient()
        self.mongo_utils = MockMongoUtils()
        
        if RAGAS_AVAILABLE:
            self.evaluator = RAGASEvaluator()
        else:
            self.evaluator = MockRAGASEvaluator()
    
    async def test_single_evaluation(self):
        """Test single query evaluation"""
        print("\n" + "="*60)
        print("TESTING SINGLE QUERY EVALUATION")
        print("="*60)
        
        # Sample queries for testing
        test_queries = [
            "How do I reset my password?",
            "What is your refund policy?",
            "How can I contact customer support?",
            "What are the API authentication requirements?",
            "When is system maintenance scheduled?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i}: {query} ---")
            
            try:
                # Step 1: Retrieve context
                context_chunks = self.retriever.retrieve_similar_chunks(query, k=3)
                print(f"Retrieved {len(context_chunks)} context chunks")
                
                # Step 2: Generate answer
                system_prompt = f"Context: {' '.join(context_chunks)}\n\nQuestion: {query}\n\nAnswer:"
                llm_response = self.llm_client.chat(system_prompt)
                answer = llm_response["response"]
                print(f"Generated answer: {answer[:100]}...")
                
                # Step 3: Prepare metadata
                metadata = {
                    "test_id": i,
                    "query_type": "customer_support",
                    "chunks_retrieved": len(context_chunks),
                    "model": llm_response.get("model", "unknown"),
                    "input_tokens": llm_response.get("input_tokens", 0),
                    "output_tokens": llm_response.get("output_tokens", 0)
                }
                
                # Step 4: Evaluate
                evaluation_result = self.evaluator.evaluate_single_response(
                    query=query,
                    answer=answer,
                    context_chunks=context_chunks,
                    metadata=metadata
                )
                
                # Step 5: Display results
                print(f"Evaluation Metrics:")
                for metric_name, value in evaluation_result.metrics.items():
                    if value is not None:
                        print(f"  {metric_name}: {value:.3f}")
                
                print(f"Evaluation completed successfully!")
                
            except Exception as e:
                print(f"Error in evaluation: {e}")
    
    async def test_batch_evaluation(self):
        """Test batch evaluation"""
        print("\n" + "="*60)
        print("TESTING BATCH EVALUATION")
        print("="*60)
        
        # Sample batch queries
        batch_queries = [
            "How do I reset my password?",
            "What is your refund policy?",
            "How can I contact customer support?",
            "What are the API authentication requirements?",
            "When is system maintenance scheduled?",
            "How do I update my billing information?",
            "What are the system requirements?",
            "How do I export my data?"
        ]
        
        try:
            # Prepare batch data
            batch_data = []
            for i, query in enumerate(batch_queries):
                context_chunks = self.retriever.retrieve_similar_chunks(query, k=3)
                system_prompt = f"Context: {' '.join(context_chunks)}\n\nQuestion: {query}\n\nAnswer:"
                llm_response = self.llm_client.chat(system_prompt)
                answer = llm_response["response"]
                
                batch_data.append((query, answer, context_chunks, {"batch_id": i}))
            
            # Run batch evaluation
            print(f"Running batch evaluation for {len(batch_data)} queries...")
            results = self.evaluator.evaluate_batch(batch_data)
            
            print(f"Batch evaluation completed! Processed {len(results)} queries.")
            
            # Display summary
            if hasattr(self.evaluator, 'get_metrics_summary'):
                summary = self.evaluator.get_metrics_summary()
                print(f"\nMetrics Summary:")
                for metric_name, stats in summary.items():
                    print(f"  {metric_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            
        except Exception as e:
            print(f"Error in batch evaluation: {e}")
    
    async def test_evaluation_report(self):
        """Test evaluation report generation"""
        print("\n" + "="*60)
        print("TESTING EVALUATION REPORT GENERATION")
        print("="*60)
        
        try:
            if hasattr(self.evaluator, 'generate_evaluation_report'):
                report = self.evaluator.generate_evaluation_report()
                
                print("Evaluation Report:")
                print(json.dumps(report, indent=2))
                
                # Test export functionality
                if hasattr(self.evaluator, 'export_evaluation_results'):
                    export_file = f"evaluation_results_{int(time.time())}.json"
                    self.evaluator.export_evaluation_results(export_file)
                    print(f"\nResults exported to: {export_file}")
            
        except Exception as e:
            print(f"Error generating report: {e}")
    
    async def run_comprehensive_test(self):
        """Run all tests"""
        print("RAGAS EVALUATION COMPREHENSIVE TEST")
        print("="*60)
        print(f"RAGAS Available: {RAGAS_AVAILABLE}")
        print(f"Test started at: {datetime.utcnow().isoformat()}")
        
        # Run tests
        await self.test_single_evaluation()
        await self.test_batch_evaluation()
        await self.test_evaluation_report()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)

def main():
    """Main function to run the tests"""
    tester = RAGASEvaluationTester()
    
    # Run the comprehensive test
    asyncio.run(tester.run_comprehensive_test())

if __name__ == "__main__":
    main() 