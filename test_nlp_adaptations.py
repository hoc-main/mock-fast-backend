import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(r"c:\Users\sande\OneDrive\Desktop\study-material\pushkal\new-fast-backend")

# Mock the embedder before importing anything that might trigger it
mock_model = MagicMock()
mock_model.encode.return_value = np.zeros((3, 384)) # MiniLM-L6-v2 size is 384

with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
    from interview_fastapi.services.evaluation import evaluate_answer
    from interview_fastapi.db.models import InterviewAnswer

def test_evaluation_fields():
    print("Testing evaluation fields...")
    candidate = "I think that first, we should focus on the core concept because it is important. For example, in many cases..."
    question_data = {
        "question": "What is the core concept of this technology?",
        "answer": "The core concept involves distributed ledger technology and consensus mechanisms.",
        "expected_keywords": ["distributed", "ledger", "consensus"]
    }
    
    result = evaluate_answer(candidate, question_data)
    
    expected_fields = [
        "semantic_score", "question_relevance", "keyword_score", 
        "overlap_score", "length_score", "lexical_diversity", 
        "discourse_score", "penalty", "final_score"
    ]
    
    for field in expected_fields:
        if field in result:
            print(f"  [OK] Found field: {field} = {result[field]}")
        else:
            print(f"  [FAIL] Missing field: {field}")
            return False
    
    print("Evaluation fields test passed.")
    return True

def test_model_fields():
    print("Testing InterviewAnswer model fields...")
    try:
        answer = InterviewAnswer(
            transcript="test",
            semantic_score=0.8,
            question_relevance=0.7,
            lexical_diversity=0.6,
            discourse_score=0.5,
            penalty=0.1,
            final_score=0.75
        )
        print(f"  [OK] Successfully instantiated InterviewAnswer with new fields.")
        print(f"  Values: rel={answer.question_relevance}, div={answer.lexical_diversity}, disc={answer.discourse_score}, pen={answer.penalty}")
        return True
    except Exception as e:
        print(f"  [FAIL] Failed to instantiate InterviewAnswer: {e}")
        return False

if __name__ == "__main__":
    s1 = test_evaluation_fields()
    print("-" * 30)
    s2 = test_model_fields()
    
    if s1 and s2:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
