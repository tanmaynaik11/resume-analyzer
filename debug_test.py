# debug_test.py - Run this to test embeddings directly
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_embeddings():
    """Test embeddings with simple examples"""
    
    # Test texts - should have high similarity
    job_text = """Senior Data Scientist position requiring Python, Machine Learning, TensorFlow, PyTorch, 
    5+ years experience, PhD preferred, AWS cloud experience, statistical analysis"""
    
    good_resume = """Dr. Sarah Chen, PhD in Statistics, 8 years experience, Python expert, 
    TensorFlow, PyTorch, machine learning, AWS, Netflix, statistical modeling"""
    
    poor_resume = """Marketing manager with 3 years experience in social media campaigns, 
    Facebook ads, content creation, brand management, no technical background"""
    
    texts = [job_text, good_resume, poor_resume]
    
    print("Testing OpenAI embeddings...")
    print(f"Job description length: {len(job_text)}")
    print(f"Good resume length: {len(good_resume)}")
    print(f"Poor resume length: {len(poor_resume)}")
    
    try:
        # Get embeddings
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = np.array([item.embedding for item in response.data])
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Calculate similarities
        job_vector = embeddings[0:1]
        resume_vectors = embeddings[1:]
        
        similarities = cosine_similarity(job_vector, resume_vectors)[0]
        
        print(f"\nSimilarity Results:")
        print(f"Good resume similarity: {similarities[0]:.3f} ({similarities[0]*100:.1f}%)")
        print(f"Poor resume similarity: {similarities[1]:.3f} ({similarities[1]*100:.1f}%)")
        
        if similarities[0] > 0.5:
            print("✅ Embeddings working correctly!")
        else:
            print("❌ Embeddings may have issues")
            
    except Exception as e:
        print(f"Error: {e}")

def test_keyword_matching():
    """Test simple keyword matching"""
    
    job_keywords = set(['python', 'machine', 'learning', 'tensorflow', 'pytorch', 'phd', 'aws', 'statistics'])
    
    good_resume_keywords = set(['python', 'machine', 'learning', 'tensorflow', 'pytorch', 'phd', 'aws', 'statistics', 'netflix'])
    poor_resume_keywords = set(['marketing', 'social', 'media', 'facebook', 'content', 'brand'])
    
    # Jaccard similarity
    good_intersection = len(job_keywords.intersection(good_resume_keywords))
    good_union = len(job_keywords.union(good_resume_keywords))
    good_similarity = good_intersection / good_union
    
    poor_intersection = len(job_keywords.intersection(poor_resume_keywords))
    poor_union = len(job_keywords.union(poor_resume_keywords))
    poor_similarity = poor_intersection / poor_union
    
    print(f"\nKeyword Matching Test:")
    print(f"Good resume keyword similarity: {good_similarity:.3f} ({good_similarity*100:.1f}%)")
    print(f"Poor resume keyword similarity: {poor_similarity:.3f} ({poor_similarity*100:.1f}%)")

if __name__ == "__main__":
    print("=== DEBUGGING EMBEDDINGS ===")
    test_embeddings()
    test_keyword_matching()
    print("\nIf OpenAI embeddings show low similarity, there might be an API issue.")
    print("If keyword matching shows good separation, the algorithm logic is sound.")