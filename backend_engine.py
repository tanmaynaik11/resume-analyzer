
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
import os
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

class CandidateRecommendationEngine:
    def __init__(self):
        self.job_vector = None
        self.resume_vectors = None
        self.candidates = []
        self.job_text = ""
        self.debug_mode = True  # Set to True for debugging
    
    def debug_print(self, message):
        """Print debug information"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded files"""
        try:
            self.debug_print(f"Extracting from file: {uploaded_file.name}")
            
            if uploaded_file.type == "application/pdf":
                text = self._extract_from_pdf_improved(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self._extract_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = str(uploaded_file.read(), "utf-8")
            else:
                raise ValueError(f"Unsupported file type: {uploaded_file.type}")
            
            # Clean and validate extracted text
            text = self._clean_extracted_text(text)
            
            self.debug_print(f"Extracted {len(text)} characters from {uploaded_file.name}")
            
            return text
            
        except Exception as e:
            self.debug_print(f"Error extracting from {uploaded_file.name}: {str(e)}")
            raise Exception(f"Error extracting text from {uploaded_file.name}: {str(e)}")
    
    def _extract_from_pdf_improved(self, uploaded_file):
        """PDF text extraction"""
        text = ""
        
        try:
            # Method 1: PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
            
            # If we got good text, return it
            if len(text.strip()) > 100:
                return text
                
        except Exception as e:
            self.debug_print(f"PyPDF2 failed: {e}")
        
        try:
            # Method 2: Read as binary and decode
            uploaded_file.seek(0)
            binary_data = uploaded_file.read()
            text = binary_data.decode('utf-8', errors='ignore')
            
            # Clean up binary artifacts
            text = re.sub(r'[^\x20-\x7E\n\r\t]', ' ', text)
            
            if len(text.strip()) > 50:
                return text
                
        except Exception as e:
            self.debug_print(f"Binary decode failed: {e}")
        
        # If all methods fail
        return f"Could not extract readable text from {uploaded_file.name}. Please try converting to .txt format."
    
    def _extract_from_docx(self, uploaded_file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            return text
        except Exception as e:
            self.debug_print(f"DOCX extraction failed: {e}")
            return f"Could not extract text from DOCX: {uploaded_file.name}"
    
    def _clean_extracted_text(self, text):
        """Clean and normalize extracted text"""
        if not text or len(text.strip()) < 10:
            return "No readable text found in document"
        
        # Handle the word-per-line issue by joining short lines
        lines = text.split('\n')
        cleaned_lines = []
        current_sentence = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # If line is very short (1-3 words), it's likely part of a broken sentence
            words_in_line = len(line.split())
            
            if words_in_line <= 3 and not line.endswith(('.', ':', ')', '●')):
                current_sentence.append(line)
            else:
                # End of sentence/section
                if current_sentence:
                    current_sentence.append(line)
                    cleaned_lines.append(' '.join(current_sentence))
                    current_sentence = []
                else:
                    cleaned_lines.append(line)
        
        # Add any remaining sentence
        if current_sentence:
            cleaned_lines.append(' '.join(current_sentence))
        
        # Join all lines
        text = ' '.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up bullet points and special characters
        text = re.sub(r'●\s*', '', text)  # Remove bullet points
        text = re.sub(r'\|\s*', ' ', text)  # Remove pipe separators
        text = re.sub(r'[^\w\s\.\-\+\#\(\)\,\;\:\@\/]', ' ', text)  # Keep useful chars
        
        # Fix common technical terms that get split
        text = re.sub(r'\bc\s*\+\s*\+', 'cplusplus', text, flags=re.IGNORECASE)
        text = re.sub(r'\bc\s*\#', 'csharp', text, flags=re.IGNORECASE)
        text = re.sub(r'\.net', 'dotnet', text, flags=re.IGNORECASE)
        text = re.sub(r'A\s*/\s*B', 'AB', text, flags=re.IGNORECASE)  # Fix "A / B testing"
        
        # Fix email addresses that got split
        text = re.sub(r'(\w+)\s*@\s*(\w+)\s*\.\s*(\w+)', r'\1@\2.\3', text)
        
        # Remove excessive whitespace again
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def enhance_text_for_matching(self, text, is_job_description=False):
        """Enhance text to improve matching by expanding technical terms"""
        
        # Expand common abbreviations and add synonyms
        enhancements = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'js': 'javascript',
            'ts': 'typescript',
            'react': 'reactjs frontend',
            'vue': 'vuejs frontend',
            'angular': 'angularjs frontend',
            'node': 'nodejs backend',
            'aws': 'amazon web services cloud',
            'gcp': 'google cloud platform',
            'azure': 'microsoft azure cloud',
            'sql': 'database querying',
            'nosql': 'database mongodb',
            'rest': 'api restful',
            'graphql': 'api query language',
            'docker': 'containerization',
            'kubernetes': 'container orchestration',
            'tensorflow': 'machine learning framework',
            'pytorch': 'deep learning framework',
            'scikit': 'machine learning library',
            'pandas': 'data manipulation',
            'numpy': 'numerical computing',
            'matplotlib': 'data visualization',
            'seaborn': 'statistical visualization',
            'plotly': 'interactive visualization',
            'jupyter': 'notebook development',
            'git': 'version control',
            'github': 'code repository',
            'agile': 'project management methodology',
            'scrum': 'agile development framework',
            'phd': 'doctoral degree doctorate',
            'masters': 'graduate degree',
            'bachelors': 'undergraduate degree',
            'senior': 'experienced lead',
            'principal': 'senior lead architect',
            'lead': 'senior management',
            'years': 'experience background'
        }
        
        enhanced_text = text.lower()
        
        # Add enhancements
        for abbrev, expansion in enhancements.items():
            if re.search(r'\b' + re.escape(abbrev) + r'\b', enhanced_text):
                enhanced_text += f" {expansion}"
        
        # Add experience level indicators
        experience_patterns = [
            (r'(\d+)\s*\+?\s*years?', r'\1 years experience'),
            (r'senior', 'experienced professional'),
            (r'junior', 'entry level'),
            (r'lead', 'leadership management'),
            (r'principal', 'senior architect expert'),
        ]
        
        for pattern, replacement in experience_patterns:
            enhanced_text = re.sub(pattern, f'{replacement}', enhanced_text)
        
        return enhanced_text
    
    def get_openai_embeddings(self, texts):
        """Get embeddings from OpenAI API with text enhancement"""
        self.debug_print(f"Getting OpenAI embeddings for {len(texts)} texts")
        
        try:
            # Enhance texts for better matching
            enhanced_texts = []
            for i, text in enumerate(texts):
                # Clean text
                clean_text = self._clean_extracted_text(text)
                
                # Enhance with synonyms and expansions
                enhanced_text = self.enhance_text_for_matching(clean_text, is_job_description=(i==0))
                
                # Truncate if too long
                if len(enhanced_text) > 8000:
                    enhanced_text = enhanced_text[:8000]
                
                enhanced_texts.append(enhanced_text)
                self.debug_print(f"Text {i}: enhanced from {len(clean_text)} to {len(enhanced_text)} chars")
            
            # Get embeddings using larger model for better quality
            response = openai.embeddings.create(
                model="text-embedding-3-large",  # Better quality than small
                input=enhanced_texts
            )
            
            embeddings = [item.embedding for item in response.data]
            embedding_array = np.array(embeddings)
            
            self.debug_print(f"OpenAI embeddings shape: {embedding_array.shape}")
            
            return embedding_array
            
        except Exception as e:
            self.debug_print(f"OpenAI embeddings failed: {e}")
            return None
    
    def process_job_description(self, job_text):
        """Process job description and create its embedding"""
        self.job_text = job_text
        self.debug_print(f"Processing job description: {len(job_text)} characters")
        
        # Create job embedding immediately
        self.job_vector = self.get_openai_embeddings([job_text])
        self.debug_print("Job description processed and embedded")
        
        return True
    
    def process_resumes(self, resumes_data):
        """Process resumes and create vector representations"""
        self.candidates = resumes_data
        self.debug_print(f"Processing {len(resumes_data)} resumes")
        
        # Extract resume texts
        resume_texts = [resume['content'] for resume in resumes_data]
        
        # Create resume embeddings (job embedding already created)
        self.resume_vectors = self.get_openai_embeddings(resume_texts)
        
        self.debug_print(f"Vectors created - Job: {self.job_vector.shape}, Resumes: {self.resume_vectors.shape}")
        return True
    
    
    
    def calculate_similarities(self):
        """Calculate cosine similarity between job and resumes"""
        if self.job_vector is None or self.resume_vectors is None:
            raise ValueError("Job description and resumes must be processed first")
        
        self.debug_print("Calculating cosine similarities...")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(self.job_vector, self.resume_vectors)[0]
        
        self.debug_print(f"Similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
        
        return similarities
    
    
    # def calculate_similarities(self):
    #     """Alternative: Simple dot product (fastest, assumes normalized embeddings)"""
    #     if self.job_vector is None or self.resume_vectors is None:
    #         raise ValueError("Job description and resumes must be processed first")
        
    #     self.debug_print("Calculating dot product similarities...")
        
    #     job_vec = self.job_vector.flatten()
    #     similarities = np.dot(self.resume_vectors, job_vec)
        
    #     # Normalize to [0, 1] range
    #     # similarities = (similarities + 1) / 2  # Convert from [-1, 1] to [0, 1]
    #     similarities = np.clip(similarities, 0, 1)
        
    #     self.debug_print(f"Similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
        
    #     return similarities
    
    def generate_ai_summary(self, job_description, resume_content, candidate_name, similarity_score):
        """Generate AI summary using OpenAI API"""
        try:
            prompt = f"""
            JOB REQUIREMENTS:
            {job_description[:1000]}
            
            CANDIDATE: {candidate_name}
            RESUME:
            {resume_content[:1000]}
            
            MATCH SCORE: {similarity_score:.1%}
            
            As a technical recruiter, write a 2-sentence assessment describing why is this person a great fit for this role, also mention what are some of the gaps between RESUME and JOB REQUIREMENTS:. Focus on specific technical skills, experience level, and key strengths or gaps.
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a technical recruiter. Be specific and concise."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=120,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.debug_print(f"AI summary failed for {candidate_name}: {e}")
            return f"{candidate_name} shows {similarity_score:.1%} compatibility based on technical skills and experience alignment with the role requirements."
    
    def get_top_candidates(self, job_description, top_n=5):
        """Get top N candidates with AI-generated summaries"""
        similarities = self.calculate_similarities()
        
        # Create results list
        results = []
        for i, (candidate, similarity) in enumerate(zip(self.candidates, similarities)):
            results.append({
                'rank': i + 1,
                'name': candidate['name'],
                'similarity_score': similarity,
                'content': candidate['content'],
                'ai_summary': None
            })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Take top N candidates
        top_results = results[:top_n]
        
        # Generate AI summaries for top candidates
        for i, result in enumerate(top_results):
            result['ai_summary'] = self.generate_ai_summary(
                job_description, 
                result['content'], 
                result['name'], 
                result['similarity_score']
            )
            
            # Small delay to avoid rate limiting
            if i < len(top_results) - 1:
                time.sleep(0.2)
        
        return top_results
