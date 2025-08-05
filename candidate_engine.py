# candidate_engine.py (Hybrid Weighted Approach)
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
from sentence_transformers import SentenceTransformer
import numpy as np

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
        self.debug_mode = True
        self.st_model = SentenceTransformer("all-mpnet-base-v2")
        
        # Hybrid approach components
        self.job_sections = {}
        self.candidate_sections = []
        self.section_weights = {
            "education": 0.35,      # 20% - Education background
            "experience": 0.50,     # 35% - Work experience (most important)
            "technical_skills": 0.15,  # 30% - Technical skills  
            # "domain_knowledge": 0.15    # 15% - Domain expertise
        }
    
    def debug_print(self, message):
        """Print debug information"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded files with improved methods"""
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
        """Improved PDF text extraction"""
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
        """Clean and normalize extracted text - handles word-per-line PDFs"""
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
    
    def parse_job_sections_with_ai(self, job_text):
        """Parse job description into weighted sections using OpenAI"""
        self.debug_print("Parsing job description into sections using OpenAI...")
        
        try:
            prompt = f"""
            Analyze the following job description and extract the key requirements into these 4 categories. 
            Return ONLY a JSON object with these exact keys: education, experience, technical_skills, domain_knowledge.
            
            IMPORTANT: While extracting technical_skills focus on explicit skill and impact rather than buzzwords. For each category, extract the relevant requirements as a SINGLE PARAGRAPH OF PLAIN TEXT. 
            Do NOT use lists, arrays, or nested JSON structures. Write everything as descriptive sentences.
            If no information is found for a category, write "Not specified".
            
            Job Description:
            {job_text}
            
            Example format:
            {{
              "education": "Master's or PhD in Computer Science required. Advanced degree preferred.",
              "experience": "5+ years of experience in software development. Experience with large-scale systems preferred.",
              "technical_skills": "Proficiency in Python, Java, and SQL required. Experience with TensorFlow, PyTorch, and cloud platforms like AWS and GCP. Knowledge of Docker, Kubernetes, and CI/CD pipelines.",
              "domain_knowledge": "Strong understanding of machine learning algorithms, statistical analysis, and data modeling. Experience with A/B testing and experimental design."
            }}
            
            JSON Response:
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR analyst. Extract job requirements into structured categories as PLAIN TEXT paragraphs, not lists or arrays. Return only valid JSON with text values."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1  # Low temperature for consistent parsing
            )
            
            # Parse the JSON response
            import json
            sections_text = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if sections_text.startswith('```json'):
                sections_text = sections_text.replace('```json', '').replace('```', '').strip()
            elif sections_text.startswith('```'):
                sections_text = sections_text.replace('```', '').strip()
            
            sections = json.loads(sections_text)
            
            # Ensure all sections are strings, not lists or dicts
            cleaned_sections = {}
            for key, value in sections.items():
                if isinstance(value, (list, dict)):
                    # Convert lists/dicts to readable text
                    if isinstance(value, dict):
                        # Convert dict to descriptive text
                        text_parts = []
                        for category, items in value.items():
                            if isinstance(items, list):
                                items_text = ", ".join(str(item) for item in items)
                                text_parts.append(f"{category}: {items_text}")
                            else:
                                text_parts.append(f"{category}: {items}")
                        cleaned_sections[key] = ". ".join(text_parts) + "."
                    elif isinstance(value, list):
                        cleaned_sections[key] = ", ".join(str(item) for item in value) + "."
                else:
                    cleaned_sections[key] = str(value)
            
            # Store job sections
            self.job_sections = {
                "education": cleaned_sections.get("education", "Bachelor's degree required"),
                "experience": cleaned_sections.get("experience", "Relevant work experience required"),
                "technical_skills": cleaned_sections.get("technical_skills", "Technical skills required"),
                "domain_knowledge": cleaned_sections.get("domain_knowledge", "Domain expertise required")
            }
            
            self.debug_print(f"AI-parsed job sections (cleaned):")
            for section, text in self.job_sections.items():
                if section=="technical_skills":
                    self.debug_print(f"  {section}: {text}...")
            
            return True
            
        except Exception as e:
            self.debug_print(f"AI job parsing failed: {e}")
            # Fallback to basic parsing
            return self._fallback_job_parsing(job_text)
    
    def parse_resume_sections_with_ai(self, resume_text, candidate_name):
        """Parse resume into sections using OpenAI"""
        self.debug_print(f"AI-parsing resume for {candidate_name}...")
        
        try:
            prompt = f"""
            Analyze the following resume and extract information into these 4 categories.
            Return ONLY a JSON object with these exact keys: education, experience, technical_skills, domain_knowledge.

            While extracting technical_skills focus on explicit skill and impact rather than buzzwords.
            For each category, extract and summarize the relevant information as a single paragraph. If no information is found for a category, write "Not specified".
            
            Resume:
            {resume_text[:2000]}  
            
            JSON Response:
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert resume analyzer. Extract resume information into structured categories. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            # Parse the JSON response
            import json
            sections_text = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if sections_text.startswith('```json'):
                sections_text = sections_text.replace('```json', '').replace('```', '').strip()
            elif sections_text.startswith('```'):
                sections_text = sections_text.replace('```', '').strip()
            
            sections = json.loads(sections_text)
            
            parsed_sections = {
                "education": sections.get("education", "Education not specified"),
                "experience": sections.get("experience", "Experience not specified"),
                "technical_skills": sections.get("technical_skills", "Technical skills not specified"),
                "domain_knowledge": sections.get("domain_knowledge", "Domain knowledge not specified")
            }
            
            self.debug_print(f"AI-parsed {candidate_name} sections:")
            for section, text in parsed_sections.items():
                if section=="technical_skills":
                    self.debug_print(f"  {section}: {text}...")
            
            return parsed_sections
            
        except Exception as e:
            self.debug_print(f"AI resume parsing failed for {candidate_name}: {e}")
            # Fallback to regex parsing
            return self._fallback_resume_parsing(resume_text)
    
    def _fallback_job_parsing(self, job_text):
        """Fallback job parsing if AI fails"""
        self.debug_print("Using fallback job parsing...")
        
        # Simple keyword-based extraction
        job_lower = job_text.lower()
        
        # Education
        education_keywords = ["phd", "master", "bachelor", "degree", "education", "university"]
        education_sentences = self._extract_sentences_with_keywords(job_text, education_keywords)
        
        # Experience  
        experience_keywords = ["years", "experience", "senior", "junior", "lead", "background"]
        experience_sentences = self._extract_sentences_with_keywords(job_text, experience_keywords)
        
        # Technical skills
        tech_keywords = ["python", "sql", "programming", "technical", "skills", "aws", "tensorflow"]
        tech_sentences = self._extract_sentences_with_keywords(job_text, tech_keywords)
        
        # Domain knowledge
        domain_keywords = ["machine learning", "data science", "statistics", "analytics", "research"]
        domain_sentences = self._extract_sentences_with_keywords(job_text, domain_keywords)
        
        self.job_sections = {
            "education": education_sentences if education_sentences else "Bachelor's degree required",
            "experience": experience_sentences if experience_sentences else "Relevant work experience required",
            "technical_skills": tech_sentences if tech_sentences else "Technical skills required",
            "domain_knowledge": domain_sentences if domain_sentences else "Domain expertise required"
        }
        
        return True
    
    def _fallback_resume_parsing(self, resume_text):
        """Fallback resume parsing if AI fails"""
        self.debug_print("Using fallback resume parsing...")
        
        # Simple keyword-based extraction
        education_keywords = ["phd", "master", "bachelor", "degree", "university", "college"]
        education_text = self._extract_sentences_with_keywords(resume_text, education_keywords)
        
        experience_keywords = ["years", "experience", "worked", "company", "role", "position"]
        experience_text = self._extract_sentences_with_keywords(resume_text, experience_keywords)
        
        tech_keywords = ["python", "sql", "programming", "technical", "aws", "tensorflow", "skills"]
        tech_text = self._extract_sentences_with_keywords(resume_text, tech_keywords)
        
        domain_keywords = ["machine learning", "data science", "statistics", "research", "analytics"]
        domain_text = self._extract_sentences_with_keywords(resume_text, domain_keywords)
        
        return {
            "education": education_text if education_text else "Education information not found",
            "experience": experience_text if experience_text else "Experience information not found", 
            "technical_skills": tech_text if tech_text else "Technical skills not found",
            "domain_knowledge": domain_text if domain_text else "Domain knowledge not found"
        }
    
    def _extract_sentences_with_keywords(self, text, keywords):
        """Extract sentences containing specific keywords"""
        sentences = re.split(r'[.!?]+', text)
        matching_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        matching_sentences.append(sentence)
                        break
        
        return ' '.join(matching_sentences[:3])  # Limit to first 3 matching sentences
    
    def process_job_description(self, job_text):
        """Process job description using AI-powered hybrid approach"""
        self.job_text = job_text
        self.debug_print(f"Processing job description: {len(job_text)} characters")
        
        # Parse job into sections using AI
        success = self.parse_job_sections_with_ai(job_text)
        
        if success:
            self.debug_print("Job description processed with AI-powered hybrid approach")
        
        return success
    
    def process_resumes(self, resumes_data):
        """Process resumes using AI-powered hybrid approach"""
        self.candidates = resumes_data
        self.debug_print(f"Processing {len(resumes_data)} resumes with AI-powered hybrid approach")
        
        self.candidate_sections = []
        
        for resume in resumes_data:
            # Parse resume into sections using AI
            resume_sections = self.parse_resume_sections_with_ai(resume['content'], resume['name'])
            self.candidate_sections.append(resume_sections)
            
            # Small delay to avoid rate limiting
            time.sleep(0.3)
        
        return True
    
    def calculate_similarities(self):
        """Calculate hybrid weighted similarities using AI-parsed sections"""
        if not hasattr(self, 'job_sections') or not self.candidate_sections:
            raise ValueError("Job description and resumes must be processed first")
        
        self.debug_print("Calculating hybrid weighted similarities...")
        
        final_similarities = []
        
        for i, candidate_sections in enumerate(self.candidate_sections):
            candidate_name = self.candidates[i]['name']
            self.debug_print(f"Processing {candidate_name}...")
            
            section_scores = {}
            
            # Calculate similarity for each section
            for section_name, weight in self.section_weights.items():
                self.debug_print(f"  >>> Starting {section_name} section processing")
                
                # Check if sections exist
                if section_name not in self.job_sections:
                    self.debug_print(f"  >>> {section_name} missing from job_sections. Available: {list(self.job_sections.keys())}")
                    continue
                    
                if section_name not in candidate_sections:
                    self.debug_print(f"  >>> {section_name} missing from candidate_sections. Available: {list(candidate_sections.keys())}")
                    continue
                
                # Get section texts
                job_section_text = self.job_sections[section_name]
                resume_section_text = candidate_sections[section_name]
                
                self.debug_print(f"  >>> Job {section_name} ({len(job_section_text)} chars): {job_section_text[:100]}...")
                self.debug_print(f"  >>> Resume {section_name} ({len(resume_section_text)} chars): {resume_section_text[:100]}...")
                
                # Skip if either section is too short or "not specified"
                if len(job_section_text) < 5:
                    self.debug_print(f"  >>> Skipping {section_name}: job section too short ({len(job_section_text)} chars)")
                    continue
                    
                if len(resume_section_text) < 5:
                    self.debug_print(f"  >>> Skipping {section_name}: resume section too short ({len(resume_section_text)} chars)")
                    continue
                    
                if "not specified" in job_section_text.lower():
                    self.debug_print(f"  >>> Skipping {section_name}: job section says 'not specified'")
                    continue
                    
                if "not specified" in resume_section_text.lower():
                    self.debug_print(f"  >>> Skipping {section_name}: resume section says 'not specified'")
                    continue
                
                try:
                    # Get embeddings for this section
                    self.debug_print(f"  >>> Getting embeddings for {section_name}...")
                    section_embeddings = self.get_sentence_transformer_embeddings([job_section_text, resume_section_text])
                    
                    if section_embeddings is None:
                        self.debug_print(f"  >>> ERROR: embeddings returned None for {section_name}")
                        continue
                    
                    if section_embeddings.shape[0] != 2:
                        self.debug_print(f"  >>> ERROR: embeddings shape wrong for {section_name}: {section_embeddings.shape}")
                        continue
                    
                    self.debug_print(f"  >>> Embeddings successful for {section_name}: {section_embeddings.shape}")
                    
                    job_emb = section_embeddings[0:1]
                    resume_emb = section_embeddings[1:2]
                    
                    self.debug_print(f"  >>> Calculating cosine similarity for {section_name}...")
                    
                    # Calculate cosine similarity for this section
                    section_similarity = cosine_similarity(job_emb, resume_emb)[0][0]
                    
                    self.debug_print(f"  >>> Cosine similarity calculated for {section_name}: {section_similarity}")
                    
                    section_scores[section_name] = {
                        'similarity': section_similarity,
                        'weight': weight
                    }
                    
                    self.debug_print(f"  >>> SUCCESS: {section_name}: {section_similarity:.3f} (weight: {weight})")

                except Exception as e:
                    self.debug_print(f"  >>> EXCEPTION in {section_name}: {type(e).__name__}: {str(e)}")
                    import traceback
                    self.debug_print(f"  >>> Traceback: {traceback.format_exc()}")
                    continue
                
                # Small delay between API calls
                time.sleep(0.3)
            
            # Calculate weighted final score
            self.debug_print(f"  >>> Section scores collected: {list(section_scores.keys())}")
            
            if section_scores:
                self.debug_print(f"  >>> Calculating final weighted score...")
                
                weighted_sum = sum(
                    scores['similarity'] * scores['weight'] 
                    for scores in section_scores.values()
                )
                total_weight = sum(scores['weight'] for scores in section_scores.values())
                final_similarity = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Show detailed breakdown
                self.debug_print(f"  >>> Section breakdown for {candidate_name}:")
                for section, scores in section_scores.items():
                    contribution = scores['similarity'] * scores['weight']
                    self.debug_print(f"    {section}: {scores['similarity']:.3f} × {scores['weight']} = {contribution:.3f}")
                self.debug_print(f"  >>> Weighted sum: {weighted_sum:.3f} / Total weight: {total_weight:.3f}")
                
            else:
                # Fallback: use overall document similarity
                self.debug_print(f"  >>> No section matches found for {candidate_name}, using fallback")
                try:
                    job_emb = self.get_sentence_transformer_embeddings([self.job_text])
                    resume_emb = self.get_sentence_transformer_embeddings([self.candidates[i]['content']])
                    if job_emb is not None and resume_emb is not None:
                        final_similarity = cosine_similarity(job_emb, resume_emb)[0][0]
                        self.debug_print(f"  >>> Fallback similarity: {final_similarity:.3f}")
                    else:
                        final_similarity = 0
                        self.debug_print(f"  >>> Fallback failed: embeddings were None")
                except Exception as e:
                    self.debug_print(f"  >>> Fallback similarity failed: {e}")
                    final_similarity = 0
            
            final_similarities.append(final_similarity)
            self.debug_print(f"  >>> Final weighted score for {candidate_name}: {final_similarity:.3f}")
        
        similarities = np.array(final_similarities)
        
        self.debug_print(f"AI-powered hybrid similarity stats: min={similarities.min():.4f}, max={similarities.max():.4f}, mean={similarities.mean():.4f}")
        
        return similarities
    
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
            
            As a technical recruiter, write a 2-sentence assessment describing why is this person a great fit for this role, also mention what are some of the gaps between RESUME and JOB REQUIREMENTS. Focus on specific technical skills, experience level, and key strengths or gaps.
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
        
    def get_sentence_transformer_embeddings(self, texts):
        """Get embeddings using a SentenceTransformer model with text enhancement"""
        self.debug_print(f"Getting SentenceTransformer embeddings for {len(texts)} texts")

        try:
            # Load a local or pre-trained SentenceTransformer model
            if not hasattr(self, 'st_model'):
                self.st_model = SentenceTransformer("all-mpnet-base-v2")  # You can also use: 'sentence-transformers/all-MiniLM-L6-v2'

            # Enhance texts for better semantic similarity
            enhanced_texts = []
            for i, text in enumerate(texts):
                clean_text = self._clean_extracted_text(text)
                enhanced_text = self.enhance_text_for_matching(clean_text, is_job_description=(i == 0))
                
                # SentenceTransformer handles token limits internally, but truncate if needed
                if len(enhanced_text) > 10000:
                    enhanced_text = enhanced_text[:10000]
                
                enhanced_texts.append(enhanced_text)
                self.debug_print(f"Text {i}: enhanced from {len(clean_text)} to {len(enhanced_text)} chars")

            # Generate embeddings
            embeddings = self.st_model.encode(
                enhanced_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Optional: normalize vectors for cosine similarity
            )

            self.debug_print(f"SentenceTransformer embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            self.debug_print(f"SentenceTransformer embeddings failed: {e}")
            return None