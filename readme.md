# Resume-Job Description Matching System

A sophisticated AI-powered system that matches resumes with job descriptions using advanced embedding techniques with Holistic or hybrid scoring approaches. The system provides both holistic resume analysis and section-wise detailed matching for comprehensive candidate evaluation.

## 🚀 Features

- **Dual Matching Approaches**: 
  - Holistic resume-JD embedding matching
  - Section-wise hybrid analysis (skills, experience, education, etc.)
- **Interactive Streamlit UI**: User-friendly web interface for easy interaction
- **Advanced NLP**: Leverages state-of-the-art language models for semantic understanding
- **Comprehensive Scoring**: Multiple scoring metrics for detailed candidate assessment
- **Export Capabilities**: Generate reports and export results

## 🚀 Important considerations or insigths
- Considering the context length of input JD and resume choose the embedding model. In Our case resumes might
  go beyond 1024 token length so chose open ai large embedding model(8191 token length) to handle large context resumes.
- For section wise embedding approach we can use sentence transformers embedding models or smaller open ai embedding model
- For creating AI summary we are using Open AI's gpt-3.5-turbo, for cost effeciency we can level open source SLM finetuned
  on summarization task.
- Used Open ai embedding model in holistic approach and for section wise approach used sentence transformers all-mpnet-base-v2 model.  

## 📁 Project Structure

```
resume-jd-matcher/
├── app.py                  # Streamlit web application UI
├── backend_engine.py       # Core backend with holistic embedding-based matching
├── candidate_engine.py     # Hybrid section-wise embedding matching engine
├── requirements.txt        # dependencies
├── README.md               # Project documentation
├── trial_resumes/          # Data storage directory
│   ├── resumes/            # Upload resume files
│── trial_jd                # Job description files
├── .env/                   # Open AI API Key 

```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p trial_resumes/resumes..
   ```

## 🚀 Usage

### Starting the Application

1. **Launch Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The Streamlit interface will be available for interaction

### Using the System

1. **Upload Documents**
   - Upload resume files (PDF, DOCX) or Input resume in TXT format.
   - Input descriptions


2. **View Results**
   - Name of Candidate
   - Similarity Score
   - Summary describing why is this person a great fit for this role
   - Export results as needed

## 🔧 Core Components

### app.py - Streamlit UI
The main user interface providing:
- File upload functionality
- Interactive controls
- Results visualization
- Export capabilities
- User-friendly navigation

### backend_engine.py - Embedding Engine
Core backend functionality including:
- Resume and JD text extraction
- Embedding generation using open ai embedding model
- Similarity calculation algorithms
- API endpoints for UI communication

### candidate_engine.py - Hybrid Matcher
Advanced section-wise analysis featuring:
- Resume parsing into sections (skills, experience, education)
- Individual section matching with JD requirements
- Weighted scoring system


## ⚙️ Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY= YOUR OPENAI KEY
```

## 📊 Matching Algorithms

### Holistic Approach (backend_engine.py)
1. **Text Preprocessing**: Clean and normalize resume/JD text
2. **Embedding Generation**: Convert text to dense vector representations
3. **Similarity Calculation**: Cosine similarity between embeddings
4. **Score Normalization**: Convert to percentage scores

### Section-wise Approach (candidate_engine.py)
1. **Resume Parsing**: Extract sections (skills, experience, education, etc.)
2. **Requirement Mapping**: Match JD requirements to resume sections
3. **Individual Scoring**: Score each section independently
4. **Weighted Aggregation**: Combine scores with configurable weights


### Scaling Considerations
- Implement caching for frequently accessed embeddings
- Use GPU acceleration for large-scale processing
- Consider distributed processing for enterprise use

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with using Python, Streamlit, and OpenAI**