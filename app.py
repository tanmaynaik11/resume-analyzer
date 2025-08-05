# app.py - Updated Streamlit UI with Text Input for Resumes
import streamlit as st
import numpy as np
import os
from backend_engine import CandidateRecommendationEngine

def main():
    # Page configuration
    st.set_page_config(
        page_title="Candidate Recommendation Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .candidate-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #10b981;
    }
    .similarity-high { color: #059669; font-weight: bold; }
    .similarity-medium { color: #d97706; font-weight: bold; }
    .similarity-low { color: #dc2626; font-weight: bold; }
    .resume-input-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .settings-box {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #0ea5e9;
        margin: 1rem 0;
    }
    .control-panel {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .candidate-rank-1 { border-left-color: #ffd700; }
    .candidate-rank-2 { border-left-color: #c0c0c0; }
    .candidate-rank-3 { border-left-color: #cd7f32; }
    .filter-section {
        background: #fefefe;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Candidate Recommendation Engine</h1>
        <p>AI-powered candidate matching with intelligent summaries</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.code("OPENAI_API_KEY=your_openai_api_key_here")
        st.info("💡 You can get your OpenAI API key from: https://platform.openai.com/api-keys")
        st.stop()
    
    # Initialize the engine
    if 'engine' not in st.session_state:
        st.session_state.engine = CandidateRecommendationEngine()
    
    # Initialize resume storage
    if 'manual_resumes' not in st.session_state:
        st.session_state.manual_resumes = []
    if 'uploaded_resumes' not in st.session_state:
        st.session_state.uploaded_resumes = []
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Enter the job description:",
            height=250,
            placeholder="Paste your job description here...",
            help="Enter the complete job description including requirements, responsibilities, and qualifications."
        )
        
        # Analysis Settings Section
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Analysis Settings")
        
        col_setting1, col_setting2 = st.columns(2)
        
        with col_setting1:
            top_n_candidates = st.slider(
                "Number of top candidates to show:",
                min_value=1,
                max_value=20,
                value=5,
                help="Select how many top-ranked candidates you want to see in the results."
            )
        
        with col_setting2:
            # Option to show all candidates
            show_all = st.checkbox(
                "Show all candidates",
                value=False,
                help="Check this to show all candidates regardless of the slider setting above."
            )
        
        # Filter options
        st.markdown("#### 🔍 Result Filters")
        sort_order = st.selectbox(
            "Sort results by:",
            ["Highest Score First", "Lowest Score First", "Alphabetical"],
            index=0,
            help="Choose how to sort the candidate results."
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("👥 Add Candidate Resumes")
        
        # Resume input method selection
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload Files", "✏️ Manual Text Entry"],
            horizontal=True
        )
        
        if input_method == "📄 Upload Files":
            # File upload section
            uploaded_files = st.file_uploader(
                "Upload candidate resumes:",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT. You can upload multiple files at once."
            )
            
            # Process uploaded files
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
                
                # Store uploaded files in session state
                st.session_state.uploaded_resumes = []
                for uploaded_file in uploaded_files:
                    try:
                        content = st.session_state.engine.extract_text_from_file(uploaded_file)
                        if content.strip():
                            name = uploaded_file.name.rsplit('.', 1)[0]
                            st.session_state.uploaded_resumes.append({
                                'name': name,
                                'content': content,
                                'source': 'file'
                            })
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Show uploaded files
                with st.expander("📁 View uploaded files"):
                    for resume in st.session_state.uploaded_resumes:
                        st.write(f"• **{resume['name']}** ({len(resume['content'])} characters)")
        
        else:
            # Manual text entry section
            st.markdown("### ✏️ Add Resumes Manually")
            
            # Form for adding individual resumes
            with st.form("add_resume_form", clear_on_submit=True):
                candidate_name = st.text_input(
                    "Candidate Name:",
                    placeholder="e.g., John Smith"
                )
                
                resume_text = st.text_area(
                    "Resume Content:",
                    height=200,
                    placeholder="Paste the complete resume content here..."
                )
                
                col1_form, col2_form = st.columns([1, 3])
                with col1_form:
                    add_resume = st.form_submit_button("➕ Add Resume", type="primary")
                with col2_form:
                    if st.form_submit_button("🗑️ Clear All Manual Resumes"):
                        st.session_state.manual_resumes = []
                        st.rerun()
                
                if add_resume:
                    if candidate_name.strip() and resume_text.strip():
                        # Check for duplicates
                        existing_names = [r['name'] for r in st.session_state.manual_resumes]
                        if candidate_name in existing_names:
                            st.error(f"❌ Candidate '{candidate_name}' already exists. Please use a different name or remove the existing entry.")
                        else:
                            st.session_state.manual_resumes.append({
                                'name': candidate_name.strip(),
                                'content': resume_text.strip(),
                                'source': 'manual'
                            })
                            st.success(f"✅ Added {candidate_name} to the candidate list!")
                            st.rerun()
                    else:
                        st.error("❌ Please provide both candidate name and resume content.")
            
            # Display manually added resumes
            if st.session_state.manual_resumes:
                st.markdown("### 📋 Manually Added Candidates")
                
                for i, resume in enumerate(st.session_state.manual_resumes):
                    with st.expander(f"👤 {resume['name']} ({len(resume['content'])} characters)"):
                        st.text_area(
                            f"Resume content:",
                            resume['content'],
                            height=150,
                            disabled=True,
                            key=f"manual_resume_preview_{i}"
                        )
                        
                        # Remove button for individual resumes
                        if st.button(f"🗑️ Remove {resume['name']}", key=f"remove_{i}"):
                            st.session_state.manual_resumes.pop(i)
                            st.rerun()
    
    # Combine all resumes
    all_resumes = st.session_state.uploaded_resumes + st.session_state.manual_resumes
    
    # Show total count with improved display
    if all_resumes:
        total_files = len(st.session_state.uploaded_resumes)
        total_manual = len(st.session_state.manual_resumes)
        
        # Determine how many will be shown
        if show_all:
            candidates_to_show = len(all_resumes)
            display_text = f"Will show: **All {len(all_resumes)} candidates**"
        else:
            candidates_to_show = min(top_n_candidates, len(all_resumes))
            display_text = f"Will show: **Top {candidates_to_show} of {len(all_resumes)} candidates**"
        
        st.markdown(f"""
        <div class="filter-section">
        📊 <strong>Candidate Summary:</strong> {len(all_resumes)} total 
        (📄 {total_files} uploaded, ✏️ {total_manual} manual)<br>
        🎯 {display_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 Analyze Candidates",
            type="primary",
            disabled=not (job_description.strip() and all_resumes),
            use_container_width=True,
            help="Click to start the AI-powered candidate analysis"
        )
    
    # Show requirements if inputs are missing
    if not job_description.strip() or not all_resumes:
        missing_items = []
        if not job_description.strip():
            missing_items.append("job description")
        if not all_resumes:
            missing_items.append("candidate resumes")
        
        st.info(f"📋 Please provide {' and '.join(missing_items)} to begin analysis.")
    
    # Analysis and Results
    if analyze_button:
        if not job_description.strip():
            st.error("Please enter a job description.")
        elif not all_resumes:
            st.error("Please add candidate resumes.")
        else:
            with st.spinner("🤖 Analyzing candidates and generating AI summaries..."):
                try:
                    # Process job description
                    st.session_state.engine.process_job_description(job_description)
                    
                    # Process resumes
                    st.session_state.engine.process_resumes(all_resumes)
                    
                    # Determine how many candidates to get
                    if show_all:
                        candidates_to_get = len(all_resumes)
                        display_message = f"Showing all {len(all_resumes)} candidates"
                    else:
                        candidates_to_get = min(top_n_candidates, len(all_resumes))
                        display_message = f"Showing top {candidates_to_get} of {len(all_resumes)} candidates"
                    
                    # Get top candidates with AI summaries - FIXED: Using candidates_to_get instead of len(all_resumes)
                    top_candidates = st.session_state.engine.get_top_candidates(
                        job_description, 
                        top_n=candidates_to_get
                    )
                    
                    # Apply sorting only (removed min_score_filter)
                    filtered_candidates = top_candidates.copy()
                    
                    if sort_order == "Lowest Score First":
                        filtered_candidates.sort(key=lambda x: x['similarity_score'])
                    elif sort_order == "Alphabetical":
                        filtered_candidates.sort(key=lambda x: x['name'].lower())
                    # Default is already "Highest Score First" from get_top_candidates
                    
                    # Store results in session state
                    st.session_state.results = filtered_candidates
                    st.session_state.job_desc = job_description
                    st.session_state.total_candidates = len(all_resumes)
                    st.session_state.shown_candidates = len(filtered_candidates)
                    st.session_state.requested_candidates = candidates_to_get
                    
                    if filtered_candidates:
                        st.success(f"✅ Analysis completed! {display_message}.")
                    else:
                        st.warning(f"⚠️ No candidates found. This shouldn't happen - please check your inputs.")
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
                    st.exception(e)  # Show full error for debugging
    
    # Display Results
    if 'results' in st.session_state and st.session_state.results:
        st.markdown("---")
        st.header("🏆 Candidate Rankings")
        
        # Results overview metrics
        st.subheader("📊 Analysis Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        results = st.session_state.results
        total_candidates = st.session_state.get('total_candidates', len(results))
        shown_candidates = st.session_state.get('shown_candidates', len(results))
        requested_candidates = st.session_state.get('requested_candidates', len(results))
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{shown_candidates}</h3>
                <p>Candidates Shown</p>
                <small>of {total_candidates} total</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if results:
                avg_score = np.mean([r['similarity_score'] for r in results])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_score:.0%}</h3>
                    <p>Average Score</p>
                    <small>Top {len(results)} candidates</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if results:
                best_score = results[0]['similarity_score'] if sort_order != "Lowest Score First" else max(r['similarity_score'] for r in results)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{best_score:.0%}</h3>
                    <p>Best Match</p>
                    <small>Highest score found</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            high_matches = len([r for r in results if r['similarity_score'] >= 0.6])
            st.markdown(f"""
            <div class="metric-card">
                <h3>{high_matches}</h3>
                <p>Strong Matches</p>
                <small>≥ 60% score</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display candidate cards
        for i, candidate in enumerate(results):
            # Determine similarity level for styling
            score = candidate['similarity_score']
            if score >= 0.7:
                score_class = "similarity-high"
                score_emoji = "🎯"
                rank_class = "candidate-rank-1" if i == 0 else ""
            elif score >= 0.5:
                score_class = "similarity-medium"
                score_emoji = "⚡"
                rank_class = "candidate-rank-2" if i == 1 else ""
            else:
                score_class = "similarity-low"
                score_emoji = "💡"
                rank_class = "candidate-rank-3" if i == 2 else ""
            
            # Create candidate card with dynamic ranking
            rank_badge = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            
            with st.container():
                st.markdown(f"""
                <div class="candidate-card {rank_class}">
                    <h3>{score_emoji} {rank_badge} {candidate['name']}</h3>
                    <p class="{score_class}">Match Score: {score:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # AI Summary
                if candidate.get('ai_summary'):
                    st.markdown(f"**🤖 AI Assessment:**")
                    st.info(candidate['ai_summary'])
                else:
                    st.warning("AI summary not available for this candidate.")
                
                # Additional candidate actions
                col_action1, col_action2, col_action3 = st.columns([2, 1, 1])
                
                with col_action1:
                    # Resume preview in expander
                    with st.expander(f"📄 View {candidate['name']}'s Resume"):
                        st.text_area(
                            "Resume Content:",
                            candidate['content'],
                            height=200,
                            disabled=True,
                            key=f"resume_content_{i}"
                        )
                
                with col_action2:
                    if st.button(f"⭐ Shortlist", key=f"shortlist_{i}", help=f"Add {candidate['name']} to shortlist"):
                        st.success(f"✅ {candidate['name']} added to shortlist!")
                
                with col_action3:
                    if st.button(f"📧 Contact", key=f"contact_{i}", help=f"Contact {candidate['name']}"):
                        st.info(f"📧 Contact information for {candidate['name']} would be shown here.")
                
                st.markdown("---")
        
        # Enhanced Export and Actions Section
        st.subheader("💾 Export & Actions")
        
        col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
        
        with col_export1:
            if st.button("📥 Download Detailed Report"):
                # Create downloadable content
                results_text = f"CANDIDATE RECOMMENDATION RESULTS\n"
                results_text += f"="*60 + "\n"
                results_text += f"Analysis Date: {st.session_state.get('analysis_date', 'N/A')}\n"
                results_text += f"Job Description Preview: {job_description[:100]}...\n"
                results_text += f"Total Candidates Analyzed: {total_candidates}\n"
                results_text += f"Candidates Shown: {shown_candidates}\n"
                results_text += f"Average Score: {np.mean([r['similarity_score'] for r in results]):.1%}\n"
                results_text += f"Sort Order: {sort_order}\n"
                results_text += "="*60 + "\n\n"
                
                for i, candidate in enumerate(results):
                    results_text += f"RANK #{i+1}: {candidate['name']}\n"
                    results_text += f"Match Score: {candidate['similarity_score']:.1%}\n"
                    results_text += f"AI Assessment: {candidate.get('ai_summary', 'Not available')}\n"
                    results_text += f"Resume Length: {len(candidate['content'])} characters\n"
                    results_text += "-"*40 + "\n\n"
                
                st.download_button(
                    label="📄 Download Full Report",
                    data=results_text,
                    file_name=f"candidate_recommendations_top_{shown_candidates}.txt",
                    mime="text/plain"
                )
        
        with col_export2:
            if st.button("📊 Export CSV Summary"):
                import pandas as pd
                
                # Create CSV data
                csv_data = []
                for i, candidate in enumerate(results):
                    csv_data.append({
                        'Rank': i + 1,
                        'Name': candidate['name'],
                        'Score': f"{candidate['similarity_score']:.1%}",
                        'Score_Numeric': candidate['similarity_score'],
                        'AI_Summary': candidate.get('ai_summary', 'Not available')[:200] + '...' if candidate.get('ai_summary') and len(candidate.get('ai_summary', '')) > 200 else candidate.get('ai_summary', 'Not available'),
                        'Resume_Length': len(candidate['content'])
                    })
                
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_string,
                    file_name=f"candidates_summary_top_{shown_candidates}.csv",
                    mime="text/csv"
                )
        
        with col_export3:
            if st.button("🔄 Re-run Analysis"):
                # Clear results to force re-analysis
                if 'results' in st.session_state:
                    del st.session_state.results
                st.rerun()
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 Powered by AI | Built with Streamlit | 
        <a href='#' style='color: #4f46e5;'>Documentation</a> | 
        <a href='#' style='color: #4f46e5;'>Support</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()