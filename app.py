import streamlit as st
import os
import tempfile
import pandas as pd
from rank import run_ranking

st.set_page_config(
    page_title="Redrob AI Recruiter - Candidate Ranker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background: #1f2937;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Redrob AI Recruiter — Candidate Ranking Sandbox")
st.write("An enterprise-grade hybrid scoring system leveraging sentence-transformer semantic embeddings, experience decay profiles, employer tiering, and active availability signals to rank the best candidates for your job description.")

st.markdown("---")

uploaded_file = st.file_uploader("📂 Upload Candidate Pool (JSON / JSONL format)", type=["jsonl", "json"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_input:
        temp_input.write(uploaded_file.getvalue())
        temp_input_path = temp_input.name
        
    temp_output_path = temp_input_path.replace(".jsonl", "_out.csv")
    temp_detailed_path = temp_output_path.replace(".csv", "_detailed.csv")
    
    with st.spinner("🧠 Analyzing resumes, filtering honeypots, and scoring semantic fit..."):
        try:
            # Run the ranking engine
            run_ranking(temp_input_path, temp_output_path)
            
            st.success("🎉 Ranking finished successfully!")
            
            # Read outputs
            sub_df = pd.read_csv(temp_output_path)
            
            # Read detailed output if it exists
            has_detailed = os.path.exists(temp_detailed_path)
            if has_detailed:
                detailed_df = pd.read_csv(temp_detailed_path)
            else:
                detailed_df = sub_df
            
            # --- Dashboard Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(sub_df)}</div><div class="metric-label">Shortlisted Candidates</div></div>', unsafe_allow_html=True)
            with col2:
                highest_score = sub_df["score"].max() if len(sub_df) > 0 else 0.0
                st.markdown(f'<div class="metric-card"><div class="metric-value">{highest_score:.4f}</div><div class="metric-label">Highest Score</div></div>', unsafe_allow_html=True)
            with col3:
                lowest_score = sub_df["score"].min() if len(sub_df) > 0 else 0.0
                st.markdown(f'<div class="metric-card"><div class="metric-value">{lowest_score:.4f}</div><div class="metric-label">Lowest Score</div></div>', unsafe_allow_html=True)
            with col4:
                # Count average score
                avg_score = sub_df["score"].mean() if len(sub_df) > 0 else 0.0
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_score:.4f}</div><div class="metric-label">Average Score</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Downloads Section ---
            st.subheader("📥 Export Results")
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                with open(temp_output_path, "r", encoding="utf-8") as f:
                    csv_data = f.read()
                st.download_button(
                    label="Download Standard Submission CSV (4-columns)",
                    data=csv_data,
                    file_name="submission.csv",
                    mime="text/csv",
                    help="Compliant 4-column CSV file required for portal upload (candidate_id, rank, score, reasoning)."
                )
                
            with d_col2:
                if has_detailed:
                    with open(temp_detailed_path, "r", encoding="utf-8") as f:
                        detailed_data = f.read()
                    st.download_button(
                        label="Download Detailed Recruiter CSV (12-columns)",
                        data=detailed_data,
                        file_name="submission_detailed.csv",
                        mime="text/csv",
                        help="Detailed CSV with full sub-score breakdowns (semantic match, skills, exp, location, company tiers) and recruiter flags."
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)

            # --- Presentation Tabs ---
            tab1, tab2 = st.tabs(["🏆 Ranked Shortlist (Expander View)", "📊 Detailed Data Table"])
            
            with tab1:
                st.write("Browse candidates with full, auto-wrapped reasoning statements:")
                for idx, row in sub_df.iterrows():
                    rank = row["rank"]
                    cid = row["candidate_id"]
                    score = row["score"]
                    reasoning = row["reasoning"]
                    
                    label = f"🏆 Rank {rank} | ID: {cid} | Score: {score:.4f}" if rank <= 10 else f"Rank {rank} | ID: {cid} | Score: {score:.4f}"
                    with st.expander(label, expanded=(rank <= 5)):
                        st.markdown(f"**Reasoning:**")
                        st.info(reasoning)
                        
            with tab2:
                st.write("Full tabular dataset (scroll and filter):")
                st.dataframe(
                    detailed_df,
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            if os.path.exists(temp_detailed_path):
                os.remove(temp_detailed_path)
