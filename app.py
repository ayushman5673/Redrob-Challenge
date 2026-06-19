import streamlit as st
import os
import tempfile
import pandas as pd
from rank import run_ranking

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")

st.title("Redrob Candidate Ranker Sandbox")
st.write("Upload a sample candidate JSONL file (typically 100 or fewer candidates) to run the ranking model offline and download the validated CSV.")

uploaded_file = st.file_uploader("Upload candidates.jsonl", type=["jsonl", "json"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_input:
        temp_input.write(uploaded_file.getvalue())
        temp_input_path = temp_input.name
        
    temp_output_path = temp_input_path.replace(".jsonl", "_out.csv")
    
    with st.spinner("Processing candidate filters and running semantic scoring..."):
        try:
            # Run the ranking engine
            run_ranking(temp_input_path, temp_output_path)
            
            st.success("Ranking finished successfully!")
            
            # Read outputs
            sub_df = pd.read_csv(temp_output_path)
            
            st.subheader("Top Ranked Candidates")
            st.dataframe(sub_df)
            
            # Read file contents for download button
            with open(temp_output_path, "r", encoding="utf-8") as f:
                csv_data = f.read()
                
            st.download_button(
                label="Download submission.csv",
                data=csv_data,
                file_name="submission.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Ranking failed: {e}")
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            detailed_path = temp_output_path.replace(".csv", "_detailed.csv")
            if os.path.exists(detailed_path):
                os.remove(detailed_path)
