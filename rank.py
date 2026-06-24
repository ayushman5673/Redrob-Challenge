import os
import sys
import json
import math
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Optimize PyTorch CPU inference threads to avoid context switching thrashing
torch.set_num_threads(4)

CURRENT_DATE = datetime.date(2026, 6, 18)

# Core company founding rules
FOUNDING_RULES = {
    "Krutrim": {"min_start_year": 2023, "max_duration_months": 30},
    "Sarvam AI": {"min_start_year": 2023, "max_duration_months": 35},
    "CRED": {"min_start_year": 2018, "max_duration_months": 91},
    "Aganitha": {"min_start_year": 2020, "max_duration_months": 78},
    "Glance": {"min_start_year": 2019, "max_duration_months": 87},
    "Rephrase.ai": {"min_start_year": 2019, "max_duration_months": 90}
}

# Trap titles
TRAP_TITLES = {
    "hr manager", "talent acquisition", "recruiter",
    "accountant", "finance analyst", "cfo",
    "marketing manager", "brand manager", "growth manager",
    "graphic designer", "ui designer",
    "customer support", "operations manager", "sales executive",
    "logistics coordinator", "supply chain manager"
}

# Consulting / Service firms
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "capgemini",
    "cognizant", "tech mahindra", "hcl", "genpact", "mphasis"
}

# Predefined product tiers for S_emp
PRODUCT_TIERS = {
    # Tier 1 (1.00): Top AI / Product
    "google": 1.00, "meta": 1.00, "openai": 1.00, "cohere": 1.00, 
    "anthropic": 1.00, "juspay": 1.00, "cred": 1.00, "phonepe": 1.00, 
    "deepmind": 1.00,
    # Tier 2 (0.80): Strong product
    "flipkart": 0.80, "swiggy": 0.80, "zomato": 0.80, "razorpay": 0.80, 
    "postman": 0.80, "browserstack": 0.80, "freshworks": 0.80,
}

# JD skill clusters
SKILL_CLUSTERS = {
    "vector_db": {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma", "pgvector"},
    "embedding_models": {"sentence-transformers", "text-embedding-ada", "e5", "bge", "all-minilm"},
    "nlp_ir": {"information retrieval", "bm25", "elasticsearch", "solr", "rag", "colbert"},
    "ml_training": {"pytorch", "tensorflow", "jax", "keras", "scikit-learn", "xgboost", "deep learning"},
    "mlops_deployment": {"docker", "kubernetes", "fastapi", "ray serve", "triton", "mlflow", "dvc"}
}

def clean_name(name):
    if not name:
        return ""
    return "".join(c for c in name.lower() if c.isalnum())

CLEAN_CLUSTERS = {
    cluster: {clean_name(s) for s in skills}
    for cluster, skills in SKILL_CLUSTERS.items()
}

ALL_CRITICAL_CLEAN_SKILLS = set().union(*CLEAN_CLUSTERS.values())

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def get_job_duration_months(job):
    duration = job.get("duration_months")
    if duration is not None:
        return duration
    start = parse_date(job.get("start_date"))
    end = parse_date(job.get("end_date")) or CURRENT_DATE
    if start:
        return (end.year - start.year) * 12 + (end.month - start.month)
    return 0

def map_seniority(title):
    title = (title or "").lower()
    if any(w in title for w in ["director", "vp", "vice president", "head", "chief"]):
        return 6
    if any(w in title for w in ["principal", "lead", "architect"]):
        return 5
    if "staff" in title:
        return 4
    if any(w in title for w in ["senior", "sr.", "sr"]):
        return 3
    if any(w in title for w in ["junior", "associate", "jr.", "jr"]):
        return 1
    if any(w in title for w in ["intern", "trainee", "student"]):
        return 0
    return 2

def score_employer_tier(company_name, company_size, industry):
    comp_lower = (company_name or "").lower()
    
    # Predefined checks
    for key, val in PRODUCT_TIERS.items():
        if key in comp_lower:
            return val
            
    for key in CONSULTING_FIRMS:
        if key in comp_lower:
            return 0.25
            
    if any(w in comp_lower for w in ["university", "lab", "research", "iit", "iisc", "institute", "academia"]):
        return 0.20
        
    # Startup size heuristic
    size = company_size or "1-10"
    ind = (industry or "").lower()
    is_tech = any(w in ind for w in ["software", "fintech", "saas", "ai", "ml", "ecommerce", "internet", "technology"])
    
    if is_tech:
        if size in ["51-200", "201-500", "501-1000"]:
            return 0.60
        elif size in ["1-10", "11-50"]:
            return 0.50
        else:
            return 0.60
            
    return 0.50

def run_ranking(candidates_file, output_file):
    print(f"Reading candidates from {candidates_file}...")
    
    stage1_candidates = []
    
    # 1. Stage 1 — Coarse Filter & Heuristic Selection
    # Detect if file is a JSON array (starts with '[') or JSONL
    first_char = ''
    with open(candidates_file, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break

    def process_cand(cand):
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        # --- 1A. Physical Validity / Honeypots ---
        is_honeypot = False
        
        # Check 1: job duration exceeds total XP
        exp = profile.get("years_of_experience", 0)
        for job in career:
            dur_years = get_job_duration_months(job) / 12.0
            if dur_years > exp + 0.1:
                is_honeypot = True
                break
        if is_honeypot:
            return
            
        # Check 2: company founding violations
        for job in career:
            comp = job.get("company", "")
            start = job.get("start_date")
            dur = get_job_duration_months(job)
            if comp in FOUNDING_RULES:
                rule = FOUNDING_RULES[comp]
                if start:
                    try:
                        start_year = int(start[:4])
                        if start_year < rule["min_start_year"]:
                            is_honeypot = True
                            break
                    except:
                        pass
                if dur > rule["max_duration_months"]:
                    is_honeypot = True
                    break
        if is_honeypot:
            return
            
        # Check 3: expert skill inflation
        expert_0_dur = [s for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
        if len(expert_0_dur) >= 5:
            return
            
        # Check 4: overlapping current roles
        current_jobs = [job for job in career if job.get("is_current")]
        if len(current_jobs) > 1:
            return
            
        # Check 5: future start dates
        for job in career:
            start = job.get("start_date")
            if start:
                try:
                    s_date = parse_date(start)
                    if s_date and s_date > CURRENT_DATE:
                        is_honeypot = True
                        break
                    end = parse_date(job.get("end_date"))
                    if s_date and end and s_date > end:
                        is_honeypot = True
                        break
                except:
                    pass
        if is_honeypot:
            return
            
        # --- 1B. Trap Title Filter ---
        current_title = profile.get("current_title", "").lower()
        if any(t in current_title for t in TRAP_TITLES):
            return
            
        # --- 1C. Location Hard Block ---
        country = profile.get("country", "").lower()
        willing_relocate = profile.get("willing_to_relocate", None)
        if willing_relocate is None:
            willing_relocate = signals.get("willing_to_relocate", None)
            
        if "india" not in country:
            if willing_relocate is False or willing_relocate is None:
                return
        
        # --- Heuristic Selection (Filter top 1,500) ---
        # 1. Experience score heuristic (peaking at 6-8 years)
        if 6.0 <= exp <= 8.0:
            h_exp = 1.00
        elif (5.0 <= exp < 6.0) or (8.0 < exp <= 9.5):
            h_exp = 0.85
        elif (4.0 <= exp < 5.0) or (9.5 < exp <= 12.0):
            h_exp = 0.60
        else:
            h_exp = 0.20
            
        # 2. Critical skills match count
        cand_skills = {clean_name(s.get("name", "")) for s in skills}
        matched_skills = cand_skills & ALL_CRITICAL_CLEAN_SKILLS
        h_skills = len(matched_skills)
        
        # 3. Title match keywords (favoring AI/ML/Software engineers)
        h_title = 0.0
        if any(w in current_title for w in ["machine learning", "ml", "ai", "artificial intelligence", "nlp", "computer vision", "data scientist", "deep learning"]):
            h_title += 1.0
        elif "software engineer" in current_title or "backend engineer" in current_title or "developer" in current_title:
            h_title += 0.5
            
        # 4. Active & Response rates (availability)
        recruiter_rate = signals.get("recruiter_response_rate", 0.5)
        
        # Heuristic composite
        h_score = h_exp * 2.0 + h_skills * 1.5 + h_title * 2.0 + recruiter_rate * 1.0
        
        stage1_candidates.append((h_score, cand))

    if first_char == '[':
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_list = json.load(f)
        for cand in candidates_list:
            process_cand(cand)
    else:
        with open(candidates_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                cand = json.loads(line)
                process_cand(cand)
            
    print(f"Candidates passed hard filters: {len(stage1_candidates)}")
    print("Selecting top 1,500 candidates for Stage 2 scoring...")
    stage1_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Slice to top 1,500 candidates for dense embedding scoring
    top_candidates = [x[1] for x in stage1_candidates[:1500]]
    
    # 2. Stage 2 — Dense Semantic Encoding
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_cache_path = os.path.join(script_dir, "model_cache")
    print(f"Loading local SentenceTransformer model from {model_cache_path}...")
    model = SentenceTransformer(model_cache_path)
    
    jd_query = (
        "Senior AI Engineer Founding Team Series A AI talent intelligence platform Pune Noida India. "
        "Deep technical depth in modern ML systems: embeddings, sentence-transformers, dense vector search, "
        "hybrid search, information retrieval, RAG, vector databases (Pinecone, Weaviate, Qdrant, Milvus, FAISS), "
        "learning-to-rank LTR, ranking evaluation NDCG MRR MAP, LLM fine-tuning, PyTorch, strong Python system shipper."
    )
    jd_embedding = model.encode(jd_query, normalize_embeddings=True)
    
    # Text Representation (Short representation to optimize sequence length & speed)
    candidate_texts = []
    for cand in top_candidates:
        profile = cand.get("profile", {})
        skills = cand.get("skills", [])
        
        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        current_title = profile.get("current_title", "")
        
        skill_texts = ", ".join(s.get("name", "") for s in skills)
        text = f"{current_title} - {headline}. Summary: {summary}. Skills: {skill_texts}."
        candidate_texts.append(text)
        
    print("Encoding candidate text representations...")
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=64, # Optimized batch size
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    s_sem_list = candidate_embeddings @ jd_embedding
    
    # 3. Stage 2 — Deep Scoring Formulas
    scored_candidates = []
    
    for idx, cand in enumerate(top_candidates):
        cid = cand.get("candidate_id")
        profile = cand.get("profile", {})
        career = cand.get("career_history", [])
        skills = cand.get("skills", [])
        signals = cand.get("redrob_signals", {})
        
        # 1. Experience Score (S_exp)
        exp = profile.get("years_of_experience", 0)
        if 6.0 <= exp <= 8.0:
            s_exp = 1.00
        elif (5.0 <= exp < 6.0) or (8.0 < exp <= 9.5):
            s_exp = 0.85
        elif (4.0 <= exp < 5.0) or (9.5 < exp <= 12.0):
            s_exp = 0.60
        elif (3.0 <= exp < 4.0) or (12.0 < exp <= 14.0):
            s_exp = 0.35
        else:
            s_exp = 0.10
            
        # 2. Location Score (S_loc)
        country = profile.get("country", "").lower()
        city = profile.get("location", "").lower()
        willing_relocate = profile.get("willing_to_relocate", None)
        if willing_relocate is None:
            willing_relocate = signals.get("willing_to_relocate", False)
            
        is_primary_city = any(c in city for c in ["pune", "noida", "delhi", "mumbai", "hyderabad", "gurgaon", "ncr"])
        is_secondary_city = any(c in city for c in ["bangalore", "chennai", "kolkata"])
        
        if "india" in country:
            if is_primary_city:
                s_loc = 1.00
            elif is_secondary_city:
                s_loc = 1.00 if willing_relocate is True or willing_relocate is None else 0.65
            else: # Tier 2/3
                s_loc = 0.85 if willing_relocate is True else 0.30
        else: # Outside India but willing to relocate
            s_loc = 0.60
            
        # 3. Employer Quality Score (S_emp)
        total_career_months = sum(get_job_duration_months(job) for job in career)
        if total_career_months == 0:
            s_emp = 0.50
        else:
            weighted_emp_score = 0.0
            for job in career:
                dur = get_job_duration_months(job)
                tier_score = score_employer_tier(
                    job.get("company", ""),
                    job.get("company_size", "1-10"),
                    job.get("industry", "")
                )
                weighted_emp_score += tier_score * dur
            s_emp = weighted_emp_score / total_career_months
            
        # 4. Semantic Score (S_sem)
        s_sem = max(0.0, float(s_sem_list[idx]))
        
        # 5. Skills Score (S_skill)
        skill_last_used_years = {}
        for s in skills:
            sname = s.get("name", "")
            sname_clean = clean_name(sname)
            
            # Default to 2.0 years since last use (treated as 2 years stale)
            years_since_use = 2.0
            
            # Search career history for matches
            latest_end_date = None
            for job in career:
                title = job.get("title", "").lower()
                desc = job.get("description", "").lower()
                if (sname_clean in clean_name(title)) or (sname_clean in clean_name(desc)):
                    end_d = parse_date(job.get("end_date"))
                    if job.get("is_current") or end_d is None:
                        latest_end_date = CURRENT_DATE
                        break
                    elif latest_end_date is None or end_d > latest_end_date:
                        latest_end_date = end_d
            
            if latest_end_date:
                years_since_use = (CURRENT_DATE - latest_end_date).days / 365.25
                if years_since_use < 0:
                    years_since_use = 0
            skill_last_used_years[sname_clean] = years_since_use
            
        proficiency_weights = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}
        
        cluster_scores = {}
        for cluster, clean_cluster_skills in CLEAN_CLUSTERS.items():
            cluster_skill_scores = []
            for s in skills:
                sname_clean = clean_name(s.get("name", ""))
                if sname_clean in clean_cluster_skills:
                    prof = s.get("proficiency", "beginner")
                    pw = proficiency_weights.get(prof, 0.2)
                    dur = s.get("duration_months", 0)
                    
                    years_since_use = skill_last_used_years.get(sname_clean, 2.0)
                    recency_w = math.exp(-0.15 * years_since_use)
                    
                    effective_skill_score = pw * recency_w * min(dur, 36) / 36.0
                    cluster_skill_scores.append(effective_skill_score)
            cluster_scores[cluster] = min(sum(cluster_skill_scores), 1.0) if cluster_skill_scores else 0.0
            
        s_skill = sum(cluster_scores.values()) / 5.0
        
        # --- Stage 2 Exclusions / Penalties ---
        flags = []
        
        # 2A. Consulting Career Penalty
        months_at_consulting = sum(get_job_duration_months(job) for job in career if any(c in job.get("company", "").lower() for c in CONSULTING_FIRMS))
        consulting_fraction = months_at_consulting / total_career_months if total_career_months > 0 else 0.0
        consulting_penalty = 1.0 - (0.5 * consulting_fraction)
        if consulting_fraction > 0.0:
            flags.append("consulting_penalty")
            
        # 2B. Pure Academic / Research Penalty
        is_academic_only = True
        for job in career:
            comp = job.get("company", "").lower()
            if not any(w in comp for w in ["university", "lab", "research", "iit", "iisc", "institute", "academia"]):
                is_academic_only = False
                break
        
        disqualify_academic = False
        if is_academic_only and len(career) > 0:
            github_score = signals.get("github_activity_score", 0)
            summary_lower = profile.get("summary", "").lower()
            has_production_sig = any(w in summary_lower for w in ["production", "deploy", "scale", "product company", "shipped"])
            for job in career:
                if any(w in job.get("description", "").lower() for w in ["production", "deploy", "scale", "shipped"]):
                    has_production_sig = True
                    
            if github_score < 50 and not has_production_sig:
                disqualify_academic = True
                
        if disqualify_academic:
            flags.append("academic_disqualification")
            continue # academic-only with zero production signal is hard excluded
            
        # 2C. LangChain-Only / Shallow AI Filter
        ml_experience_months = 0
        for job in career:
            title = job.get("title", "").lower()
            desc = job.get("description", "").lower()
            if any(w in title or w in desc for w in ["machine learning", "ml", "ai", "deep learning", "nlp", "computer vision", "recommendation"]):
                ml_experience_months += get_job_duration_months(job)
                
        core_ml_skills = {"pytorch", "tensorflow", "jax", "keras", "scikit-learn", "deep learning"}
        shallow_ai_skills = {"langchain", "openai", "chatgpt", "prompt engineering"}
        
        cand_clean_skills = {clean_name(s.get("name", "")) for s in skills}
        has_core_ml = bool(cand_clean_skills & core_ml_skills)
        has_shallow_ai = bool(cand_clean_skills & shallow_ai_skills)
        
        shallow_ai_penalty = 1.0
        if ml_experience_months < 12 and not has_core_ml and has_shallow_ai:
            shallow_ai_penalty = 0.05
            flags.append("shallow_ai_penalty")
            
        # 2D. Non-Coding Tech Leads Filter
        current_title_lower = current_title.lower()
        current_managerial = any(w in current_title_lower for w in ["manager", "lead", "architect", "head", "director"])
        has_recent_coding = False
        months_checked = 0
        for job in career:
            dur = get_job_duration_months(job)
            desc = job.get("description", "").lower()
            if any(w in desc for w in ["code", "develop", "implement", "program", "python", "pytorch", "tensorflow", "engineer", "build"]):
                has_recent_coding = True
                break
            months_checked += dur
            if months_checked >= 18:
                break
                
        non_coding_penalty = 1.0
        if current_managerial and not has_recent_coding:
            non_coding_penalty = 0.10
            flags.append("non_coding_lead_penalty")
            
        # 2E. Title-Chaser Filter
        jobs_checked = 0
        total_months_checked = 0
        for job in career:
            comp_size = job.get("company_size", "10001+")
            desc = job.get("description", "").lower()
            is_startup = comp_size in ["1-10", "11-50"] or any(w in desc for w in ["seed", "series a", "pre-product", "early stage"])
            is_layoff = "layoff" in desc or "laid off" in desc
            
            if not is_startup and not is_layoff:
                jobs_checked += 1
                total_months_checked += get_job_duration_months(job)
                
        avg_eligible_duration = total_months_checked / jobs_checked if jobs_checked > 0 else 24.0
        title_chaser_penalty = 1.0
        if avg_eligible_duration < 12.0 and jobs_checked >= 3:
            title_chaser_penalty = 0.20
            flags.append("title_chaser_penalty")
            
        # 2F. CV/Speech-Only (No NLP/IR) Filter
        cv_speech_skills = {"computervision", "imageclassification", "objectdetection", "speechrecognition", "tts", "stt", "robotics"}
        nlp_ir_skills = {"nlp", "informationretrieval", "search", "rag", "embeddings", "sentencetransformers", "vectorsearch"}
        
        has_cv_speech = bool(cand_clean_skills & cv_speech_skills)
        has_nlp_ir = bool(cand_clean_skills & nlp_ir_skills)
        
        cv_speech_penalty = 1.0
        if has_cv_speech and not has_nlp_ir:
            cv_speech_penalty = 0.05
            flags.append("cv_speech_only_penalty")
            
        # --- Stage 2 Bonus Signals ---
        
        # 7.1 GitHub Topic Signal
        github_bonus = 0.0
        github_score = signals.get("github_activity_score", -1)
        if github_score != -1:
            desc_lower = profile.get("summary", "").lower() + " " + " ".join(j.get("description", "").lower() for j in career)
            if any(w in desc_lower for w in ["rag", "vector search", "embedding", "information retrieval", "nlp", "large language model"]):
                github_bonus += 0.04
            if github_score > 50:
                github_bonus += 0.03
            if any(w in desc_lower for w in ["huggingface", "hugging face", "sentence-transformers", "faiss", "langchain"]):
                github_bonus += 0.05
            if github_score > 30:
                github_bonus += 0.02
            github_bonus = min(github_bonus, 0.10)
            
        # 7.2 Career Trajectory Signal
        trajectory_multiplier = 1.0
        if exp > 0 and len(career) >= 2:
            sorted_jobs = []
            for j in career:
                start_d = parse_date(j.get("start_date"))
                if start_d:
                    sorted_jobs.append((start_d, j))
            sorted_jobs.sort(key=lambda x: x[0])
            
            if len(sorted_jobs) >= 2:
                first_job = sorted_jobs[0][1]
                last_job = sorted_jobs[-1][1]
                
                seniority_first = map_seniority(first_job.get("title", ""))
                seniority_now = map_seniority(last_job.get("title", ""))
                
                trajectory = (seniority_now - seniority_first) / exp
                trajectory_multiplier = max(0.90, min(1.15, 0.90 + 0.05 * trajectory))
                
        # 7.3 Founding-Team Fit Signal
        founding_bonus = 0.0
        has_early_startup = False
        for job in career:
            comp_size = job.get("company_size", "1-10")
            desc = job.get("description", "").lower()
            if comp_size in ["1-10", "11-50"]:
                has_early_startup = True
                break
                
        if has_early_startup:
            founding_bonus += 0.05
            
        desc_full_lower = profile.get("summary", "").lower() + " " + " ".join(j.get("description", "").lower() for j in career)
        if any(w in desc_full_lower for w in ["founding member", "founding engineer", "first engineer", "first developer"]):
            founding_bonus += 0.08
        if any(w in desc_full_lower for w in ["built ml infra from scratch", "built search from scratch", "built recommendation from scratch", "built from scratch"]):
            founding_bonus += 0.03
            
        founding_bonus = min(founding_bonus, 0.10)
        
        # 7.4 JD Coverage Signal
        clusters_matched = sum(1 for c, clean_c_skills in CLEAN_CLUSTERS.items() if any(clean_name(s.get("name", "")) in clean_c_skills for s in skills))
        coverage_multiplier = 0.85 + (0.15 * clusters_matched / 5.0)
        
        # --- Base Score ---
        base_score = (0.35 * s_sem) + (0.30 * s_skill) + (0.15 * s_exp) + (0.12 * s_emp) + (0.08 * s_loc)
        bonuses = github_bonus + founding_bonus
        
        # --- 8.3 Behavioral Modifier (M_beh) ---
        recruiter_response_rate = signals.get("recruiter_response_rate", 0.5)
        
        last_active_str = signals.get("last_active_date")
        last_active_days = 540
        if last_active_str:
            try:
                last_active_date = parse_date(last_active_str)
                if last_active_date:
                    last_active_days = (CURRENT_DATE - last_active_date).days
            except:
                pass
                
        if last_active_days <= 30:
            recency_mult = 1.00
        elif last_active_days <= 90:
            recency_mult = 0.85
        elif last_active_days <= 180:
            recency_mult = 0.70
        elif last_active_days <= 365:
            recency_mult = 0.50
        else:
            recency_mult = 0.20
            
        notice_days = signals.get("notice_period_days")
        if notice_days is None:
            notice_mult = 0.60
        elif notice_days <= 30:
            notice_mult = 1.00
        elif notice_days <= 60:
            notice_mult = 0.70
        elif notice_days <= 90:
            notice_mult = 0.40
        else:
            notice_mult = 0.20
            
        email_verified = signals.get("verified_email", False)
        phone_verified = signals.get("verified_phone", False)
        verification_mult = 1.0 if (email_verified and phone_verified) else 0.90
        
        open_to_work = signals.get("open_to_work_flag", False)
        open_to_work_boost = 1.1 if open_to_work else 1.0
        
        m_beh = recruiter_response_rate * recency_mult * notice_mult * verification_mult * open_to_work_boost
        
        # --- Final Score ---
        final_score = (base_score + bonuses) * m_beh * coverage_multiplier * trajectory_multiplier * consulting_penalty
        
        # Apply other hard exclusions as penalties
        final_score = final_score * shallow_ai_penalty * non_coding_penalty * title_chaser_penalty * cv_speech_penalty
        
        # --- Profile Completeness ---
        has_career_desc = 1.0 if any(j.get("description") for j in career) else 0.0
        has_skills_5 = 1.0 if len(skills) >= 5 else 0.0
        has_github = 1.0 if github_score != -1 else 0.0
        has_verified_contact = 1.0 if (email_verified or phone_verified) else 0.0
        has_career_history = 1.0 if len(career) > 0 else 0.0
        
        completeness = (has_career_desc * 0.30) + (has_skills_5 * 0.20) + (has_github * 0.20) + (has_verified_contact * 0.15) + (has_career_history * 0.15)
        
        name = profile.get("anonymized_name", "Anonymous Candidate")
        last_job = career[-1] if career else {}
        top_company = last_job.get("company", "Product Company")
        last_title = last_job.get("title", "AI Engineer")
        
        # Resolve top 3 critical skills
        matched_skills = []
        for s in skills:
            sname = s.get("name", "")
            if clean_name(sname) in ALL_CRITICAL_CLEAN_SKILLS:
                matched_skills.append((sname, s.get("duration_months", 0)))
        matched_skills.sort(key=lambda x: x[1], reverse=True)
        top_3 = [x[0] for x in matched_skills[:3]]
        if len(top_3) < 3:
            for s in skills:
                sname = s.get("name", "")
                if sname not in top_3:
                    top_3.append(sname)
                if len(top_3) >= 3:
                    break
        top_3_skills_str = ", ".join(top_3) if top_3 else "Python, Machine Learning"
        
        city_name = profile.get("location", "India")
        if willing_relocate:
            location_note = f"Willing to relocate from {city_name}"
        else:
            location_note = f"Based in {city_name}"
            
        github_note = " · Active GitHub contributor" if github_bonus > 0.05 else ""
        notice_days_val = notice_days if notice_days is not None else 60
        
        # Build reasoned string
        gap_notes = []
        if consulting_fraction > 0.0:
            gap_notes.append("includes consulting history")
        if recruiter_response_rate < 0.70:
            gap_notes.append("lower recruiter responsiveness")
        if last_active_days > 60:
            gap_notes.append("recent inactivity")
        if notice_days_val > 60:
            gap_notes.append("longer notice period")
        if completeness < 0.70:
            gap_notes.append("incomplete profile details")
            
        gap_str = ""
        if gap_notes:
            gap_str = f" Note: {', '.join(gap_notes)}."
            
        reasoning = (
            f"{name} brings {int(exp)}y of ML/AI experience, most recently at {top_company} as {last_title}. "
            f"Strong hands-on match on {top_3_skills_str}. {location_note}. "
            f"Available in {notice_days_val} days{github_note}.{gap_str}"
        )
        
        scored_candidates.append({
            "candidate_id": cid,
            "final_score": round(final_score, 4),
            "s_sem": round(s_sem, 4),
            "s_skill": round(s_skill, 4),
            "s_exp": round(s_exp, 4),
            "s_emp": round(s_emp, 4),
            "s_loc": round(s_loc, 4),
            "m_beh": round(m_beh, 4),
            "profile_completeness": round(completeness, 4),
            "flags": "|".join(flags) if flags else "none",
            "reasoning": reasoning
        })
        
    print(f"Scoring complete. Total scored survivors: {len(scored_candidates)}")
    
    # 4. Sort and Rank
    # Primary: final_score descending
    # Tie-break: candidate_id ascending (mandated by validator)
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    results_detailed = []
    results_submission = []
    
    for i, item in enumerate(scored_candidates):
        rank = i + 1
        item_detailed = item.copy()
        item_detailed["rank"] = rank
        results_detailed.append(item_detailed)
        
        results_submission.append({
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "score": item["final_score"],
            "reasoning": item["reasoning"] if rank <= 100 else ""
        })
        
    # Write submission CSV (Exactly 4 columns: candidate_id,rank,score,reasoning)
    print(f"Writing final submission CSV to {output_file}...")
    sub_df = pd.DataFrame(results_submission[:100])
    # Reorder columns to ensure exact order
    sub_df = sub_df[["candidate_id", "rank", "score", "reasoning"]]
    sub_df.to_csv(output_file, index=False)
    
    # Write detailed CSV for recruiter explainability (All 12 columns)
    detailed_file = output_file.replace(".csv", "_detailed.csv")
    print(f"Writing detailed rankings to {detailed_file}...")
    detailed_df = pd.DataFrame(results_detailed)
    # Ensure rank is the first column
    detailed_cols = ["rank", "candidate_id", "final_score", "s_sem", "s_skill", "s_exp", "s_emp", "s_loc", "m_beh", "profile_completeness", "flags", "reasoning"]
    detailed_df = detailed_df[detailed_cols]
    detailed_df.to_csv(detailed_file, index=False)
    print("Ranking finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates against Senior AI Engineer JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV")
    args = parser.parse_args()
    
    run_ranking(args.candidates, args.out)
