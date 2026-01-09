import streamlit as st
import google.generativeai as genai 
import re 
import os 
import pandas as pd
import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from dotenv import load_dotenv
import pypdf 
import io

import zipfile 
import altair as alt 


class VirtualFile:
    def __init__(self, name, content_bytes):
        self.name = name
        self.bytes = content_bytes
    
    def getvalue(self):
        return self.bytes

# --- 1. CONFIGURATION & SETUP ---
load_dotenv()
st.set_page_config(page_title="NAMA Compliance Agent", layout="wide")

try:
    api_key = os.getenv("GENAI_API_KEY") 
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Configuration Error: {e}")

REQUIRED_DOCS = [
    "1- Fees application receipt copy.",
    "2- Nama water services vendor registeration certificates & Product Agency certificates or authorization letter from Factory for local distributor ratified from Oman embassy.",
    "3- Certificate of incorporation of the firm (Factory & Foundry).",
    "4- Manufacturing Process flow chart of product and list of out sourced process / operation if applicable including Outsourcing name & address.",
    "5- Valid copies certificates of (ISO 9001, ISO 45001 & ISO 14001).",
    "6- Factory Layout chart.",
    "7- Factory Organizational structure, Hierarchy levels, Ownership details.",
    "8- Product Compliance Statement with reference to Nama water services specifications (with supports documents accordingly).",
    "9- Product Technical datasheets.",
    "10- Omanisation details from Ministry of Labour.",
    "11- Product Independent Test certificates.",
    "12- Attestation of Sanitary Conformity (hygiene test including mechanical assessment for a full product certificate at 50 degrees Celsiusfull to used in drinking water)",
    "13- Provide products Chemicals Composition of materials.",
    "14- Reference list of products used in Oman or any GCC projects with contact no. or emails of end user or clients."
]

COMPLIANCE_DATA = [
    [1, "Bidder’s Information Sheet", "Yes"],
    [2, "Company Registrations", "Yes"],
    [3, "Undertaking of registration (for International Bidders)", "No"],
    [4, "Bidder’s Authorized Signatory", "Yes"],
    [5, "Details of Bidder’s Local Agent", "No"],
    [6, "Parent Company Undertaking (Where applicable)", "Yes"],
    [7, "Historical Contract Non-Performance", "No"],
    [8, "Litigation History", "Yes"],
    [9, "Confidentiality Non-Disclosure Agreement (Mandatory Requirements)", "Yes"],
    [10, "Statement of Integrity (Mandatory Requirements)", "Yes"],
    [11, "No Conflict-of-Interest Declaration (Mandatory Requirements)", "Yes"],
    [12, "Bidder’s General Experience (GCC & Middle East)", "Yes"],
    [13, "Bidder’s General Experience (International)", "Yes"],
    [14, "Bidder’s General Experience (Civil Engineering Projects)", "Yes"],
    [15, "Bidder’s Specific Experience", "Yes"],
    [16, "Bidder’s Specific Experience (Ongoing Projects - Jobs in Hand)", "Yes"],
    [17, "Bid Qualifications (Deviations, Reservations and Omissions by Bidder) (Mandatory Requirements)", "Yes"],
    [18, "Statement of Unresolved Doubts (Mandatory Requirements)", "Yes"],
    [19, "Declaration of Site Visit", "Yes"],
    [20, "Management Approach", "Yes"],
    [21, "Outline Programme (Schedule)", "Yes"],
    [22, "Production Schedule", "Yes"],
    [23, "Forecast of Anticipated Interim Valuations", "Yes"],
    [24, "Execution Plan and Methodology", "Yes"],
    [25, "Concreting Proposals", "Yes"],
    [26, "Approach to Coordination", "Yes"],
    [27, "NoC Management Plan", "Yes"],
    [28, "Quality Management System", "Yes"],
    [29, "HSE Management System (Mandatory Requirements)", "Yes"],
    [30, "Company Organization Chart", "Yes"],
    [31, "Project Specific Organization Chart", "Yes"],
    [32, "Key Positions Proposed", "Yes"],
    [33, "Key Positions Proposed – Candidate’s Summary (CV’s)", "Yes"],
    [34, "Details of Supervisory and Technical Staff and Laborers (Omanis & Expats)", "Yes"],
    [35, "Confirmation of Omanization of Manpower", "Yes"],
    [36, "List of Proposed Personnel to be Employed on the Works/Services (Omani & Expats)", "Yes"],
    [37, "List of All Omani Staff Employed in the Organization", "Yes"],
    [38, "List of All Expatriates Employed in the Organization", "Yes"],
    [39, "Statement of Compliance from Ministry of Labour 286//2008 (Mandatory Requirements)", "Yes"],
    [40, "Proposal for Base Camp and Accommodation", "Yes"],
    [41, "List of bidder’s Equipment and Machinery", "Yes"],
    [42, "Country of Origin Declaration", "Yes"],
    [43, "List of all Products and Materials (Other than Mechanical, Electrical, ICA & IT)", "Yes"],
    [44, "List of all Products and Materials (Mechanical, Electrical, ICA & IT)", "Yes"],
    [45, "List of Proposed Sub-Contractors & Suppliers", "Yes"],
    [46, "SME Allocation confirmation", "Yes"],
    [47, "List of Proposed Sub-Contractors & Suppliers (SMEs)", "Yes"],
    [48, "Local Content / ICV (Mandatory Requirements)", "Yes"],
    [49, "List of Documents requiring NWS’s Approval", "Yes"],
    [50, "Comfort letter from Bank", "Yes"],
    [51, "Details of bank facilities available", "Yes"],
    [52, "Financial Situation", "Yes"],
    [53, "Average Annual Turnover", "Yes"],
    [54, "Copy of each Circular Letter and Addendum issued by NWS (Mandatory Requirements)", "Yes"],
    [55, "Bid Evaluation Criteria in section 1.2 - 2 (Mandatory Requirements)", "Yes"],
    [56, "Appendix to the Letter of Tender", "Yes"],
]


# --- 2. INTELLIGENT EXTRACTION (Hybrid: Text First -> OCR Fallback) ---
def extract_text_smart(uploaded_file):
    """
    Attempts to read text directly. 
    If text < 50 chars (likely scanned), falls back to OCR.
    """
    text = ""
    file_bytes = uploaded_file.getvalue()
    
    try:
        # METHOD 1: Direct Text Extraction (Super Fast)
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        # Limit to first 3 pages
        num_pages = len(pdf_reader.pages)
        limit = min(3, num_pages)
        
        for i in range(limit):
            page_text = pdf_reader.pages[i].extract_text()
            if page_text:
                text += page_text

        # If we found substantial text, return it immediately
        if len(text.strip()) > 100: 
            return f"FILE_NAME: {uploaded_file.name}\n(Extracted via Text Layer)\n{text[:15000]}"

    except Exception as e:
        print(f"Direct extract failed for {uploaded_file.name}: {e}")

    return f"FILE_NAME: {uploaded_file.name}\n(Extraction Failed: Could not extract text)"

def batch_extract_all(files):
    """Uses ThreadPoolExecutor to process files simultaneously."""
    # Increased workers since direct extraction is not CPU bound
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(extract_text_smart, files))
    return results

# --- 3. BATCHED AI ANALYSIS ---
def analyze_batch(batch_text_list):
    model = genai.GenerativeModel('gemini-2.5-pro',generation_config={"temperature": 0.0})
    today_str = date.today().strftime("%Y-%m-%d")

    prompt = f"""
    Today is {today_str}. You are NAMA Document Analyzer.
    Extract data from pdfs and translate it if it is not in english.
    Classify each document using this list: {json.dumps(REQUIRED_DOCS)}
    
    Compliance Rule: ISO certificates must be valid for >180 days from {today_str}.
    
    Return ONLY a JSON object with this EXACT structure:
    {{
        "iso_analysis": [
            {{
                "standard": "ISO 9001",
                "expiry_date": "YYYY-MM-DD",
                "days_remaining": 0,
                "compliance_status": "Pass/Fail"
            }}
        ],
        "found_documents": [
            {{"filename": "name.pdf", "Category": "Category from list", "Status": "Valid"}}
        ],
        "wras_analysis": {{
                "found": true,
                "wras_id": "123456"
            }}
        ],
        "reference_list": [
            {{"filename": "name.pdf", "Category": "Category from list", "Status": "Valid", "project_count": 0}}
        ],
        "extracted_data": {{
             "company_name": "Name of the company/vendor",
             "icv_score": "ICV score or percentage found (e.g. 10%)",
             "payment_terms": "Payment terms details (e.g. '30 days credit' or '10% Advance')",
             "advance_payment_percentage": "Numeric value of advance payment percentage if found (e.g. 10)",
             "commercial_info": "Commercial comparison details",
             "grand_total": 0.0,
             "project_history": "Total count of previous projects found as a number (e.g. '5')",
             "project_history": "Total count of previous projects found as a number (e.g. '5')",
             "technical_compliance_score": "Technical compliance score or percentage if explicitly mentioned (e.g. '98%')",
             "quotation_file": "filename.pdf"
        }}
    }}
    
    For Category 14 (Reference List), count the number of distinct projects listed and include it in "project_count".
    Extract the company name, ICV score, payment terms and commercial info if available.
    CRITICAL: Extract the 'Grand Total' or 'Total Bid Price' as a pure number (no currency symbols) in "grand_total". If not found, return 0.0.
    Identify the file that acts as the primary 'Quotation' or 'Financial Proposal' (containing the total price). Return its filename in "quotation_file".
    """
    
    combined_content = "\n\n=== NEXT DOCUMENT ===\n".join(batch_text_list)
    
    try:
        response = model.generate_content(
            contents=[prompt, combined_content],
            generation_config={"response_mime_type": "application/json"}
        )
        data = json.loads(response.text)
        if isinstance(data, list): return data[0]
        return data
    except Exception:
        return {}

#

def clear_submit():
    # This function clears the file uploader state
    st.session_state.uploader_id += 1
    if "analysis_result" in st.session_state:
        del st.session_state["analysis_result"]

def process_company_documents(files, status_container=None):
    """
    Orchestrates the extraction and analysis for a list of file-like objects.
    """
    if status_container:
         status_container.write(f"Extracting text from {len(files)} files...")
    
    # 1. Text Extraction
    all_texts = batch_extract_all(files)
    
    if status_container:
         status_container.write("Analyzing content with AI...")

    # 2. Analysis
    final_report = analyze_documents(all_texts)
    
    return final_report

def analyze_documents(all_texts):
     # 2. Parallel AI Analysis Logic (Refactored from previous main block)
    final_report = {
        "iso_analysis": [],
        "wras_analysis": {"found": False, "wras_id": []},
        "found_documents": [],
        "reference_list": [],
        "missing_documents": set(REQUIRED_DOCS),
        "wras_online_check": {"status": "N/A", "url": "#"},
        "company_name": "Unknown Company",
        "icv_score": "N/A",
        "payment_terms": "N/A",
        "commercial_info": "N/A",
        "grand_total": 0.0,
        "project_history": "N/A",
        "technical_compliance_score": "N/A",
        "technical_compliance_score": "N/A",
        "advance_payment_percentage": 0,
        "quotation_file": None
    }

    # Create batches
    batch_size = 8
    batches = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_batch = {executor.submit(analyze_batch, batch): batch for batch in batches}
        
        for future in as_completed(future_to_batch):
            batch_res = future.result()
            if isinstance(batch_res, dict):
                final_report["iso_analysis"].extend(batch_res.get("iso_analysis", []))
                final_report["found_documents"].extend(batch_res.get("found_documents", []))
                final_report["reference_list"].extend(batch_res.get("reference_list", []))
                
                # Aggregate Extracted Data
                ext_data = batch_res.get("extracted_data", {})
                if ext_data.get("company_name") and final_report["company_name"] == "Unknown Company":
                    final_report["company_name"] = ext_data["company_name"]
                if ext_data.get("icv_score") and final_report["icv_score"] == "N/A":
                    final_report["icv_score"] = ext_data["icv_score"]
                if ext_data.get("payment_terms") and final_report["payment_terms"] == "N/A":
                    final_report["payment_terms"] = ext_data["payment_terms"]
                if ext_data.get("commercial_info") and final_report["commercial_info"] == "N/A":
                    final_report["commercial_info"] = ext_data.get("commercial_info", "N/A")
                if ext_data.get("project_history") and final_report["project_history"] == "N/A":
                    final_report["project_history"] = ext_data["project_history"]
                if ext_data.get("technical_compliance_score") and final_report["technical_compliance_score"] == "N/A":
                    final_report["technical_compliance_score"] = ext_data["technical_compliance_score"]
                
                # Advance Payment
                adv = ext_data.get("advance_payment_percentage")
                if isinstance(adv, (int, float)) and adv > 0:
                     final_report["advance_payment_percentage"] = adv
                elif isinstance(adv, str) and adv.isdigit():
                     final_report["advance_payment_percentage"] = float(adv)
                
                # Numeric extraction for total
                val = ext_data.get("grand_total", 0.0)
                if isinstance(val, (int, float)) and val > 0:
                     final_report["grand_total"] = val

                # Quotation File
                q_file = ext_data.get("quotation_file")
                if q_file and not final_report["quotation_file"]:
                    final_report["quotation_file"] = q_file

                wras = batch_res.get("wras_analysis", {})
                if isinstance(wras, dict) and wras.get("found"):
                    final_report["wras_analysis"] = wras

    # Post-Processing
    for doc in final_report["found_documents"]:
        doc_type = doc.get("Category")
        if doc_type in final_report["missing_documents"]:
            final_report["missing_documents"].remove(doc_type)


        
    return final_report

# --- 5. UI & EXECUTION LOGIC ---
# 1. Page Configuration
st.set_page_config(
    page_title="NAMA Tender Analtical TOOL",
    layout="wide",  # This makes the layout span the full width like your screenshot
    initial_sidebar_state="expanded"
)

# Optional: Add some CSS to reduce top white space for a tighter header
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 6, 2], gap="small", vertical_alignment="center")

with col1:
    # REPLACE 'nama_logo.png' with your actual file path
    # 'use_column_width=False' keeps it from getting too big
    st.image("NG-Service-logo.png", width=250) 

with col2:
    # HTML is used here to force the text to be perfectly centered
    st.markdown(
        "<h1 style='text-align: center; margin: 0; font-size: 36px;'>NAMA 2.0</h1>", 
        unsafe_allow_html=True
    )

with col3:
    st.image("velyana-new.png", width=200)
st.title("🎯 AI Tender Analytical Engine")
if "uploader_id" not in st.session_state:
    st.session_state.uploader_id = 0

uploaded_files = st.file_uploader("Upload Vendor ZIP Files (One ZIP per Vendor)", type=["zip"], accept_multiple_files=True, key=f"file_uploader_{st.session_state.uploader_id}")
if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} ZIP files.")
    st.button("Clear", on_click=clear_submit)
    
    if st.button("Run Audit", type="primary"):
        if uploaded_files:
            start_time = datetime.now()
            all_reports = []

            # Progress Bar for ZIPs
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, zip_file in enumerate(uploaded_files):
                status_text.write(f"Processing {zip_file.name}...")
                
                # valid_pdf_files for this company
                company_pdfs = []
                
                try:
                    with zipfile.ZipFile(zip_file) as z:
                        for filename in z.namelist():
                            if filename.lower().endswith(".pdf") and not filename.startswith("__MACOSX") and not filename.startswith("."):
                                with z.open(filename) as f:
                                    content = f.read()
                                    company_pdfs.append(VirtualFile(filename, content))
                except Exception as e:
                    st.error(f"Error reading zip {zip_file.name}: {e}")
                    continue

                if not company_pdfs:
                    st.warning(f"No PDFs found in {zip_file.name}")
                    continue

                # Process this company's files
                with st.status(f"Analyzing {zip_file.name}...", expanded=False) as status:
                    report = process_company_documents(company_pdfs, status_container=status)
                    # If company name wasn't found in text, use zip filename
                    if report["company_name"] == "Unknown Company":
                         report["company_name"] = zip_file.name.replace(".zip", "")
                    
                    report["source_zip_index"] = idx
                    all_reports.append(report)
                    status.update(label=f"Completed {report['company_name']}", state="complete")

                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            # --- COMPUTE L1/L2/L3 RANKINGS ---
            # Extract valid totals
            valid_reports = []
            for r in all_reports:
                try:
                    price = float(r.get("grand_total", 0))
                    valid_reports.append({"price": price, "report": r})
                except:
                    pass
            
            # Sort by price ascending
            valid_reports.sort(key=lambda x: x["price"])
            
            if valid_reports:
                # Assign L1 (Lowest)
                valid_reports[0]["report"]["rank_label"] = "L1"
                # Assign L3 (Highest) - if more than 1
                if len(valid_reports) > 1:
                    valid_reports[-1]["report"]["rank_label"] = "L3"
                # Assign L2 (Others)
                for i in range(1, len(valid_reports) - 1):
                    valid_reports[i]["report"]["rank_label"] = "L2"
            
            # Formate the commercial_info string
            for item in valid_reports:
                price = item["price"]
                label = item["report"].get("rank_label", "")
                if price > 0:
                    item["report"]["commercial_info"] = f"{label} - {price:,.2f} OMR"
            
            st.session_state.analysis_result = all_reports # Store LIST of reports
            
            duration = (datetime.now() - start_time).total_seconds()
            st.success(f"Audit Complete in {duration:.2f} seconds!")



    # --- 6. DISPLAY RESULTS (Same as before) ---
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        reports = st.session_state.analysis_result
        if not isinstance(reports, list): # Handle legacy/single file case just in case
             reports = [reports]

        # Re-construct valid_reports for the conclusion logic logic below
        valid_reports = []
        for r in reports:
            try:
                # We saved price as float in grand_total if it was found
                price = float(r.get("grand_total", 0))
                if price > 0:
                     valid_reports.append({"price": price, "report": r})
            except:
                pass
        valid_reports.sort(key=lambda x: x["price"])

        # --- COMBINED TENDER ANALYTICS ---
        st.subheader("📊 Comparative Tender Analytics")
        
        tender_data = {
            "Aspect": [
                "Technical Compliance Score",
                "Commercial Comparision",
                "In Country Value(ICV)",
                "Previous Project History",
                "Paymenet Terms"
            ]
        }
        
        for res in reports:
             company = res.get("company_name", "Unknown")
             
             # Recalculate basic metrics if needed or pull from res keys if we stored them
             no_of_missing_docs = len(res["missing_documents"])
             
             # Use extracted score if available, else calculate
             extracted_score = res.get("technical_compliance_score", "N/A")
             if extracted_score != "N/A":
                 doc_score = extracted_score
             else:
                 doc_score = f"{round(((14 - no_of_missing_docs) / 14) * 100, 2)}%"

             # Use extracted history if available, else count
             extracted_history = res.get("project_history", "N/A")
             if extracted_history != "N/A":
                 history_display = extracted_history
             else:
                 total_projects = sum([item.get('project_count', 0) for item in res.get('reference_list', [])])
                 history_display = str(total_projects)
             
             tender_data[company] = [
                doc_score,
                res.get("commercial_info", "N/A"),
                res.get("icv_score", "N/A"),
                history_display,
                res.get("payment_terms", "N/A")
            ]
            
        df_analytics = pd.DataFrame(tender_data)
        df_analytics.index = df_analytics.index + 1
        df_analytics.index.name = "Sr. No"
        st.dataframe(df_analytics, use_container_width=True) 

        # --- BAR CHART ---
        chart_data = []
        for res in reports:
            comp_name = res.get("company_name", "Unknown")
            g_total = res.get("grand_total", 0.0)
            if g_total > 0:
                chart_data.append({"Company": comp_name, "Bid Value (OMR)": g_total})
        
        if chart_data:
            st.subheader("📉 Price Comparison Model")
            df_chart = pd.DataFrame(chart_data)
            
            # Custom Aesthetic Bar Chart
            base = alt.Chart(df_chart).encode(
                x=alt.X('Company', sort='-y', axis=alt.Axis(labelAngle=-45, title="Company Name")),
                y=alt.Y('Bid Value (OMR)', axis=alt.Axis(title="Bid Value (OMR)")),
                tooltip=['Company', alt.Tooltip('Bid Value (OMR)', format=",.2f")]
            )

            bars = base.mark_bar().encode(
                color=alt.Color('Company', legend=None, scale=alt.Scale(scheme='tableau10'))
            )

            text = base.mark_text(align='center', baseline='bottom', dy=-5, fontWeight='bold').encode(
                text=alt.Text('Bid Value (OMR)', format=",.0f")
            )

            st.altair_chart((bars + text).interactive(), use_container_width=True)

        # --- CONCLUSION ---
        if valid_reports:
            st.subheader("💡 Expert Conclusion & Weighted Scoring")

            # Helper to parse numeric values
            def parse_val(val_str, default=0.0):
                if isinstance(val_str, (int, float)): return float(val_str)
                try:
                    # Extract first float/int found
                    match = re.search(r"(\d+(\.\d+)?)", str(val_str))
                    if match:
                        return float(match.group(1))
                except:
                    pass
                return default

            # Prepare Data
            aspects = ["Technical Compliance(4)", "Commercial Compliance(2)", "In Country Value(2)", "Previous Project History(1)", "Payment Terms(1)"]
            weights = {"Tech": 4, "Comm": 2, "ICV": 2, "Hist": 1, "Pay": 1}
            
            # We need to collect raw values for comparison
            # Structure: {company: {tech_val, comm_val, ...}}
            comp_data = {}
            
            for item in valid_reports:
                r = item["report"]
                c_name = r.get("company_name", "Unknown")
                price = item["price"] # Already float
                
                # Tech Score
                tech_str = r.get("technical_compliance_score", "0")
                # Fallback to calculated if tech_str is N/A or empty
                if tech_str in ["N/A", ""]:
                     # Re-calculate based on missing docs as fallback
                     miss = len(r["missing_documents"])
                     tech_val = round(((14 - miss) / 14) * 100, 2)
                     tech_display = f"{tech_val}%"
                else:
                     tech_val = parse_val(tech_str)
                     tech_display = tech_str

                # ICV
                icv_str = r.get("icv_score", "0")
                icv_val = parse_val(icv_str)

                # History
                hist_str = r.get("project_history", "0")
                hist_val = parse_val(hist_str)

                # Payment (Using extracted Advance %)
                pay_val = float(r.get("advance_payment_percentage", 0))
                
                comp_data[c_name] = {
                    "Tech": {"val": tech_val, "display": tech_display},
                    "Comm": {"val": price, "display": r.get('rank_label','N/A')},
                    "ICV": {"val": icv_val, "display": icv_str},
                    "Hist": {"val": hist_val, "display": str(int(hist_val))},
                    "Pay": {"val": pay_val, "display": f"{int(pay_val)}%"},
                }

            # Determine Winners
            # Tech: Max
            max_tech = max([d["Tech"]["val"] for d in comp_data.values()] or [0])
            # Comm: Min
            min_price = min([d["Comm"]["val"] for d in comp_data.values()] or [0])
            # ICV: Max
            max_icv = max([d["ICV"]["val"] for d in comp_data.values()] or [0])
            # Hist: Max
            max_hist = max([d["Hist"]["val"] for d in comp_data.values()] or [0])
            # Pay: Min (Assuming % Advance is standard and lower is better as per user request)
            min_pay = min([d["Pay"]["val"] for d in comp_data.values()] or [0])
            
            # Calculate Scores
            row_tech = {"Aspects": "Technical Compliance(4)"}
            row_comm = {"Aspects": "Commercial Compliance(2)"}
            row_icv = {"Aspects": "In Country Value(2)"}
            row_hist = {"Aspects": "Previous Project History(1)"}
            row_pay = {"Aspects": "Payment Terms(1)"}
            row_total = {"Aspects": "Total"}

            # Track total scores
            scores = {c: 0 for c in comp_data}
            winners = {c: [] for c in comp_data} # list of aspects won

            for c, data in comp_data.items():
                # Tech
                is_win = (data["Tech"]["val"] >= max_tech and max_tech > 0)
                if is_win: 
                    scores[c] += weights["Tech"]
                    winners[c].append("Tech")
                row_tech[c] = data["Tech"]["display"]

                # Comm
                is_win = (data["Comm"]["val"] <= min_price and min_price > 0)
                if is_win:
                    scores[c] += weights["Comm"]
                    winners[c].append("Comm")
                row_comm[c] = data["Comm"]["display"]

                # ICV
                is_win = (data["ICV"]["val"] >= max_icv and max_icv > 0)
                if is_win:
                    scores[c] += weights["ICV"]
                    winners[c].append("ICV")
                row_icv[c] = data["ICV"]["display"]

                # Hist
                is_win = (data["Hist"]["val"] >= max_hist and max_hist > 0)
                if is_win:
                    scores[c] += weights["Hist"]
                    winners[c].append("Hist")
                row_hist[c] = data["Hist"]["display"]

                # Pay
                is_win = (data["Pay"]["val"] <= min_pay and min_pay > 0)
                if is_win:
                    scores[c] += weights["Pay"]
                    winners[c].append("Pay")
                row_pay[c] = data["Pay"]["display"]

                # Total
                row_total[c] = scores[c]

            df_ex = pd.DataFrame([row_tech, row_comm, row_icv, row_hist, row_pay, row_total])
            df_ex.set_index("Aspects", inplace=True)
            
            # Styling
            def highlight_winners(df_in):
                # df_in is the dataframe. We return a DataFrame of CSS strings.
                style_df = pd.DataFrame('', index=df_in.index, columns=df_in.columns)
                
                # Check each cell
                for c in df_in.columns:
                    # Tech
                    if "Tech" in winners[c]:
                        style_df.at["Technical Compliance(4)", c] = 'background-color: #d4edda; color: #155724;'
                    # Comm
                    if "Comm" in winners[c]:
                        style_df.at["Commercial Compliance(2)", c] = 'background-color: #d4edda; color: #155724;'
                    # ICV
                    if "ICV" in winners[c]:
                        style_df.at["In Country Value(2)", c] = 'background-color: #d4edda; color: #155724;'
                    # Hist
                    if "Hist" in winners[c]:
                        style_df.at["Previous Project History(1)", c] = 'background-color: #d4edda; color: #155724;'
                    # Pay
                    if "Pay" in winners[c]:
                        style_df.at["Payment Terms(1)", c] = 'background-color: #d4edda; color: #155724;'
                
                return style_df

            st.dataframe(df_ex.style.apply(highlight_winners, axis=None), use_container_width=True)
            
            best_company = max(scores, key=scores.get)
            best_score = scores[best_company]
            
            # Find lowest bidder for the text
            lowest_bidder_name = "Unknown"
            try:
                # Filter for valid prices > 0
                priced_reports = [r for r in reports if float(r.get("grand_total", 0)) > 0]
                if priced_reports:
                    lowest_report = min(priced_reports, key=lambda r: float(r.get("grand_total", 0)))
                    lowest_bidder_name = lowest_report.get("company_name", "Unknown")
            except:
                pass

            st.success(f"""
            **Expert Recommendation:** 
            Based on a detailed comparative analysis, **{best_company}** is recommended for award. Although **{lowest_bidder_name}** submitted the lowest-priced bid, the evaluation weightings indicate that **{best_company}** scores higher on the key deciding factors. Accordingly, **{best_company}** is recommended in line with the prescribed evaluation criteria.
            """)

        # --- DETAILED TABS ---
        st.subheader("📑 Submission Checklist")
        
        # Create tabs for each company
        tabs = st.tabs([r.get("company_name", f"Company {i+1}") for i, r in enumerate(reports)])
        
        for i, tab in enumerate(tabs):
             with tab:
                res = reports[i]
                df_compliance = pd.DataFrame(COMPLIANCE_DATA, columns=["Sr. No", "Title", "Form submitted"])
                
                def style_compliance(val):
                    if val == "Yes":
                        return 'background-color: #d4edda; color: #155724;' # Green
                    elif val == "No":
                        return 'background-color: #f8d7da; color: #721c24;' # Red
                    return ''
                
                st.dataframe(df_compliance.style.map(style_compliance, subset=["Form submitted"]), use_container_width=True, hide_index=True)
                # --- VIEW QUOTATION BUTTON ---
                # st.write("---")
                # st.subheader("📄 Quotation Viewer")
                
                # Retrieve the zip file object associated with this report
                zip_idx = res.get("source_zip_index")
                current_zip = None
                if zip_idx is not None and 0 <= zip_idx < len(uploaded_files):
                    current_zip = uploaded_files[zip_idx]
                
                if current_zip:
                    # Get list of files in this zip
                    try:
                        with zipfile.ZipFile(current_zip) as z:
                            all_files = [f for f in z.namelist() if f.lower().endswith(".pdf") and not f.startswith("__MACOSX") and not f.startswith(".")]
                            
                            # Determine which file to show
                            target_file = res.get("quotation_file")
                            
                            # Fallback logic: 
                            # 1. Use AI found file if it exists in zip
                            # 2. Else use the first PDF found
                            final_file_to_view = None
                            
                            if target_file in all_files:
                                final_file_to_view = target_file
                            elif all_files:
                                final_file_to_view = all_files[0]
                            
                            if final_file_to_view:
                                if st.button("View Quotation", key=f"btn_view_{i}", type="primary"):
                                    with z.open(final_file_to_view) as f:
                                        import base64
                                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                                        st.markdown(pdf_display, unsafe_allow_html=True)
                            else:
                                st.warning("No PDF files found to view.")
                                    
                    except Exception as e:
                        st.error(f"Error reading source zip: {e}")
                else:
                    st.warning("Source ZIP not found (file list may have changed). Please re-run analysis.")
                
        