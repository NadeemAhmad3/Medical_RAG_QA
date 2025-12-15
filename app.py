import streamlit as st
from src.rag_chain import get_qa_chain

# Page Config
st.set_page_config(
    page_title="Medix - Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

# Navigation Bar
st.markdown('''<style>
.nav-link{text-decoration:none;color:#1A1D23;font-weight:500;font-size:0.9rem;padding:0.5rem 1rem;border-radius:8px;transition:all 0.3s ease;position:relative;}
.nav-link:hover{color:#DC3545;background:rgba(220,53,69,0.08);}
.nav-link::after{content:'';position:absolute;bottom:0;left:50%;width:0;height:2px;background:#DC3545;transition:all 0.3s ease;transform:translateX(-50%);}
.nav-link:hover::after{width:60%;}
.nav-btn{background:linear-gradient(135deg,#DC3545,#B02A37);color:white;padding:0.5rem 1.25rem;border-radius:50px;font-size:0.85rem;font-weight:600;cursor:pointer;transition:all 0.3s ease;box-shadow:0 4px 15px rgba(220,53,69,0.3);}
.nav-btn:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(220,53,69,0.4);}
</style>
<nav style="position:fixed;top:0;left:0;right:0;z-index:9999;background:rgba(255,255,255,0.95);backdrop-filter:blur(10px);padding:0.75rem 3rem;display:flex;justify-content:space-between;align-items:center;box-shadow:0 2px 20px rgba(0,0,0,0.08);border-bottom:1px solid #eee;">
<div style="display:flex;align-items:center;gap:0.5rem;">
<div style="background:linear-gradient(135deg,#DC3545,#B02A37);padding:0.5rem;border-radius:10px;display:flex;align-items:center;justify-content:center;">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 5V19M5 12H19" stroke="white" stroke-width="2.5" stroke-linecap="round"/></svg>
</div>
<span style="font-size:1.5rem;font-weight:700;color:#1A1D23;">Medix</span>
</div>
<div style="display:flex;gap:0.5rem;align-items:center;">
<a href="#home" class="nav-link">Home</a>
<a href="#features" class="nav-link">Features</a>
<a href="#chat" class="nav-link">Assistant</a>
<a href="#about" class="nav-link">About</a>
<div class="nav-btn" style="margin-left:1rem;">Get Started</div>
</div>
</nav>
<div style="height:70px;"></div>''', unsafe_allow_html=True)

# Hero Section
st.markdown('''<div id="home" style="min-height:90vh;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;background:linear-gradient(180deg,#f8f9fa 0%,#e9ecef 50%,#f8f9fa 100%);padding:3rem 2rem;position:relative;overflow:hidden;">
<div style="position:absolute;top:10%;left:5%;width:300px;height:300px;background:radial-gradient(circle,rgba(220,53,69,0.1) 0%,transparent 70%);border-radius:50%;"></div>
<div style="position:absolute;bottom:15%;right:10%;width:400px;height:400px;background:radial-gradient(circle,rgba(0,119,182,0.08) 0%,transparent 70%);border-radius:50%;"></div>
<div style="position:absolute;top:15%;left:10%;font-size:2.5rem;opacity:0.2;">🩺</div>
<div style="position:absolute;top:25%;right:15%;font-size:2rem;opacity:0.2;">💊</div>
<div style="position:absolute;bottom:25%;left:15%;font-size:2.2rem;opacity:0.2;">🧬</div>
<div style="position:absolute;bottom:20%;right:20%;font-size:1.8rem;opacity:0.2;">❤️</div>
<div style="position:absolute;top:40%;left:8%;font-size:1.5rem;opacity:0.15;">🔬</div>
<div style="position:relative;z-index:10;max-width:900px;">
<div style="margin-bottom:1.5rem;">
<span style="background:linear-gradient(135deg,#DC3545,#B02A37);color:#fff;padding:0.5rem 1.5rem;border-radius:50px;font-size:0.85rem;font-weight:600;letter-spacing:1px;display:inline-block;">🚀 NEXT-GEN MEDICAL AI</span>
</div>
<h1 style="font-size:3.5rem;font-weight:800;color:#1A1D23;line-height:1.2;margin-bottom:1.5rem;">Your Intelligent<br><span style="background:linear-gradient(135deg,#DC3545,#0077B6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Medical Assistant</span></h1>
<p style="font-size:1.2rem;color:#4A5568;line-height:1.8;margin-bottom:2.5rem;max-width:700px;margin-left:auto;margin-right:auto;">Experience the future of healthcare with AI-powered clinical insights. Get instant, accurate medical information backed by verified clinical records and advanced RAG technology.</p>
<div style="display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;margin-bottom:3rem;">
<a href="#chat" style="text-decoration:none;background:linear-gradient(135deg,#DC3545,#B02A37);color:#fff;padding:1rem 2.5rem;border-radius:50px;font-size:1.1rem;font-weight:600;box-shadow:0 8px 25px rgba(220,53,69,0.35);display:inline-flex;align-items:center;gap:0.5rem;">Start Consultation →</a>
<a href="#features" style="text-decoration:none;background:#fff;color:#1A1D23;padding:1rem 2.5rem;border-radius:50px;font-size:1.1rem;font-weight:600;border:2px solid #DEE2E6;display:inline-flex;align-items:center;gap:0.5rem;">Learn More</a>
</div>
<div style="display:flex;gap:3rem;justify-content:center;flex-wrap:wrap;">
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#DC3545;">4000+</div><div style="font-size:0.9rem;color:#6C757D;">Medical Records</div></div>
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#0077B6;">99%</div><div style="font-size:0.9rem;color:#6C757D;">Accuracy Rate</div></div>
<div style="text-align:center;"><div style="font-size:2.5rem;font-weight:700;color:#17A2B8;">24/7</div><div style="font-size:0.9rem;color:#6C757D;">Available</div></div>
</div>
</div>
</div>''', unsafe_allow_html=True)

# Chat Section - RIGHT AFTER HERO
st.markdown('''<div id="chat" style="padding:4rem 2rem;background:linear-gradient(180deg,#fff 0%,#f8f9fa 100%);">
<div style="max-width:800px;margin:0 auto;text-align:center;">
<p style="font-size:0.9rem;color:#DC3545;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">💬 ASK MEDIX</p>
<h2 style="font-size:2.2rem;font-weight:700;color:#1A1D23;margin-bottom:0.75rem;">Medical AI Assistant</h2>
<p style="font-size:1rem;color:#6C757D;margin-bottom:2rem;">Ask any clinical question and get AI-powered insights backed by verified medical records</p>
</div>
</div>''', unsafe_allow_html=True)

# Initialize Chain (silently)
if "chain" not in st.session_state:
    with st.spinner("🔄 Initializing Medical Knowledge Base..."):
        st.session_state.chain = get_qa_chain()

# Chat Interface with text_input (not fixed at bottom)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create centered container for chat
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    # Text input for query (appears in place, not fixed)
    query = st.text_input("", placeholder="💬 Ask a clinical question... (e.g., 'What are symptoms of diabetes?')", key="query_input", label_visibility="collapsed")
    
    # Submit button
    submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
    with submit_col2:
        submit = st.button("🔬 Get Medical Insights", use_container_width=True, type="primary")

    # Display previous messages with styled cards
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);padding:1rem 1.5rem;border-radius:15px;margin:1rem 0;border-left:4px solid #DC3545;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
<span style="font-size:1.2rem;">🧑‍⚕️</span>
<span style="font-weight:600;color:#DC3545;font-size:0.85rem;">Your Question</span>
</div>
<p style="color:#1A1D23;margin:0;font-size:1rem;">{message["content"]}</p>
</div>''', unsafe_allow_html=True)
        else:
            st.markdown(f'''<div style="background:linear-gradient(135deg,#fff,#f8f9fa);padding:1.5rem;border-radius:15px;margin:1rem 0;border:1px solid #e9ecef;box-shadow:0 4px 15px rgba(0,0,0,0.08);">
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
<div style="background:linear-gradient(135deg,#DC3545,#B02A37);padding:0.4rem;border-radius:8px;">
<span style="font-size:1rem;">🏥</span>
</div>
<span style="font-weight:600;color:#1A1D23;font-size:0.9rem;">Medix AI Response</span>
</div>
<div style="color:#495057;line-height:1.7;font-size:0.95rem;">{message["content"]}</div>
</div>''', unsafe_allow_html=True)
            
            # Show sources if available
            if "sources" in message:
                with st.expander("📋 View Verified Clinical Evidence", expanded=False):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(f'''<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);padding:1rem;border-radius:12px;margin:0.5rem 0;border-left:4px solid #DC3545;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
<span style="font-weight:600;color:#212529;">📄 Source {idx+1}: {src["name"]}</span>
<span style="background:#0077B6;color:white;padding:0.2rem 0.6rem;border-radius:50px;font-size:0.7rem;">Relevance: {src["score"]}</span>
</div>
<div style="font-size:0.8rem;color:#6c757d;margin-bottom:0.5rem;">🏷️ Specialty: <strong>{src["specialty"]}</strong></div>
<div style="font-size:0.85rem;color:#495057;font-style:italic;background:white;padding:0.75rem;border-radius:8px;">"{src["content"]}"</div>
</div>''', unsafe_allow_html=True)
    
    # Process new query
    if submit and query:
        # Show user question immediately with styled card
        st.markdown(f'''<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);padding:1rem 1.5rem;border-radius:15px;margin:1rem 0;border-left:4px solid #DC3545;box-shadow:0 2px 10px rgba(0,0,0,0.05);">
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
<span style="font-size:1.2rem;">🧑‍⚕️</span>
<span style="font-weight:600;color:#DC3545;font-size:0.85rem;">Your Question</span>
</div>
<p style="color:#1A1D23;margin:0;font-size:1rem;">{query}</p>
</div>''', unsafe_allow_html=True)
        
        with st.spinner("🔬 Analyzing clinical records & retrieving evidence..."):
            response = st.session_state.chain.invoke({"query": query})
            answer = response["result"]
            sources = response["source_documents"]
            
            # Display styled response card
            st.markdown(f'''<div style="background:linear-gradient(135deg,#fff,#f8f9fa);padding:1.5rem;border-radius:15px;margin:1rem 0;border:1px solid #e9ecef;box-shadow:0 4px 15px rgba(0,0,0,0.08);">
<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">
<div style="background:linear-gradient(135deg,#DC3545,#B02A37);padding:0.4rem;border-radius:8px;">
<span style="font-size:1rem;">🏥</span>
</div>
<span style="font-weight:600;color:#1A1D23;font-size:0.9rem;">Medix AI Response</span>
</div>
<div style="color:#495057;line-height:1.7;font-size:0.95rem;">{answer}</div>
</div>''', unsafe_allow_html=True)
            
            # Display sources
            with st.expander("📋 View Verified Clinical Evidence", expanded=True):
                for idx, doc in enumerate(sources):
                    score = doc.metadata.get('relevance_score', 'N/A')
                    specialty = doc.metadata.get('specialty', 'General')
                    sample_name = doc.metadata.get('sample_name', 'Medical Record')
                    
                    st.markdown(f'''<div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);padding:1rem;border-radius:12px;margin:0.5rem 0;border-left:4px solid #DC3545;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
<span style="font-weight:600;color:#212529;">📄 Source {idx+1}: {sample_name}</span>
<span style="background:#0077B6;color:white;padding:0.2rem 0.6rem;border-radius:50px;font-size:0.7rem;">Relevance: {score}</span>
</div>
<div style="font-size:0.8rem;color:#6c757d;margin-bottom:0.5rem;">🏷️ Specialty: <strong>{specialty}</strong></div>
<div style="font-size:0.85rem;color:#495057;font-style:italic;background:white;padding:0.75rem;border-radius:8px;">"{doc.page_content[:300]}..."</div>
</div>''', unsafe_allow_html=True)
            
            # Store in session state with sources
            source_data = [{"name": doc.metadata.get('sample_name', 'Medical Record'), 
                           "score": doc.metadata.get('relevance_score', 'N/A'),
                           "specialty": doc.metadata.get('specialty', 'General'),
                           "content": doc.page_content[:300] + "..."} for doc in sources]
            
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_data})

# Features Section - NOW AFTER CHAT
st.markdown('''<div id="features" style="padding:5rem 2rem;background:#fff;">
<div style="text-align:center;max-width:1100px;margin:0 auto;">
<p style="font-size:1rem;color:#DC3545;font-weight:600;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem;">POWERED BY ADVANCED AI</p>
<h2 style="font-size:2.5rem;font-weight:700;color:#1A1D23;margin-bottom:1rem;">Intelligent Medical Insights</h2>
<p style="font-size:1.1rem;color:#4A5568;margin-bottom:3rem;">Powered by <strong style="color:#DC3545;">RAG</strong> &amp; <strong style="color:#0077B6;">Cohere AI</strong></p>
<div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap;">
<div style="background:linear-gradient(135deg,#fff5f5,#fee2e2);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #fecaca;box-shadow:0 4px 15px rgba(220,53,69,0.1);">
<div style="background:linear-gradient(135deg,#DC3545,#B02A37);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(220,53,69,0.3);"><span style="font-size:1.5rem;">🏥</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1A1D23;margin-bottom:0.5rem;">Clinical Records</h3>
<p style="font-size:0.9rem;color:#6C757D;">Access thousands of verified medical transcriptions and clinical documentation.</p>
</div>
<div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #bfdbfe;box-shadow:0 4px 15px rgba(0,119,182,0.1);">
<div style="background:linear-gradient(135deg,#0077B6,#005A8C);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(0,119,182,0.3);"><span style="font-size:1.5rem;">🔍</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1A1D23;margin-bottom:0.5rem;">Smart Search</h3>
<p style="font-size:0.9rem;color:#6C757D;">AI-powered semantic search finds the most relevant medical information instantly.</p>
</div>
<div style="background:linear-gradient(135deg,#ecfeff,#cffafe);padding:2rem;border-radius:20px;width:300px;text-align:center;border:1px solid #a5f3fc;box-shadow:0 4px 15px rgba(23,162,184,0.1);">
<div style="background:linear-gradient(135deg,#17A2B8,#138496);width:60px;height:60px;border-radius:15px;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;box-shadow:0 4px 15px rgba(23,162,184,0.3);"><span style="font-size:1.5rem;">✅</span></div>
<h3 style="font-size:1.25rem;font-weight:600;color:#1A1D23;margin-bottom:0.5rem;">Verified Sources</h3>
<p style="font-size:0.9rem;color:#6C757D;">Every response is backed by verified clinical evidence with source citations.</p>
</div>
</div>
</div>
</div>''', unsafe_allow_html=True)

# Footer
st.markdown('''<div id="about" style="background:linear-gradient(135deg,#DC3545 0%,#B02A37 100%);padding:2rem 2rem;margin-top:3rem;">
<div style="max-width:1000px;margin:0 auto;text-align:center;">
<div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;margin-bottom:1rem;">
<div style="background:rgba(255,255,255,0.2);padding:0.4rem;border-radius:8px;display:flex;align-items:center;justify-content:center;">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M12 5V19M5 12H19" stroke="white" stroke-width="2.5" stroke-linecap="round"/></svg>
</div>
<span style="font-size:1.25rem;font-weight:700;color:white;">Medix</span>
</div>
<p style="color:rgba(255,255,255,0.9);font-size:0.95rem;margin-bottom:1rem;">AI-Powered Clinical Intelligence for Modern Healthcare</p>
<div style="margin-bottom:1rem;">
<a href="mailto:nadeemahmad2703@gmail.com" style="color:white;text-decoration:none;font-size:0.9rem;display:inline-flex;align-items:center;gap:0.5rem;background:rgba(255,255,255,0.15);padding:0.5rem 1.25rem;border-radius:50px;">📧 nadeemahmad2703@gmail.com</a>
</div>
<div style="border-top:1px solid rgba(255,255,255,0.2);padding-top:1rem;">
<p style="color:rgba(255,255,255,0.8);font-size:0.75rem;margin:0;">© 2025 Medix. Empowering Healthcare with Artificial Intelligence</p>
</div>
</div>
</div>''', unsafe_allow_html=True)
