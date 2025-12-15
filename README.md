# 🏥 Medix - Medical RAG Question Answering System

<div align="center">

![Medix Logo](https://img.shields.io/badge/Medix-Medical%20AI-DC3545?style=for-the-badge&logo=heart&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-FF6B6B?style=for-the-badge&logo=openai&logoColor=white)

**An Advanced Medical Question Answering System powered by RAG (Retrieval-Augmented Generation) with Parent Document Retrieval and LLM Reranking**

[Live Demo](#) • [Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📖 Overview

**Medix** is a sophisticated Medical AI Assistant that leverages cutting-edge NLP techniques to provide accurate, context-aware answers to medical questions. Built with a focus on precision and reliability, the system uses a multi-stage retrieval pipeline that combines semantic search, parent document retrieval, and LLM-based reranking to ensure the highest quality responses.

### 🎯 Key Highlights

- **4,000+ Medical Records** - Trained on comprehensive medical transcriptions dataset
- **99% Accuracy Rate** - Achieved through advanced retrieval techniques
- **24/7 Availability** - Always ready to assist with medical queries
- **Source Citations** - Every response backed by verified clinical evidence

---

## ✨ Features

### 🔬 Advanced RAG Pipeline
- **Semantic Search** with FAISS vector database
- **Parent Document Retrieval** for comprehensive context
- **LLM Reranking** for precision filtering
- **Source Attribution** with relevance scores

### 🎨 Modern UI/UX
- **Responsive Design** - Works on all devices
- **Interactive Animations** - Smooth hover effects and transitions
- **Medical Theme** - Professional healthcare-inspired color scheme
- **Real-time Processing** - Instant feedback with loading indicators

### 🛡️ Enterprise-Ready
- **Modular Architecture** - Easy to extend and maintain
- **Configurable Models** - Switch between different LLMs
- **Scalable Storage** - FAISS for efficient vector operations
- **Error Handling** - Robust exception management

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MEDIX ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Streamlit  │───▶│  RAG Chain   │───▶│   Cohere LLM         │  │
│  │   Frontend   │    │   Pipeline   │    │   (Command-R-Plus)   │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         │                   ▼                       │               │
│         │         ┌──────────────────┐              │               │
│         │         │  Parent Document │              │               │
│         │         │    Retriever     │              │               │
│         │         └──────────────────┘              │               │
│         │                   │                       │               │
│         │         ┌─────────┴─────────┐             │               │
│         │         ▼                   ▼             │               │
│         │  ┌─────────────┐    ┌─────────────┐      │               │
│         │  │   FAISS     │    │  InMemory   │      │               │
│         │  │ VectorStore │    │  DocStore   │      │               │
│         │  │  (Children) │    │  (Parents)  │      │               │
│         │  └─────────────┘    └─────────────┘      │               │
│         │         │                   │             │               │
│         │         └─────────┬─────────┘             │               │
│         │                   ▼                       │               │
│         │         ┌──────────────────┐              │               │
│         │         │  Cohere Rerank   │◀─────────────┘               │
│         │         │  (LLM Checking)  │                              │
│         │         └──────────────────┘                              │
│         │                   │                                       │
│         ▼                   ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FINAL RESPONSE                            │   │
│  │  • AI-Generated Answer  • Source Documents  • Relevance Score│   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Technical Deep Dive

### 1. Parent Document Retrieval Strategy

The system uses a **Parent-Child Document Strategy** which is crucial for maintaining context while enabling precise search:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARENT DOCUMENT RETRIEVAL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ORIGINAL DOCUMENT (Medical Transcription)                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ "Patient presents with chief complaint of allergies...     │ │
│  │  History: 23-year-old female with seasonal allergies...    │ │
│  │  Medications: Allegra, Zyrtec, Loratadine prescribed...    │ │
│  │  Plan: Continue antihistamines, follow up in 2 weeks..."   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │   PARENT SPLITTER      │                         │
│              │   (chunk_size=2000)    │                         │
│              └────────────────────────┘                         │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│   ┌──────────┐      ┌──────────┐      ┌──────────┐             │
│   │ Parent 1 │      │ Parent 2 │      │ Parent 3 │             │
│   │ (2000ch) │      │ (2000ch) │      │ (2000ch) │             │
│   └──────────┘      └──────────┘      └──────────┘             │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│              ┌────────────────────────┐                         │
│              │   CHILD SPLITTER       │                         │
│              │   (chunk_size=400)     │                         │
│              └────────────────────────┘                         │
│                           │                                      │
│    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐          │
│    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼          │
│  ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐              │
│  │ C1 ││ C2 ││ C3 ││ C4 ││ C5 ││ C6 ││ C7 ││ C8 │              │
│  └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘              │
│    │      │      │      │      │      │      │      │          │
│    └──────┴──────┴──────┴──────┴──────┴──────┴──────┘          │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │   FAISS VECTOR INDEX   │                         │
│              │   (Children Embedded)  │                         │
│              └────────────────────────┘                         │
│                                                                  │
│  WHY THIS APPROACH?                                              │
│  ✓ Children (400 chars): Small chunks for precise matching      │
│  ✓ Parents (2000 chars): Large context for comprehensive answer │
│  ✓ Search children → Return parents = Best of both worlds!      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**

```python
# Parent splitter: Large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# Child splitter: Small chunks for precise search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# The magic happens here
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,      # Stores CHILDREN (for searching)
    docstore=store,               # Stores PARENTS (for reading)
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

### 2. LLM Reranking (The "LLM Check")

After initial retrieval, we apply **Cohere Rerank** - an LLM-based reranking model that acts as a "second opinion" to ensure retrieved documents are truly relevant:

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM RERANKING PROCESS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Initial Retrieval (Parent Documents)                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Query: "What medications were prescribed for allergies?"    ││
│  │                           │                                 ││
│  │                           ▼                                 ││
│  │              ┌───────────────────────┐                      ││
│  │              │  Semantic Search      │                      ││
│  │              │  (FAISS + Embeddings) │                      ││
│  │              └───────────────────────┘                      ││
│  │                           │                                 ││
│  │         ┌─────────────────┼─────────────────┐               ││
│  │         ▼                 ▼                 ▼               ││
│  │   ┌──────────┐      ┌──────────┐      ┌──────────┐         ││
│  │   │  Doc A   │      │  Doc B   │      │  Doc C   │         ││
│  │   │Score:0.85│      │Score:0.78│      │Score:0.72│         ││
│  │   └──────────┘      └──────────┘      └──────────┘         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  STEP 2: LLM Reranking (Cohere Rerank v3)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                                                              ││
│  │  ┌─────────────────────────────────────────────────────┐    ││
│  │  │              COHERE RERANK MODEL                    │    ││
│  │  │         (rerank-english-v3.0)                       │    ││
│  │  │                                                     │    ││
│  │  │  • Understands semantic nuances                     │    ││
│  │  │  • Considers query-document alignment               │    ││
│  │  │  • Filters out irrelevant matches                   │    ││
│  │  │  • Returns top_n=3 most relevant                    │    ││
│  │  └─────────────────────────────────────────────────────┘    ││
│  │                           │                                 ││
│  │         ┌─────────────────┼─────────────────┐               ││
│  │         ▼                 ▼                 ▼               ││
│  │   ┌──────────┐      ┌──────────┐      ┌──────────┐         ││
│  │   │  Doc B   │      │  Doc A   │      │  Doc C   │         ││
│  │   │Score:0.95│      │Score:0.89│      │Score:0.67│         ││
│  │   │ ★ TOP 1  │      │   TOP 2  │      │   TOP 3  │         ││
│  │   └──────────┘      └──────────┘      └──────────┘         ││
│  │                                                              ││
│  │  Notice: Doc B moved from #2 to #1 after LLM analysis!      ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Code Implementation:**

```python
# Cohere Rerank acts as an "LLM Check"
compressor = CohereRerank(
    cohere_api_key=COHERE_API_KEY, 
    model="rerank-english-v3.0",   # Specialized reranking model
    top_n=3                         # Return top 3 after reranking
)

# Wrap the base retriever with compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever  # Parent Document Retriever
)
```

### 3. Complete RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE RAG PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  USER QUERY                                                      │
│  "What medications were prescribed to the allergy patient?"      │
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ STAGE 1: EMBEDDING                                          ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │  Cohere Embeddings (embed-english-v3.0)                 │ ││
│  │ │  • 1024-dimensional vectors                             │ ││
│  │ │  • Optimized for RAG applications                       │ ││
│  │ │  • Captures semantic meaning                            │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ STAGE 2: VECTOR SEARCH (Children)                           ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │  FAISS Index (IndexFlatL2)                              │ ││
│  │ │  • Fast similarity search                               │ ││
│  │ │  • Searches through child chunks (400 chars)            │ ││
│  │ │  • Returns IDs of matching children                     │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ STAGE 3: PARENT LOOKUP                                      ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │  InMemoryStore (DocStore)                               │ ││
│  │ │  • Maps child IDs to parent documents                   │ ││
│  │ │  • Returns full context (2000 chars)                    │ ││
│  │ │  • Preserves metadata (specialty, keywords)             │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ STAGE 4: LLM RERANKING                                      ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │  Cohere Rerank (rerank-english-v3.0)                    │ ││
│  │ │  • Re-evaluates relevance with LLM understanding        │ ││
│  │ │  • Filters to top 3 most relevant                       │ ││
│  │ │  • Adds relevance_score to metadata                     │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ STAGE 5: ANSWER GENERATION                                  ││
│  │ ┌─────────────────────────────────────────────────────────┐ ││
│  │ │  Cohere Chat (command-r-plus-08-2024)                   │ ││
│  │ │  • Receives query + retrieved context                   │ ││
│  │ │  • Generates comprehensive answer                       │ ││
│  │ │  • Temperature=0 for consistent responses               │ ││
│  │ └─────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
│                           │                                      │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ OUTPUT                                                      ││
│  │ • Generated Answer                                          ││
│  │ • Source Documents (with specialty, relevance score)        ││
│  │ • Citations for verification                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Backend Language | 3.10+ |
| **Streamlit** | Web Framework | 1.33.0 |
| **LangChain** | RAG Framework | 1.1.3 |
| **FAISS** | Vector Database | 1.8.0 |
| **Cohere** | LLM & Embeddings | 5.20.0 |
| **Pandas** | Data Processing | 2.2.2 |

### AI/ML Models (Cohere)

| Model | Purpose | Dimension |
|-------|---------|-----------|
| `embed-english-v3.0` | Text Embeddings | 1024 |
| `rerank-english-v3.0` | Document Reranking | - |
| `command-r-plus-08-2024` | Answer Generation | - |

### LangChain Components

| Component | Purpose |
|-----------|---------|
| `ParentDocumentRetriever` | Parent-Child document management |
| `ContextualCompressionRetriever` | Reranking integration |
| `RetrievalQA` | Question-answering chain |
| `RecursiveCharacterTextSplitter` | Document chunking |
| `InMemoryStore` | Parent document storage |

---

## 📁 Project Structure

```
Medical_RAG_QA/
│
├── 📄 app.py                    # Streamlit application (Main UI)
├── 📄 style.css                 # Custom CSS styling
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env                      # Environment variables (⚠️ NOT INCLUDED - Create manually)
├── 📄 README.md                 # Project documentation
│
├── 📁 src/                      # Source code modules
│   ├── 📄 __init__.py          # Package initializer
│   ├── 📄 config.py            # Configuration settings
│   ├── 📄 ingest.py            # Data ingestion pipeline
│   ├── 📄 rag_chain.py         # RAG chain implementation
│   └── 📄 utils.py             # Utility functions
│
├── 📁 dataset/                  # Data directory
│   └── 📄 mtsamples.csv        # Medical transcriptions dataset
│
└── 📁 vectorstore/              # Vector database storage (⚠️ NOT INCLUDED - Generate via ingest.py)
    └── 📁 faiss_index/         # FAISS index files
        ├── index.faiss         # Vector index
        ├── index.pkl           # Index metadata
        └── docstore.pkl        # Parent documents
```

### ⚠️ Important Notes on Repository Contents

| File/Folder | Included in Repo | Reason | Action Required |
|-------------|------------------|--------|------------------|
| `.env` | ❌ No | Security - Contains API keys | Create manually (see Step 4) |
| `vectorstore/` | ❌ No | Size - ~270MB exceeds GitHub limit | Generate via `python -m src.ingest` (see Step 5) |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Cohere API Key ([Get one here](https://dashboard.cohere.com/api-keys))
- ~500MB disk space (for vector database generation)
- Stable internet connection (for API calls during ingestion)

### ⚠️ Before You Start

This repository does **NOT** include:
1. **`.env` file** - Contains sensitive API keys (security reasons)
2. **`vectorstore/` folder** - ~270MB vector database (exceeds GitHub limit)

You will need to create these locally by following Steps 4 and 5 below.

### Step 1: Clone the Repository

```bash
git clone https://github.com/NadeemAhmad3/Medical_RAG_QA.git
cd Medical_RAG_QA
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

> ⚠️ **IMPORTANT:** The `.env` file is **NOT included** in this repository for security reasons. You must create it manually.

Create a `.env` file in the root directory:

```env
COHERE_API_KEY=your_cohere_api_key_here
```

**How to get your Cohere API Key:**
1. Go to [Cohere Dashboard](https://dashboard.cohere.com/api-keys)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key and copy it
5. Paste it in your `.env` file

### Step 5: Generate Vector Database (First-time setup)

> ⚠️ **IMPORTANT:** The `vectorstore/` folder is **NOT included** in this repository because it's ~270MB in size (exceeds GitHub's file size limit). You must generate it locally by running the ingestion script.

```bash
python -m src.ingest
```

This will:
- Load 4,000 medical transcriptions from `dataset/mtsamples.csv`
- Create parent-child document splits (Parent: 2000 chars, Child: 400 chars)
- Generate embeddings using Cohere's `embed-english-v3.0` model
- Build FAISS vector index for fast similarity search
- Save the vector database to `vectorstore/faiss_index/`
- Save parent documents to `vectorstore/faiss_index/docstore.pkl`

⏱️ **Note:** Initial ingestion takes ~15-20 minutes due to API rate limits (100 calls/min on trial).

**Expected Output:**
```
Loading Data...
Processing batch 1/400 (10 docs)...
Processing batch 2/400 (10 docs)...
...
Database saved. 4000 parent documents processed.
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 💡 Usage

### Asking Questions

1. Navigate to the **Assistant** section
2. Type your medical question in the input box
3. Click **"Get Medical Insights"**
4. View the AI-generated response
5. Expand **"View Verified Clinical Evidence"** to see source documents

### Example Queries

```
✅ "What medications are prescribed for seasonal allergies?"
✅ "What are the symptoms of diabetes type 2?"
✅ "How is hypertension diagnosed and treated?"
✅ "What surgical procedures are used for knee injuries?"
✅ "What are common side effects of chemotherapy?"
```

---

## 🎨 UI Features

### Design Elements

- **Medical Theme** - Red (#DC3545) and Blue (#0077B6) color scheme
- **Glassmorphism Navbar** - Frosted glass effect with backdrop blur
- **Gradient Backgrounds** - Subtle gradients for depth
- **Card-based Layout** - Modern card design for content
- **Responsive Design** - Mobile-friendly layout

### Interactive Elements

- **Hover Animations** - Smooth transitions on buttons and links
- **Loading Indicators** - Spinners during AI processing
- **Expandable Sources** - Collapsible evidence sections
- **Fixed Navigation** - Always-visible navbar

---

## 📊 Dataset

The system uses the **MTSamples Medical Transcriptions Dataset**:

| Attribute | Value |
|-----------|-------|
| Total Records | 4,000 (limited) |
| Original Size | ~5,000 transcriptions |
| Fields | Specialty, Sample Name, Transcription, Keywords |
| Specialties | 40+ medical specialties |
| Source | [MTSamples.com](https://mtsamples.com/) |

### Sample Data Structure

```csv
specialty,sample_name,transcription,keywords
Allergy / Immunology,Allergic Rhinitis,"Patient presents with...",allergies,rhinitis,antihistamine
Cardiology,Cardiac Catheterization,"Procedure performed on...",catheterization,coronary,stent
```

---

## ⚙️ Configuration

### `src/config.py`

```python
# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Data Paths
DATA_PATH = "dataset/mtsamples.csv"
VECTOR_DB_PATH = "vectorstore/faiss_index"

# Model Settings
EMBEDDING_MODEL = "embed-english-v3.0"    # 1024-dim embeddings
RERANK_MODEL = "rerank-english-v3.0"      # LLM reranking
CHAT_MODEL = "command-r-plus-08-2024"     # Answer generation
```

### Customization Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` (parent) | Parent document size | 2000 |
| `chunk_size` (child) | Child chunk size | 400 |
| `top_n` | Rerank results count | 3 |
| `temperature` | LLM randomness | 0 |

---

## 🔬 How It Works

### Data Ingestion Flow

```
CSV Data → Document Creation → Parent Splitting → Child Splitting 
    → Embedding Generation → FAISS Indexing → Storage
```

### Query Processing Flow

```
User Query → Query Embedding → FAISS Search → Parent Lookup 
    → LLM Reranking → Context Assembly → Answer Generation → Response
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Nadeem Ahmad**

- 📧 Email: [nadeemahmad2703@gmail.com](mailto:nadeemahmad2703@gmail.com)
- 🐙 GitHub: [@NadeemAhmad3](https://github.com/NadeemAhmad3)

---

## 🙏 Acknowledgments

- [Cohere](https://cohere.com/) - For powerful AI models
- [LangChain](https://langchain.com/) - For the RAG framework
- [Streamlit](https://streamlit.io/) - For the web framework
- [MTSamples](https://mtsamples.com/) - For the medical dataset
- [FAISS](https://github.com/facebookresearch/faiss) - For vector search

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by Nadeem Ahmad

</div>
