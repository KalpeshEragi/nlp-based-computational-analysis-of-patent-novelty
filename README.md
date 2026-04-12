<div align="center">

# ⚖️ Computational Analysis of Patent Novelty Using NLP

**A computational microservice for automated patentability assessment using NLP and semantic embeddings.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-MiniLM-orange.svg)]()
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg?logo=spacy&logoColor=white)]()
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4.svg?logo=google&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

</div>

---

## 🛑 Problem Statement
The patent review process is fundamentally bottlenecked. Determining if a new patent claim is truly "novel" (the inventive step) requires manual, highly subjective cross-referencing against massive troves of prior art. It takes thousands of hours of expert legal analysis to find the exact overlapping clauses and functions.

## 💡 Solution Overview
This project presents an **AI-powered computational inference engine** designed to solve this at scale. It acts as a modular microservice that ingests patent claims and prior art data, tokenizes them using NLP, computes dense semantic embeddings, and runs a strict, mathematical rule-based engine to quantitatively prove or disprove the novelty of a patent claim. 

## ✨ Key Features
- **Semantic Claim Extraction**: Deconstructs dense legal text into distinct `components` (nouns) and `functions` (verbs).
- **Dense Vector Embeddings**: Uses high-dimensional vector representations to find the most semantically related prior art.
- **Rule-Based Inference Engine**: An algorithm that computes overlap ratios and relation similarities to output deterministic verdicts (`INVENTIVE`, `WEAK INVENTIVE`, `NOT INVENTIVE`).
- **Graph Visualization**: Dynamically builds bipartite Knowledge Graphs to visually prove the relations between components and functions.
- **LLM Summarization**: Feeds mathematical outputs into Google Gemini to generate human-readable, legal-style summaries.
- **Interactive Dashboard**: A beautiful, real-time Streamlit UI providing a bird's-eye view of all pipeline metrics.

---

## 🛠️ Tech Stack

| Domain | Technology | Use Case |
| :--- | :--- | :--- |
| **Language** | `Python` | Core backend logic & scripting |
| **NLP** | `spaCy` | Tokenization, POS tagging, and entity extraction |
| **Embeddings** | `SentenceTransformers` | `MiniLM-L6-v2` for dense vector creation |
| **Similarity** | `scikit-learn` & `numpy` | Cosine similarity computations |
| **LLM Inference** | `Google Gemini API` | Narrative legal summary generation |
| **Graphing** | `NetworkX` & `Matplotlib` | Bipartite claim visualization |
| **Frontend UI** | `Streamlit` | Interactive data dashboard |

---

## ⚙️ Architecture & Data Workflow

This microservice is built with strict separation of concerns, executing through a 6-step pipeline:

1. **Input Generation:** Ingests mock JSON dicts representing upstream Prior Art search results.
2. **Parsing (`claim_parser.py`):** spaCy extracts core structural representations of the claim.
3. **Embedding (`similarity.py`):** Sentences are vectorized and ranked via Cosine Similarity (Top-K).
4. **Comparison (`comparator.py`):** Calculates precise subset matching for components and functions against the highest-ranked prior art.
5. **Decision (`inventive_engine.py`):** The logic gateway rules on the novelty based on the mathematical thresholds.
6. **Visualization & UI (`graph.py`/`app.py`):** Outputs are pushed to NetworkX graphs or rendered into the Streamlit dashboard.

---

## 📂 Project Structure

```text
├── data/                    # Upstream mocking data (Prior Art datasets)
├── outputs/                 # Directory for generated reports and network graphs
├── .env.example             # Template for API keys
├── .gitignore               # Ignored system files & cache
├── requirements.txt         # Project dependencies
├── app.py                   # Streamlit Frontend Dashboard
├── claim_parser.py          # NLP tokenization module
├── comparator.py            # Component overlap module
├── graph.py                 # NetworkX visualization module
├── inventive_engine.py      # Core logic/inference algorithm
├── main.py                  # CLI Orchestration pipeline
├── similarity.py            # Vector embedding and semantic search scoring
└── summary.py               # Google Gemini LLM wrapper
```

---

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/computational-analysis-of-patent-novelty.git
cd computational-analysis-of-patent-novelty
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the required spaCy NLP model**
```bash
python -m spacy download en_core_web_sm
```

**5. Configure Environment Variables**
Copy the example environment file and insert your API credentials for the LLM summaries.
```bash
cp .env.example .env
```
*(Open `.env` and add your `GEMINI_API_KEY`)*

---

## 💻 Usage

### 1. Run the interactive Dashboard (Recommended)
Launch the Streamlit frontend to explore the data dynamically:
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser.*

### 2. Run the CLI Pipeline 
If you simply want to execute the analytics pipeline and generate outputs programmatically:
```bash
python main.py
```

---

## 📊 Example Output
When processing a patent, the engine returns quantitative matching:
> **Claim:** "A navigation system comprising a neural network module for real-time trajectory calculation." <br>
> **Verdict:** `WEAK INVENTIVE` <br>
> **Reason:** Strong semantic similarity (0.89) to Patent-B. Uses the same core components (navigation system, sensor) but introduces novel unmapped functions (neural network trajectory calculation).

It will also output **Knowledge Graphs** plotting the literal relations.

---

## 🔮 Future Improvements
As an open-source project, there is massive room for scale. Planned upgrades include:
* **Database Integration:** Replacing `/data` stubs with a vector database (like Pinecone or ChromaDB) hosting real USPTO patent records.
* **Finer Thresholds:** Utilizing fine-tuned BERT models specializing in patent legalese rather than standard Mini-LM embeddings.
* **Component Weighting:** Assigning TF-IDF or probabilistic weights to certain physical components based on field of invention.

---

## 🤝 Contributing
Contributions are always welcome. To contribute:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
  <b>Built for AI/ML Architects and NLP Engineers.</b>
</div>
