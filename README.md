# DEMO ADK Agent: Using Vanna (SQL) + NetworkX (Graph)

This project implements a sophisticated "Singleton Agent" using the **Google Agent Development Kit (ADK)**. It provides a unified interface to answer complex business questions by intelligently routing queries to one of two powerful, specialized backends:

1.  **Vanna.ai (SQL Analytics):** Leverages a local Qdrant vector store and Google's Gemini model to translate natural language into BigQuery SQL. Ideal for quantitative analysis (e.g., "What was our total revenue?").
2.  **NetworkX (Graph Analytics):** Uses a pre-built `networkx` graph to explore relationships between corporate entities. Ideal for qualitative analysis, fraud detection, and connections (e.g., "How is User A connected to User B?").

The `RootAgent` analyzes user intent, selects the single best tool, and transforms the tool's raw JSON output into a clean, human-readable answer.

---

## 🚀 Core Features

* **Intelligent Tool Routing:** A single `RootAgent` orchestrator that dynamically switches between SQL and Graph tools based on natural language intent.
* **Natural Language to SQL:** Generates, executes, and summarizes BigQuery SQL results (powered by Vanna).
* **1st-Degree Graph Traversal:** Explores direct connections of any node (e.g., "What did USER 123 buy?").
* **Degree-of-Separation Analysis:** Finds the shortest path between two nodes (e.g., "Show connection between USER_123 and IP_4.5.6.7").
* **Collaborative Filtering:** Provides "Users who bought X also bought..." recommendations.
* **Interactive Graph Visualization:** Generates interactive Plotly HTML files on demand (e.g., "Visualize USER 8088").

---

## 🏗️ Architecture

The project runs in two phases: an offline **ETL/Training** phase and an online **Runtime** phase.

### 1. Offline ETL & Training (One-Time Setup)
Before running the agent, specialized knowledge bases are built from BigQuery:
* **`bigquery_to_graph.py`:** Fetches entity relationships from BigQuery and builds a NetworkX DiGraph, saving it to `data/graph.gpickle`.
* **`vanna_trainer.py`:** Reads a schema file, generates DDL statements, and trains a local Vanna instance, storing embeddings in `data/qdrant_db`.

### 2. Online Agent Runtime
The `RootAgent` (`agent.py`) receives user prompts and follows a strict instruction set:
1.  **Analyze Intent:** Determine if the question is SQL-based or Graph-based.
2.  **Call Tool:** Execute the corresponding function from `my_tools.py`.
3.  **Parse JSON:** The tool returns a standardized JSON string.
4.  **Format Response:** The agent translates the JSON into a final human-readable text or table.

---

## 🛠️ Setup & Installation

### 1. Clone & Environment
```bash
git clone [your-repo-url]
cd [your-repo-name]

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
### 2. Google Cloud Authentication (And also, GCP Project Name are Hard Coded)
```
gcloud auth application-default login
```
### 3. Create .env File Create a .env file in the project root
```
GEMINI_API_KEY="your_google_gemini_api_key_here"
GEMINI_MODEL="gemini-2.5-flash"
```
### 4. Data Prep
Place your schema file at: `data/ecomm-schema.csv`
If your file has a different name, update the SCHEMA_FILE_PATH variable in vanna_trainer.py.

### Build the Graph Knowledge Base
`python src/bigquery_to_graph.py`

### Train the Vanna SQL Model
`python src/vanna_trainer.py`

### Run the Agent from the Root Folder
`adk web .`
