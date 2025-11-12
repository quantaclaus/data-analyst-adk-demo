import os
import networkx as nx
import pickle
import json
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

# --- NEW IMPORTS FOR VISUALIZATION ---
import plotly.graph_objects as go
# --- END NEW IMPORTS ---

# --- Vanna Imports ---
from vanna.qdrant import Qdrant_VectorStore
from vanna.google import GoogleGeminiChat
# --- End Vanna Imports ---

# --- LOGGER IMPORT ---
from .custom_logger import LoganLogger
# --- END LOGGER IMPORT ---

# Load .env file from the project root
load_dotenv()

# --- Project Path Header ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
# --- End Path Header ---

# --- LOGGER SETUP ---
logger = LoganLogger('my_tools', LOG_DIR, 'etl.log')
# --- END LOGGER SETUP ---


# --- VANNA CONFIG (LOCAL) ---
QDRANT_PATH = os.path.join(DATA_DIR, 'qdrant_db')

class MyVanna(Qdrant_VectorStore, GoogleGeminiChat):
    def __init__(self, config=None):
        client = QdrantClient(path=QDRANT_PATH) 
        Qdrant_VectorStore.__init__(self, config={'client': client})
        GoogleGeminiChat.__init__(self, 
            config={'api_key': os.getenv('GEMINI_API_KEY'), 'model_name': os.getenv('GEMINI_MODEL')}
        )

vn_local = MyVanna()
vn_local.connect_to_bigquery(project_id='')

logger.info(f"TOOL LOADED: Vanna.ai configured for LOCAL (GemGKey) mode from '{QDRANT_PATH}'.")
# --- END VANNA CONFIG ---


# --- LAZY LOADING SETUP ---
G = None
# --- END LAZY LOADING SETUP ---

# --- HELPER FUNCTION TO LOAD GRAPH ---
def get_graph():
    """Helper to lazy-load the graph."""
    global G
    if G is None:
        logger.info("LAZY LOAD: Loading graph.gpickle into memory (first call)...")
        GRAPH_FILE = os.path.join(DATA_DIR, "graph.gpickle") 
        if not os.path.exists(GRAPH_FILE):
             logger.error(f"FATAL: Graph file not found at {GRAPH_FILE}")
             return None
        G = pickle.load(open(GRAPH_FILE, 'rb'))
        logger.info(f"LAZY LOAD: Graph loaded with {G.number_of_nodes()} nodes.")
    return G
# --- END HELPER FUNCTION ---


# --- 1. Vanna.ai SQL Tool (REAL) ---
def ask_vanna_ai(question: str) -> str:
    """
    Use this tool to answer any questions about sales, revenue, 
    customer counts, or other data in the SQL database.
    """
    logger.info(f"REAL CALL (LOCAL): 'ask_vanna_ai' received: {question}")
    try:
        answer = vn_local.ask(question)
        
        if answer is None:
            logger.error("VANNA ERROR: vn_local.ask() returned None.")
            return json.dumps({"error": "Vanna was unable to generate a response (ask() returned None)."})

        answer_text = None
        answer_sql = None

        if isinstance(answer, tuple):
            logger.info("Vanna returned a tuple. Unpacking...")
            answer_sql, answer_text = answer[0], answer[1]
            
            if answer_text is None:
                logger.error("VANNA ERROR: Vanna returned a tuple but the answer was None.")
                return json.dumps({"error": "Vanna returned an empty tuple, indicating an internal failure."})
            
            if isinstance(answer_text, pd.DataFrame):
                logger.info("Answer is a DataFrame. Converting to MARKDOWN...")
                answer_text = answer_text.to_markdown(index=False)
            
        else:
            logger.info("Vanna returned an Answer object. Accessing .text and .sql...")
            answer_text, answer_sql = answer.text, answer.sql
        
        logger.info(f"VANNA SUCCESS: Returning answer for: {question}")
        return json.dumps({"status": "success", "answer": answer_text, "query": answer_sql})
        
    except Exception as e:
        logger.error(f"Vanna exception: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Vanna exception: {str(e)}"})


# --- 2. NetworkX Graph Tool (Simple Text) ---
def get_graph_connections(source_node_type: str, source_node_id: str) -> str:
    """
    Use this simple tool to find the *direct*, 1st-degree connections 
    for a single node (e.g., 'What did USER 123 buy?').
    This returns TEXT, not a visual.
    """
    logger.info(f"REAL CALL: 'get_graph_connections' received: {source_node_type}_{source_node_id}")
    try:
        G = get_graph() # Lazy-load graph
        if G is None: return json.dumps({"status": "error", "error": "Graph file not found."})
        
        node_id = f"{source_node_type.upper()}_{source_node_id}"
        if node_id not in G:
            logger.warning(f"GRAPH WARNING: Node {node_id} not found.")
            return json.dumps({"status": "error", "error": f"Node {node_id} not found."})

        connections = {}
        # Get outgoing connections
        for successor in G.successors(node_id):
            edge_data = G.get_edge_data(node_id, successor)
            relation_type = edge_data.get('type', 'UNKNOWN_RELATION')
            
            successor_data = G.nodes[successor]
            s_type, s_id, s_detail = successor_data.get('type'), successor_data.get('id'), successor_data.get('detail', '')
            
            if relation_type not in connections: connections[relation_type] = []
            connections[relation_type].append(f"{s_type} {s_id} ({s_detail})")
        
        # Get incoming connections
        for predecessor in G.predecessors(node_id):
            edge_data = G.get_edge_data(predecessor, node_id)
            relation_type = edge_data.get('type', 'UNKNOWN_RELATION') + "_BY"
            
            predecessor_data = G.nodes[predecessor]
            p_type, p_id, p_detail = predecessor_data.get('type'), predecessor_data.get('id'), predecessor_data.get('detail', '')

            if relation_type not in connections: connections[relation_type] = []
            connections[relation_type].append(f"{p_type} {p_id} ({p_detail})")
            
        if not connections:
            logger.info(f"GRAPH INFO: Node {node_id} has no connections.")
            return json.dumps({"status": "success", "message": f"Node {node_id} has no connections."})
        
        logger.info(f"GRAPH SUCCESS: Returning {len(connections)} connection types for {node_id}.")
        return json.dumps({"status": "success", "connections": connections})
    
    except Exception as e:
        logger.error(f"Graph tool exception: {str(e)}", exc_info=True)
        return json.dumps({"status": "error", "error": f"Graph tool exception: {str(e)}"})


# --- 3. Graph Tool (Degree of Separation - Text) ---
def find_shortest_path(node_1_type: str, node_1_id: str, node_2_type: str, node_2_id: str) -> str:
    """
    Use this advanced tool to find the shortest path and degree of
    separation between any two nodes in the graph (e.g., 'How is USER 123
    connected to USER 456?'). This returns TEXT.
    """
    logger.info(f"REAL CALL: 'find_shortest_path' between {node_1_type}_{node_1_id} and {node_2_type}_{node_2_id}")
    try:
        G = get_graph()
        if G is None: return json.dumps({"status": "error", "error": "Graph file not found."})
        
        node_1 = f"{node_1_type.upper()}_{node_1_id}"
        node_2 = f"{node_2_type.upper()}_{node_2_id}"

        if node_1 not in G: return json.dumps({"status": "error", "error": f"Node {node_1} not found."})
        if node_2 not in G: return json.dumps({"status": "error", "error": f"Node {node_2} not found."})

        G_undirected = G.to_undirected(as_view=True)
        
        try:
            path = nx.shortest_path(G_undirected, source=node_1, target=node_2)
            degree = len(path) - 1
            
            formatted_path = []
            for node_id in path:
                node_data = G.nodes[node_id]
                formatted_path.append(f"{node_data.get('type')}_{node_data.get('id')}")
                
            logger.info(f"PATH SUCCESS: Found path of degree {degree}.")
            return json.dumps({"status": "success", "degree_of_separation": degree, "path": formatted_path})
            
        except nx.NetworkXNoPath:
            logger.warning("PATH WARNING: No path found between nodes.")
            return json.dumps({"status": "error", "message": f"No path found between {node_1} and {node_2}."})
            
    except Exception as e:
        logger.error(f"Pathfinding tool exception: {str(e)}", exc_info=True)
        return json.dumps({"status": "error", "error": f"Pathfinding tool exception: {str(e)}"})


# --- 4. Graph Tool (Collaborative Filtering - Text) ---
def get_collaborative_recommendations(product_id: str) -> str:
    """
    Use this advanced tool for 'users who bought this product also
    bought...' recommendations. This returns TEXT.
    """
    logger.info(f"REAL CALL: 'get_collaborative_recommendations' for PRODUCT_{product_id}")
    try:
        G = get_graph()
        if G is None: return json.dumps({"status": "error", "error": "Graph file not found."})

        start_product_node = f"PRODUCT_{product_id}"

        if start_product_node not in G:
            return json.dumps({"status": "error", "error": f"Node {start_product_node} not found."})

        product_counts = {}
        # Get all users (predecessors) who bought this product
        users_who_bought = G.predecessors(start_product_node)
        
        for user_node in users_who_bought:
            # Get all nodes this user is connected to
            other_nodes = G.successors(user_node)
            
            for node in other_nodes:
                if node == start_product_node: 
                    continue # Skip the original product

                # Get the node's data
                node_data = G.nodes[node]

                # --- THIS IS THE FIX ---
                # Check if this node is actually a PRODUCT before counting it
                if node_data.get('type') == 'PRODUCT':
                    # It IS a product, so add it to the recommendations
                    product_key = (node_data.get('id'), node_data.get('detail'))
                    product_counts[product_key] = product_counts.get(product_key, 0) + 1
                # --- END FIX ---
                # (If it's not a PRODUCT, the loop just continues, ignoring it)
        
        if not product_counts:
            return json.dumps({"status": "success", "message": "Users who bought this product did not purchase any other items."})

        sorted_products = sorted(product_counts.items(), key=lambda item: item[1], reverse=True)
        
        recommendations = []
        for (prod_id, prod_name), count in sorted_products[:5]: # Get top 5
            recommendations.append({
                "product_id": prod_id,
                "name": prod_name,
                "also_bought_count": count
            })
        
        logger.info(f"RECOMMENDATION SUCCESS: Found {len(recommendations)} recommendations.")
        return json.dumps({"status": "success", "recommendations": recommendations})

    except Exception as e:
        logger.error(f"Recommendation tool exception: {str(e)}", exc_info=True)
        return json.dumps({"status": "error", "error": f"Recommendation tool exception: {str(e)}"})


# --- 5. NEW VISUALIZER TOOL ---

# --- HELPER FUNCTION (Copied from Streamlit) ---
def create_plotly_graph_visualization(subgraph, G):
    """Creates an interactive Plotly visualization for a NetworkX subgraph."""
    if subgraph is None or subgraph.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(subgraph, k=0.7, iterations=40, seed=42)

    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    color_map = {
        'USER': '#4287f5', 'PRODUCT': '#f5a442', 'IP_ADDRESS': '#f54242',
        'SESSION': '#f542f2', 'CITY': '#42f56c', 'CENTER': '#a45d2f',
        'CATEGORY': '#7b42f5'
    }
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        node_id = node_data.get('id', node)
        node_detail = node_data.get('detail', '')
        
        node_text.append(f"Type: {node_type}<br>ID: {node_id}<br>Detail: {node_detail}")
        node_colors.append(color_map.get(node_type, 'gray'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_colors,
            size=15,
            line=dict(width=1, color='black')
        ))

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict(
                        text='Interactive Graph Visualization',
                        font=dict(size=16)
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig
# --- END HELPER FUNCTION ---

def visualize_node_to_file(node_type: str, node_id: str) -> str:
    """
    Use this tool when the user explicitly asks to "visualize", "show graph",
    or "see" a specific node.
    This tool creates an interactive HTML file and returns the path to it.
    """
    logger.info(f"REAL CALL: 'visualize_node_to_file' for {node_type}_{node_id}")
    try:
        G = get_graph()
        if G is None: return json.dumps({"status": "error", "error": "Graph file not found."})
            
        full_node_id = f"{node_type.upper()}_{node_id}"
        if full_node_id not in G:
            logger.warning(f"GRAPH WARNING: Node {full_node_id} not found.")
            return json.dumps({"status": "error", "error": f"Node {full_node_id} not found."})

        # Get all 1st-degree neighbors (incoming and outgoing)
        nodes_to_plot = [full_node_id] + list(G.successors(full_node_id)) + list(G.predecessors(full_node_id))
        subgraph = G.subgraph(nodes_to_plot)
        
        fig = create_plotly_graph_visualization(subgraph, G)
        
        if fig is None:
            return json.dumps({"status": "error", "error": "Failed to generate figure."})

        # Save the figure to an HTML file
        # Ensure the DATA_DIR exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        file_name = f"{full_node_id}.html"
        file_path = os.path.join(DATA_DIR, file_name)
        
        fig.write_html(file_path)
        file_uri = Path(file_path).as_uri()
        
        logger.info(f"VISUALIZATION SUCCESS: Saved to {file_path}")
        # Return the path so the agent can show the user
        return json.dumps({"status": "success", "file_path": file_uri})

    except Exception as e:
        logger.error(f"Visualization tool exception: {str(e)}", exc_info=True)
        return json.dumps({"status": "error", "error": f"Visualization tool exception: {str(e)}"})
