"""
Main entry point for the Topic Sorting application.
"""
import json
import os
from pathlib import Path
from openai import OpenAI
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.express as px
from dotenv import load_dotenv
import numpy as np

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_data(file_path: str) -> dict:
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
def embeded_conversations(conversations: list) -> pd.DataFrame:
    """
    Embed the conversations into a list of topics.
    Each conversation's messages are combined into a single embedding.
    """
    embeddings = []
    for conversation_list in conversations:
        conversation = conversation_list[0]  # Get the first (and only) object in the list
        
        # Combine all messages from the conversation
        combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation['messages']])
        
        response = client.embeddings.create(
            input=combined_text,
            model="text-embedding-3-small"
        )
        embeddings.append({
            'chat_id': conversation['chat_id'],
            'content': combined_text,
            'embedding': response.data[0].embedding
        })
    return pd.DataFrame(embeddings)


def cluster_embeddings(embeddings: pd.DataFrame, n_clusters=10) -> pd.DataFrame:
    """
    Cluster the embeddings into a list of topics.
    """
    # Extract just the embedding vectors for clustering
    embedding_vectors = embeddings['embedding'].tolist()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    embeddings['cluster'] = kmeans.fit_predict(embedding_vectors)
    return embeddings

def label_clusters(embeddings: pd.DataFrame) -> dict:
    """
    Label the clusters with the most common topics.
    """
    labels = {}
    for cluster in embeddings['cluster'].unique():
        # Get sample messages from this cluster
        sample = embeddings[embeddings['cluster'] == cluster].sample(min(5, len(embeddings[embeddings['cluster'] == cluster])))
        sample_text = "\n---\n".join(sample['content'].tolist())
        prompt = f"""
Analyze the following support conversations and identify a hierarchical topic label. Include a high-level category and subcategory.

Conversations:
{sample_text}

Format:
Category: <top-level>
Subcategory: <specific issue>
"""
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        try:
            category = response.split("Category:")[1].split("\n")[0].strip()
            subcategory = response.split("Subcategory:")[1].split("\n")[0].strip()
        except IndexError:
            category, subcategory = "Uncategorized", "Uncategorized"

        labels[cluster] = {"category": category, "subcategory": subcategory}
    return labels

def visualize(embeddings: pd.DataFrame, labels: dict):
    """
    Visualize the clusters with Plotly.
    """
    # Extract embedding vectors for t-SNE and convert to numpy array
    embedding_vectors = np.array(embeddings['embedding'].tolist())
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embedding_vectors)
    
    # Add t-SNE results to DataFrame
    embeddings['tsne-2d-one'] = tsne_results[:,0]
    embeddings['tsne-2d-two'] = tsne_results[:,1]
    embeddings['label'] = embeddings['cluster'].map(lambda x: f"{labels[x]['category']} > {labels[x]['subcategory']}")
    
    # Create visualization
    fig = px.scatter(embeddings, 
                    x="tsne-2d-one", 
                    y="tsne-2d-two", 
                    color="label", 
                    hover_data=["chat_id", "content"])
    fig.update_layout(title="Conversation Topic Clustering")
    fig.show()

def generate_topic_breakdown(embeddings: pd.DataFrame, labels: dict):
    """
    Generate a topic breakdown of the clusters.
    """
    summary = []
    for cluster, group in embeddings.groupby('cluster'):
        label_info = labels[cluster]
        summary.append({
            "topic": f"{label_info['category']} > {label_info['subcategory']}",
            "count": len(group),
            "conversation_ids": group['chat_id'].unique().tolist(),
            "sample_messages": group['content'].sample(min(3, len(group))).tolist()
        })
    return summary

def analyze_trending_topics(embeddings: pd.DataFrame, labels: dict):
    """
    Analyze and display trending topics as a percentage table.
    """
    # Count conversations per cluster
    cluster_counts = embeddings['cluster'].value_counts()
    total_conversations = len(embeddings)
    
    # Create a list to store topic information
    topic_stats = []
    
    # Calculate percentages and gather information for each cluster
    for cluster, count in cluster_counts.items():
        percentage = (count / total_conversations) * 100
        label_info = labels[cluster]
        topic_stats.append({
            'Category': label_info['category'],
            'Subcategory': label_info['subcategory'],
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })
    
    # Convert to DataFrame for nice table display
    topic_df = pd.DataFrame(topic_stats)
    
    # Sort by count in descending order
    topic_df = topic_df.sort_values('Count', ascending=False)
    
    # Print the table
    print("\n=== Trending Topics ===")
    print(topic_df.to_string(index=False))
    print("\nTotal Conversations:", total_conversations)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / 'assets'

    ## Load chat logs
    conversations = load_data(str(assets_dir / 'chatlogs_v1.json'))
    print(f"Loaded {len(conversations)} conversations")
    
    # Iterate through conversations (each conversation is in a list with one object)
    # for conversation_list in conversations:
    #     conversation = conversation_list[0]  # Get the first (and only) object in the list
    #     print(conversation['chat_id'])
    #     for message in conversation['messages']:
    #         print(message['role'], ':', message['content'])
    #         print("-"*100)
    #     print("-"*100)
    
    # ## Embed the conversations
    embeddings = embeded_conversations(conversations)
    print(f"Embedded {len(embeddings)} conversations")
    print(embeddings.head())
    
    # ## Cluster the embeddings
    clustered_embeddings = cluster_embeddings(embeddings)
    print(f"Clustered {len(clustered_embeddings)} conversations")

    # ## Label the clusters
    labels = label_clusters(clustered_embeddings)
    print(f"Labelled {len(labels)} clusters")

    # ## Analyze trending topics
    analyze_trending_topics(clustered_embeddings, labels)

    # ## Visualize the clusters
    visualize(clustered_embeddings, labels)

    # ## Generate a topic breakdown
    topic_breakdown = generate_topic_breakdown(clustered_embeddings, labels)
    print(f"Generated {len(topic_breakdown)} topic breakdowns")
    with open("topic_breakdown.json", "w") as f:
        json.dump(topic_breakdown, f, indent=2)

if __name__ == "__main__":
    main() 