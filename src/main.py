"""
Main entry point for the Topic Sorting application.
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Constants
DEFAULT_N_CLUSTERS_RANGE = range(3, 11)
DEFAULT_MAX_SEGMENT_LENGTH = 5
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TSNE_PERPLEXITY = 5
TSNE_RANDOM_STATE = 42
KMEANS_RANDOM_STATE = 0

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Timer:
    """Context manager for timing code execution."""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self) -> 'Timer':
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name}: {duration:.2f} seconds")

def load_data(file_path: str) -> Dict[str, Any]:
    """Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data as a dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)
    
def segment_conversation(messages: List[Dict[str, str]], max_segment_length: int = DEFAULT_MAX_SEGMENT_LENGTH) -> List[List[Dict[str, str]]]:
    """Split a conversation into meaningful segments based on context.
    
    Args:
        messages: List of conversation messages
        max_segment_length: Maximum number of messages per segment
        
    Returns:
        List of conversation segments
    """
    segments = []
    current_segment = []
    current_role = None
    
    for message in messages:
        if (len(current_segment) >= max_segment_length or
            (current_role and message['role'] != current_role) or
            len(message['content'].split()) > 50):
            
            if current_segment:
                segments.append(current_segment)
            current_segment = []
            current_role = message['role']
        
        current_segment.append(message)
        current_role = message['role']
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

def create_embeddings(texts: List[str], metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create embeddings for a list of texts.
    
    Args:
        texts: List of texts to embed
        metadata: List of metadata dictionaries for each text
        
    Returns:
        DataFrame containing embeddings and metadata
    """
    embeddings = []
    for text, meta in zip(texts, metadata):
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        embeddings.append({
            **meta,
            'content': text,
            'embedding': response.data[0].embedding
        })
    return pd.DataFrame(embeddings)

def cluster_texts(embeddings: pd.DataFrame, n_clusters_range: range = DEFAULT_N_CLUSTERS_RANGE) -> Dict[int, pd.DataFrame]:
    """Cluster texts using their embeddings.
    
    Args:
        embeddings: DataFrame containing text embeddings
        n_clusters_range: Range of k values to try
        
    Returns:
        Dictionary containing clustering results for each k value
    """
    results = {}
    embedding_vectors = np.array(embeddings['embedding'].tolist())
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE)
        cluster_centers = kmeans.fit(embedding_vectors).cluster_centers_
        
        distances = np.zeros((len(embedding_vectors), n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.linalg.norm(embedding_vectors - cluster_centers[i], axis=1)
        
        similarities = np.exp(-distances)
        similarities = similarities / similarities.sum(axis=1, keepdims=True)
        top_clusters = np.argsort(similarities, axis=1)[:, -1:]
        
        k_embeddings = embeddings.copy()
        k_embeddings['primary_cluster'] = top_clusters[:, 0]
        results[n_clusters] = k_embeddings
    
    return results

def prepare_conversation_texts(conversations: List[List[Dict[str, Any]]], segment: bool = False) -> tuple[List[str], List[Dict[str, Any]]]:
    """Prepare conversation texts for embedding.
    
    Args:
        conversations: List of conversations to process
        segment: Whether to segment conversations or keep them whole
        
    Returns:
        Tuple of (texts, metadata) for embedding
    """
    texts = []
    metadata = []
    
    for conversation_list in conversations:
        conversation = conversation_list[0]
        topics = conversation.get('topics', [])
        
        if segment:
            segments = segment_conversation(conversation['messages'])
            for i, segment in enumerate(segments):
                combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in segment])
                texts.append(combined_text)
                metadata.append({
                    'chat_id': conversation['chat_id'],
                    'segment_id': i,
                    'topics': topics
                })
        else:
            # Embed the whole conversation as a single text
            combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation['messages']])
            texts.append(combined_text)
            metadata.append({
                'chat_id': conversation['chat_id'],
                'topics': topics
            })
    
    return texts, metadata

def calculate_cluster_statistics(embeddings: pd.DataFrame, n_clusters: int) -> Dict[str, Any]:
    """Calculate statistics for a clustering result.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        n_clusters: Number of clusters used
        
    Returns:
        Dictionary containing clustering statistics
    """
    primary_counts = embeddings['primary_cluster'].value_counts()
    total_items = len(embeddings)
    
    return {
        'total_items': total_items,
        'avg_cluster_size': total_items / n_clusters,
        'max_cluster_size': primary_counts.max(),
        'min_cluster_size': primary_counts.min(),
        'cluster_size_std': primary_counts.std(),
        'unique_topics': len(primary_counts)
    }

def create_topic_stats_df(embeddings: pd.DataFrame, labels: Dict[int, Dict[str, str]]) -> pd.DataFrame:
    """Create a DataFrame of topic statistics.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        labels: Dictionary mapping cluster IDs to their labels
        
    Returns:
        DataFrame containing topic statistics
    """
    primary_counts = embeddings['primary_cluster'].value_counts()
    total_items = len(embeddings)
    
    topic_stats = []
    for cluster in primary_counts.index:
        count = primary_counts.get(cluster, 0)
        label_info = labels[cluster]
        percentage = (count / total_items) * 100
        
        topic_stats.append({
            'Topic': label_info['topic'],
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })
    
    return pd.DataFrame(topic_stats).sort_values('Count', ascending=False)

def save_analysis_results(df: pd.DataFrame, filename: str, title: str = None) -> None:
    """Save analysis results to CSV and print summary.
    
    Args:
        df: DataFrame containing analysis results
        filename: Name of the output CSV file
        title: Optional title to print before the results
    """
    if title:
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
    
    output_path = Path('outputs') / filename
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")

def label_clusters(embeddings: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Label the clusters with the most common topics.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        
    Returns:
        Dictionary mapping cluster IDs to their labels
    """
    labels = {}
    for cluster in embeddings['primary_cluster'].unique():
        # Get sample messages from this cluster
        sample = embeddings[embeddings['primary_cluster'] == cluster].sample(
            min(5, len(embeddings[embeddings['primary_cluster'] == cluster]))
        )
        sample_text = "\n---\n".join(sample['content'].tolist())
        
        prompt = f"""
Analyze the following support conversations and identify a topic label. 

Conversations:
{sample_text}

Format:
Category: <topic>
"""
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        
        try:
            topic = response.split("Category:")[1].split("\n")[0].strip()
        except IndexError:
            topic = "Uncategorized"

        labels[cluster] = {"topic": topic}
    return labels

def visualize_clusters(embeddings: pd.DataFrame, labels: Dict[int, Dict[str, str]], k: int = None) -> None:
    """Visualize the clusters with Plotly.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        labels: Dictionary mapping cluster IDs to their labels
        k: Optional k value for the clustering (used in title)
    """
    # Extract embedding vectors for t-SNE
    embedding_vectors = np.array(embeddings['embedding'].tolist())
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=TSNE_RANDOM_STATE)
    tsne_results = tsne.fit_transform(embedding_vectors)
    
    # Add t-SNE results to DataFrame
    embeddings['tsne-2d-one'] = tsne_results[:,0]
    embeddings['tsne-2d-two'] = tsne_results[:,1]
    
    # Create labels for clusters
    embeddings['primary_label'] = embeddings['primary_cluster'].map(
        lambda x: f"{labels[x]['topic']}"
    )
    
    # Add a preview column for hover
    embeddings['content_preview'] = embeddings['content'].str.slice(0, 120) + '...'
    hover_data = ["chat_id", "primary_label", "topics", "content_preview"]
    if "segment_id" in embeddings.columns:
        hover_data.insert(1, "segment_id")
    
    # Create visualization
    title = f"Conversation Topic Clustering (k={k})" if k is not None else "Conversation Topic Clustering"
    fig = px.scatter(embeddings, 
                    x="tsne-2d-one", 
                    y="tsne-2d-two", 
                    color="primary_label",
                    hover_data=hover_data)
    fig.update_layout(title=title)
    fig.show()

def analyze_trending_topics(embeddings: pd.DataFrame, labels: Dict[int, Dict[str, str]], k: int = None) -> None:
    """Analyze and display trending topics.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        labels: Dictionary mapping cluster IDs to their labels
        k: Optional k value for the clustering (used in output filename)
    """
    df = create_topic_stats_df(embeddings, labels)
    
    # Create filename based on whether k is provided
    filename = f'trending_topics_k{k}.csv' if k is not None else 'trending_topics.csv'
    title = f'Trending Topics (k={k})' if k is not None else 'Trending Topics'
    
    save_analysis_results(df, filename, title)
    print(f"\nTotal Conversation Segments: {len(embeddings)}")

def generate_topic_breakdown(embeddings: pd.DataFrame, labels: Dict[int, Dict[str, str]]) -> List[Dict[str, Any]]:
    """Generate a topic breakdown of the clusters.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        labels: Dictionary mapping cluster IDs to their labels
        
    Returns:
        List of topic breakdowns
    """
    summary = []
    for cluster, group in embeddings.groupby('primary_cluster'):
        label_info = labels[cluster]
        summary.append({
            "topic": label_info['topic'],
            "count": len(group),
            "conversation_ids": group['chat_id'].unique().tolist(),
            "sample_messages": group['content'].sample(min(3, len(group))).tolist()
        })
    return summary

def compare_clustering_results(clustering_results: dict):
    """
    Compare clustering results across different k values.
    
    Args:
        clustering_results (dict): Dictionary containing clustering results for each k value
    """
    comparison_data = []
    
    for k, embeddings in clustering_results.items():
        # Calculate cluster sizes
        primary_counts = embeddings['primary_cluster'].value_counts()
        # secondary_counts = embeddings['secondary_cluster'].value_counts()
        # tertiary_counts = embeddings['tertiary_cluster'].value_counts()
        
        # Calculate statistics
        total_conversations = len(embeddings)
        avg_cluster_size = total_conversations / k
        max_cluster_size = primary_counts.max()
        min_cluster_size = primary_counts.min()
        cluster_size_std = primary_counts.std()
        
        comparison_data.append({
            'k': k,
            'Total Conversations': total_conversations,
            'Avg Cluster Size': f"{avg_cluster_size:.1f}",
            'Max Cluster Size': max_cluster_size,
            'Min Cluster Size': min_cluster_size,
            'Cluster Size Std Dev': f"{cluster_size_std:.1f}",
            'Unique Primary Topics': len(primary_counts),
            # 'Unique Secondary Topics': len(secondary_counts),
            # 'Unique Tertiary Topics': len(tertiary_counts)
        })
    
    # Convert to DataFrame and display
    comparison_df = pd.DataFrame(comparison_data)
    print("\n=== Clustering Comparison (k=5 to k=10) ===")
    print(comparison_df.to_string(index=False))
    
    # Save to CSV for easier viewing
    comparison_df.to_csv('outputs/clustering_comparison.csv', index=False)
    print("\nComparison saved to 'clustering_comparison.csv'")

def print_topic_chat_mapping(embeddings: pd.DataFrame, labels: Dict[int, Dict[str, str]]) -> None:
    """Print a detailed table showing topics and their associated chat IDs.
    
    Args:
        embeddings: DataFrame containing clustered embeddings
        labels: Dictionary mapping cluster IDs to their labels
    """
    topic_mapping = []
    
    for cluster in sorted(embeddings['primary_cluster'].unique()):
        cluster_chats = embeddings[embeddings['primary_cluster'] == cluster]
        label_info = labels[cluster]
        
        for _, chat in cluster_chats.iterrows():
            topic_mapping.append({
                'Primary Topic': label_info['topic'],
                'Chat ID': chat['chat_id']
            })
    
    mapping_df = pd.DataFrame(topic_mapping).sort_values(['Primary Topic', 'Chat ID'])
    save_analysis_results(mapping_df, 'topic_chat_mapping.csv')

def main():
    """Main entry point for the application."""
    total_start_time = time.time()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / 'assets'

    # Load chat logs
    with Timer("Loading chat logs"):
        conversations = load_data(str(assets_dir / 'chatlogs_v1.json'))
        print(f"Loaded {len(conversations)} conversations")
    
    # Process conversations
    with Timer("Processing conversations"):
        # Prepare texts and metadata
        texts, metadata = prepare_conversation_texts(conversations, segment=False)
        
        # Create embeddings
        embeddings = create_embeddings(texts, metadata)
        print(f"Created embeddings for {len(embeddings)} texts")
        
        # Cluster the embeddings
        clustering_results = cluster_texts(embeddings)
        print(f"Clustered texts with k=5 to k=10")
        
        # Compare clustering results
        with Timer("Comparing clustering results"):
            compare_clustering_results(clustering_results)
        
        # Analyze and visualize for each k value
        with Timer("Analyzing and visualizing for each k value"):
            for k, k_embeddings in clustering_results.items():
                print(f"\n=== Analyzing k={k} ===")
                labels = label_clusters(k_embeddings)
                print(f"Labelled {len(labels)} clusters")
                
                # Analyze trends
                analyze_trending_topics(k_embeddings, labels, k=k)
                
                # Visualize clusters
                print(f"\nVisualizing clusters for k={k}")
                visualize_clusters(k_embeddings, labels, k=k)
        
        # Use k=10 for detailed analysis
        clustered_embeddings = clustering_results[10]
        labels = label_clusters(clustered_embeddings)
        
        # Print topic-chat mapping
        with Timer("Creating topic-chat mapping"):
            print_topic_chat_mapping(clustered_embeddings, labels)

        # Generate topic breakdown
        with Timer("Generating topic breakdown"):
            topic_breakdown = generate_topic_breakdown(clustered_embeddings, labels)
            print(f"Generated {len(topic_breakdown)} topic breakdowns")
            with open("outputs/topic_breakdown.json", "w") as f:
                json.dump(topic_breakdown, f, indent=2)
    
    total_duration = time.time() - total_start_time
    print(f"\nTotal execution time: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 