"""
Main entry point for the Topic Sorting application.
"""
import json
import os
import time
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

class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name}: {duration:.2f} seconds")

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
    
def segment_conversation(messages: list, max_segment_length: int = 5) -> list:
    """
    Split a conversation into meaningful segments based on context.
    Each segment should represent a distinct topic or interaction.
    """
    segments = []
    current_segment = []
    current_role = None
    
    for message in messages:
        # Start a new segment if:
        # 1. We've reached max length
        # 2. The role changes (e.g., from user to agent)
        # 3. The message is very long (indicating a new topic)
        if (len(current_segment) >= max_segment_length or
            (current_role and message['role'] != current_role) or
            len(message['content'].split()) > 50):  # Long messages might indicate new topics
            
            if current_segment:  # Only add if we have messages
                segments.append(current_segment)
            current_segment = []
            current_role = message['role']
        
        current_segment.append(message)
        current_role = message['role']
    
    # Add any remaining messages as the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Print segmentation info for debugging
    # print(f"Created {len(segments)} segments from {len(messages)} messages")
    # for i, segment in enumerate(segments):
    #     print(f"Segment {i+1}: {len(segment)} messages")
    
    return segments

def embeded_conversations(conversations: list) -> pd.DataFrame:
    """
    Embed the conversations into a list of topics.
    Each conversation is split into segments, and each segment is embedded separately.
    """
    embeddings = []
    for conversation_list in conversations:
        conversation = conversation_list[0]  # Get the first (and only) object in the list
        
        # Split conversation into segments
        segments = segment_conversation(conversation['messages'])
        
        for i, segment in enumerate(segments):
            # Combine messages in the segment
            combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in segment])
            
            response = client.embeddings.create(
                input=combined_text,
                model="text-embedding-3-small"
            )
            embeddings.append({
                'chat_id': conversation['chat_id'],
                'segment_id': i,
                'content': combined_text,
                'embedding': response.data[0].embedding
            })
    return pd.DataFrame(embeddings)

def cluster_embeddings(embeddings: pd.DataFrame, n_clusters_range: range = range(5, 11)) -> dict:
    """
    Cluster the embeddings using soft clustering to allow multiple topics per segment.
    Now supports multiple k values for comparison.
    
    Args:
        embeddings (pd.DataFrame): DataFrame containing embeddings
        n_clusters_range (range): Range of k values to try (default: range(5, 11) for k=5 to k=10)
        
    Returns:
        dict: Dictionary containing clustering results for each k value
    """
    results = {}
    
    for n_clusters in n_clusters_range:
        # Extract just the embedding vectors for clustering
        embedding_vectors = np.array(embeddings['embedding'].tolist())
        
        # Use KMeans for initial clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_centers = kmeans.fit(embedding_vectors).cluster_centers_
        
        # Calculate distances to all cluster centers
        distances = np.zeros((len(embedding_vectors), n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.linalg.norm(embedding_vectors - cluster_centers[i], axis=1)
        
        # Convert distances to similarities (using softmax)
        similarities = np.exp(-distances)
        similarities = similarities / similarities.sum(axis=1, keepdims=True)
        
        # Assign top 3 clusters for each segment
        top_clusters = np.argsort(similarities, axis=1)[:, -3:]
        
        # Create a copy of the embeddings DataFrame for this k value
        k_embeddings = embeddings.copy()
        k_embeddings['primary_cluster'] = top_clusters[:, 2]
        k_embeddings['secondary_cluster'] = top_clusters[:, 1]
        k_embeddings['tertiary_cluster'] = top_clusters[:, 0]
        
        results[n_clusters] = k_embeddings
    
    return results

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
        total_segments = len(embeddings)
        avg_cluster_size = total_segments / k
        max_cluster_size = primary_counts.max()
        min_cluster_size = primary_counts.min()
        cluster_size_std = primary_counts.std()
        
        comparison_data.append({
            'k': k,
            'Total Segments': total_segments,
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

def label_clusters(embeddings: pd.DataFrame) -> dict:
    """
    Label the clusters with the most common topics.
    """
    labels = {}
    for cluster in embeddings['primary_cluster'].unique():
        # Get sample messages from this cluster
        sample = embeddings[embeddings['primary_cluster'] == cluster].sample(min(5, len(embeddings[embeddings['primary_cluster'] == cluster])))
        sample_text = "\n---\n".join(sample['content'].tolist())
        prompt = f"""
Analyze the following support conversations and identify a topic label. 

Conversations:
{sample_text}

Format:
Category: <topic>
"""
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        response = completion.choices[0].message.content
        try:
            topic = response.split("Category:")[1].split("\n")[0].strip()
            
        except IndexError:
            topic = "Uncategorized"

        labels[cluster] = {"topic": topic}
    return labels

def visualize(embeddings: pd.DataFrame, labels: dict):
    """
    Visualize the clusters with Plotly.
    Now shows both primary and secondary cluster assignments.
    """
    # Extract embedding vectors for t-SNE
    embedding_vectors = np.array(embeddings['embedding'].tolist())
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embedding_vectors)
    
    # Add t-SNE results to DataFrame
    embeddings['tsne-2d-one'] = tsne_results[:,0]
    embeddings['tsne-2d-two'] = tsne_results[:,1]
    
    # Create labels for both primary and secondary clusters
    embeddings['primary_label'] = embeddings['primary_cluster'].map(
        lambda x: f"{labels[x]['topic']}"
    )
    embeddings['secondary_label'] = embeddings['secondary_cluster'].map(
        lambda x: f"{labels[x]['topic']}"
    )
    embeddings['tertiary_label'] = embeddings['tertiary_cluster'].map(
        lambda x: f"{labels[x]['topic']}"
    )
    
    # Create visualization
    fig = px.scatter(embeddings, 
                    x="tsne-2d-one", 
                    y="tsne-2d-two", 
                    color="primary_label",
                    hover_data=["chat_id", "segment_id", "primary_label", "secondary_label", "tertiary_label", "content"])
    fig.update_layout(title="Conversation Topic Clustering (Primary Topics)")
    fig.show()
    
    # Create second visualization for secondary topics
    fig2 = px.scatter(embeddings, 
                     x="tsne-2d-one", 
                     y="tsne-2d-two", 
                     color="secondary_label",
                     hover_data=["chat_id", "segment_id", "primary_label", "secondary_label", "tertiary_label", "content"])
    fig2.update_layout(title="Conversation Topic Clustering (Secondary Topics)")
    fig2.show()

    # Create third visualization for tertiary topics
    fig3 = px.scatter(embeddings, 
                     x="tsne-2d-one", 
                     y="tsne-2d-two", 
                     color="tertiary_label",
                     hover_data=["chat_id", "segment_id", "primary_label", "secondary_label", "tertiary_label", "content"])
    fig3.update_layout(title="Conversation Topic Clustering (Tertiary Topics)")
    fig3.show()

def generate_topic_breakdown(embeddings: pd.DataFrame, labels: dict):
    """
    Generate a topic breakdown of the clusters.
    """
    summary = []
    for cluster, group in embeddings.groupby('primary_cluster'):
        label_info = labels[cluster]
        summary.append({
            "topic": f"{label_info['topic']}",
            "count": len(group),
            "conversation_ids": group['chat_id'].unique().tolist(),
            "sample_messages": group['content'].sample(min(3, len(group))).tolist()
        })
    return summary

def analyze_trending_topics(embeddings: pd.DataFrame, labels: dict):
    """
    Analyze and display trending topics in three different ways:
    1. Only primary topics
    2. Primary + Secondary topics
    3. Primary + Secondary + Tertiary topics
    """
    # Count clusters
    primary_counts = embeddings['primary_cluster'].value_counts()
    # secondary_counts = embeddings['secondary_cluster'].value_counts()
    # tertiary_counts = embeddings['tertiary_cluster'].value_counts()
    total_segments = len(embeddings)
    
    # Create a list to store topic information for each counting method
    topic_stats_primary = []
    topic_stats_primary_secondary = []
    topic_stats_all = []
    
    # Calculate statistics for all clusters
    # all_clusters = set(primary_counts.index) | set(secondary_counts.index) | set(tertiary_counts.index)
    for cluster in primary_counts.index:
        primary_count = primary_counts.get(cluster, 0)
        # secondary_count = secondary_counts.get(cluster, 0)
        # tertiary_count = tertiary_counts.get(cluster, 0)
        
        label_info = labels[cluster]
        base_info = {
            'Topic': label_info['topic'],
            
        }
        
        # 1. Primary only
        primary_percentage = (primary_count / total_segments) * 100
        topic_stats_primary.append({
            **base_info,
            'Count': primary_count,
            'Percentage': f"{primary_percentage:.1f}%"
        })
        
        # 2. Primary + Secondary
        # primary_secondary_count = primary_count + secondary_count
        # primary_secondary_percentage = (primary_secondary_count / total_segments) * 100
        # topic_stats_primary_secondary.append({
        #     'Count': primary_secondary_count,
        #     'Percentage': f"{primary_secondary_percentage:.1f}%"
        # })
        
        # 3. All three
        # total_count = primary_count + secondary_count + tertiary_count
        # total_percentage = (primary_count / total_segments) * 100
        # topic_stats_all.append({
        #     **base_info,
        #     'Count': primary_count,
        #     'Percentage': f"{total_percentage:.1f}%"
        # })
    
    # Convert to DataFrames and sort by count
    df_primary = pd.DataFrame(topic_stats_primary).sort_values('Count', ascending=False)
    # df_primary_secondary = pd.DataFrame(topic_stats_primary_secondary).sort_values('Count', ascending=False)
    # df_all = pd.DataFrame(topic_stats_all).sort_values('Count', ascending=False)
    
    # Print the tables
    print("\n=== Trending Topics (Primary Only) ===")
    print(df_primary.to_string(index=False))
    
    # print("\n=== Trending Topics (Primary + Secondary) ===")
    # print(df_primary_secondary.to_string(index=False))
    
    # print("\n=== Trending Topics (All Topics) ===")
    # print(df_all.to_string(index=False))
    
    print(f"\nTotal Conversation Segments: {total_segments}")

def analyze_whole_conversation_topics(conversations: list, n_clusters_range: range = range(5, 11)) -> dict:
    """
    Analyze entire conversations to find their main topics without segmentation.
    Each conversation is embedded as a whole and assigned 2-3 main topics.
    Now supports multiple k values for comparison.
    
    Args:
        conversations (list): List of conversations to analyze
        n_clusters_range (range): Range of k values to try (default: range(5, 11) for k=5 to k=10)
        
    Returns:
        dict: Dictionary containing clustering results for each k value
    """
    embeddings = []
    for conversation_list in conversations:
        conversation = conversation_list[0]  # Get the first (and only) object in the list
        
        # Combine all messages in the conversation
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
    
    # Convert to DataFrame
    embeddings_df = pd.DataFrame(embeddings)
    results = {}
    
    # Extract embedding vectors for clustering
    embedding_vectors = np.array(embeddings_df['embedding'].tolist())
    
    for n_clusters in n_clusters_range:
        # Use KMeans for initial clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_centers = kmeans.fit(embedding_vectors).cluster_centers_
        
        # Calculate distances to all cluster centers
        distances = np.zeros((len(embedding_vectors), n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.linalg.norm(embedding_vectors - cluster_centers[i], axis=1)
        
        # Convert distances to similarities (using softmax)
        similarities = np.exp(-distances)
        similarities = similarities / similarities.sum(axis=1, keepdims=True)
        
        # Assign top 3 clusters for each conversation
        top_clusters = np.argsort(similarities, axis=1)[:, -3:]
        
        # Create a copy of the embeddings DataFrame for this k value
        k_embeddings = embeddings_df.copy()
        k_embeddings['primary_cluster'] = top_clusters[:, 2]
        # k_embeddings['secondary_cluster'] = top_clusters[:, 1]
        # k_embeddings['tertiary_cluster'] = top_clusters[:, 0]
        
        results[n_clusters] = k_embeddings
    
    return results

def compare_whole_conversation_clustering(clustering_results: dict):
    """
    Compare whole conversation clustering results across different k values.
    
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
    print("\n=== Whole Conversation Clustering Comparison (k=5 to k=10) ===")
    print(comparison_df.to_string(index=False))
    
    # Save to CSV for easier viewing
    comparison_df.to_csv('outputs/whole_conversation_clustering_comparison.csv', index=False)
    print("\nComparison saved to 'whole_conversation_clustering_comparison.csv'")

def analyze_whole_conversation_trends(clustering_results: dict, labels: dict):
    """
    Analyze and display trending topics for whole conversations across different k values.
    Shows primary topics for each k value.
    
    Args:
        clustering_results (dict): Dictionary containing clustering results for each k value
        labels (dict): Dictionary containing labels for each cluster
    """
    for k, embeddings in clustering_results.items():
        print(f"\n=== Whole Conversation Topics (k={k}) ===")
        
        # Count clusters
        primary_counts = embeddings['primary_cluster'].value_counts()
        total_conversations = len(embeddings)
        
        # Create a list to store topic information
        topic_stats = []
        
        # Calculate statistics for all clusters
        for cluster in primary_counts.index:
            primary_count = primary_counts.get(cluster, 0)
            
            label_info = labels[cluster]
            base_info = {
                'Topic': label_info['topic'],
                
            }
            
            # Calculate percentage
            percentage = (primary_count / total_conversations) * 100
            topic_stats.append({
                **base_info,
                'Count': primary_count,
                'Percentage': f"{percentage:.1f}%"
            })
        
        # Convert to DataFrame and sort by count
        df = pd.DataFrame(topic_stats).sort_values('Count', ascending=False)
        
        # Print the table
        print(df.to_string(index=False))
        print(f"\nTotal Conversations: {total_conversations}")
        
        # Save to CSV for easier viewing
        df.to_csv(f'outputs/whole_conversation_trends_k{k}.csv', index=False)
        print(f"Trends saved to 'whole_conversation_trends_k{k}.csv'")

def visualize_whole_conversations(embeddings: pd.DataFrame, labels: dict):
    """
    Visualize the whole conversation clusters with Plotly.
    Shows both primary and secondary cluster assignments.
    """
    # Extract embedding vectors for t-SNE
    embedding_vectors = np.array(embeddings['embedding'].tolist())
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_results = tsne.fit_transform(embedding_vectors)
    
    # Add t-SNE results to DataFrame
    embeddings['tsne-2d-one'] = tsne_results[:,0]
    embeddings['tsne-2d-two'] = tsne_results[:,1]
    
    # Create labels for both primary and secondary clusters
    embeddings['primary_label'] = embeddings['primary_cluster'].map(
        lambda x: f"{labels[x]['category']}"
    )
    # embeddings['secondary_label'] = embeddings['secondary_cluster'].map(
    #     lambda x: f"{labels[x]['category']} > {labels[x]['subcategory']}"
    # )
    # embeddings['tertiary_label'] = embeddings['tertiary_cluster'].map(
    #     lambda x: f"{labels[x]['category']} > {labels[x]['subcategory']}"
    # )
    
    # Create visualization for primary topics
    fig = px.scatter(embeddings, 
                    x="tsne-2d-one", 
                    y="tsne-2d-two", 
                    color="primary_label",
                    hover_data=["chat_id", "primary_label", "content"])
    fig.update_layout(title="Whole Conversation Topic Clustering (Primary Topics)")
    fig.show()
    
    # # Create visualization for secondary topics
    # fig2 = px.scatter(embeddings, 
    #                  x="tsne-2d-one", 
    #                  y="tsne-2d-two", 
    #                  color="secondary_label",
    #                  hover_data=["chat_id", "primary_label", "secondary_label", "tertiary_label", "content"])
    # fig2.update_layout(title="Whole Conversation Topic Clustering (Secondary Topics)")
    # fig2.show()

    # # Create visualization for tertiary topics
    # fig3 = px.scatter(embeddings, 
    #                  x="tsne-2d-one", 
    #                  y="tsne-2d-two", 
    #                  color="tertiary_label",
    #                  hover_data=["chat_id", "primary_label", "secondary_label", "tertiary_label", "content"])
    # fig3.update_layout(title="Whole Conversation Topic Clustering (Tertiary Topics)")
    # fig3.show()

def print_topic_chat_mapping(embeddings: pd.DataFrame, labels: dict):
    """
    Print a detailed table showing topics and their associated chat IDs.
    Groups chats by their primary topic and shows all topic assignments.
    """
    # Create a list to store the mapping information
    topic_mapping = []
    
    # Group by primary cluster
    for cluster in sorted(embeddings['primary_cluster'].unique()):
        cluster_chats = embeddings[embeddings['primary_cluster'] == cluster]
        label_info = labels[cluster]
        
        # Get all chats in this primary cluster
        for _, chat in cluster_chats.iterrows():
            primary_label = f"{label_info['category']}"
            # secondary_label = f"{labels[chat['secondary_cluster']]['category']} > {labels[chat['secondary_cluster']]['subcategory']}"
            # tertiary_label = f"{labels[chat['tertiary_cluster']]['category']} > {labels[chat['tertiary_cluster']]['subcategory']}"
            
            topic_mapping.append({
                'Primary Topic': primary_label,
                # 'Secondary Topic': secondary_label,
                # 'Tertiary Topic': tertiary_label,
                'Chat ID': chat['chat_id']
            })
    
    # Convert to DataFrame and sort by Primary Topic and Chat ID
    mapping_df = pd.DataFrame(topic_mapping)
    mapping_df = mapping_df.sort_values(['Primary Topic', 'Chat ID'])
    
    # Print the table
    # print("\n=== Topic-Chat Mapping ===")
    # print(mapping_df.to_string(index=False))
    
    # Also save to CSV for easier viewing in spreadsheet software
    mapping_df.to_csv('outputs/topic_chat_mapping.csv', index=False)
    print("\nMapping saved to 'topic_chat_mapping.csv'")

def main():
    total_start_time = time.time()
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    assets_dir = project_root / 'assets'

    ## Load chat logs
    with Timer("Loading chat logs"):
        conversations = load_data(str(assets_dir / 'chatlogs_v1.json'))
        print(f"Loaded {len(conversations)} conversations")
    
    # ## Analyze whole conversations
    with Timer("Analyzing whole conversations"):
        whole_conversation_results = analyze_whole_conversation_topics(conversations)
        print(f"Analyzed {len(conversations)} whole conversations with k=5 to k=10")
        
        # Compare whole conversation clustering results
        with Timer("Comparing whole conversation clustering results"):
            compare_whole_conversation_clustering(whole_conversation_results)
        
        # Use k=10 for labeling clusters (you can change this to any k value)
        whole_conversation_embeddings = whole_conversation_results[10]
        
        # Label the clusters
        whole_conversation_labels = label_clusters(whole_conversation_embeddings)
        print(f"Labelled {len(whole_conversation_labels)} clusters")
        
        # Analyze trends for all k values
        with Timer("Analyzing trends for all k values"):
            analyze_whole_conversation_trends(whole_conversation_results, whole_conversation_labels)
        
        # Print topic-chat mapping
        with Timer("Creating topic-chat mapping"):
            print_topic_chat_mapping(whole_conversation_embeddings, whole_conversation_labels)
        
        # Visualize whole conversations
        with Timer("Visualizing whole conversations"):
            visualize_whole_conversations(whole_conversation_embeddings, whole_conversation_labels)

        # Generate topic breakdown
        with Timer("Generating topic breakdown"):
            topic_breakdown = generate_topic_breakdown(whole_conversation_embeddings, whole_conversation_labels)
            print(f"Generated {len(topic_breakdown)} topic breakdowns")
            with open("outputs/topic_breakdown.json", "w") as f:
                json.dump(topic_breakdown, f, indent=2)
    
    # # ## Embed the conversations (segmented)
    # with Timer("Embedding conversations"):
    #     embeddings = embeded_conversations(conversations)
    #     print(f"Embedded {len(embeddings)} conversation segments")
    
    # # ## Cluster the embeddings with multiple k values
    # with Timer("Clustering embeddings"):
    #     clustering_results = cluster_embeddings(embeddings)
    #     print(f"Clustered {len(embeddings)} segments with k=5 to k=10")
        
    #     # Compare clustering results
    #     with Timer("Comparing clustering results"):
    #         compare_clustering_results(clustering_results)
        
    #     # Use k=10 for the rest of the analysis (you can change this to any k value)
    #     clustered_embeddings = clustering_results[10]

    # # ## Label the clusters
    # with Timer("Labeling clusters"):
    #     labels = label_clusters(clustered_embeddings)
    #     print(f"Labelled {len(labels)} clusters")

    # # ## Analyze trending topics
    # with Timer("Analyzing trending topics"):
    #     analyze_trending_topics(clustered_embeddings, labels)

    # # ## Visualize the clusters
    # with Timer("Visualizing clusters"):
    #     visualize(clustered_embeddings, labels)

    # # ## Generate a topic breakdown
    # with Timer("Generating topic breakdown"):
    #     topic_breakdown = generate_topic_breakdown(clustered_embeddings, labels)
    #     print(f"Generated {len(topic_breakdown)} topic breakdowns")
    #     with open("outputs/topic_breakdown.json", "w") as f:
    #         json.dump(topic_breakdown, f, indent=2)
    
    # total_duration = time.time() - total_start_time
    # print(f"\nTotal execution time: {total_duration:.2f} seconds")

if __name__ == "__main__":
    main() 