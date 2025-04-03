import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

def load_retail_data():
    """
    Load retail dataset from Hugging Face or fall back to a direct source.
    """
    print("Loading retail dataset...")
    try:
        # Try to load from Hugging Face datasets
        from datasets import load_dataset
        dataset = load_dataset("singh-aditya/online_retail")
        df = dataset['train'].to_pandas()
        print(f"Dataset loaded successfully from Hugging Face with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading from Hugging Face: {e}")
        print("Falling back to direct download...")

        # Fallback to UCI repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
        try:
            df = pd.read_excel(url)
            print(f"Dataset loaded successfully from UCI with {df.shape[0]} rows.")
        except Exception as e2:
            print(f"Error with direct download: {e2}")
            # Final fallback to CSV version
            url2 = "https://raw.githubusercontent.com/KeithGalli/Pandas-Data-Science-Tasks/master/SampleData/online_retail.csv"
            df = pd.read_csv(url2, encoding='unicode_escape')
            print(f"Dataset loaded from GitHub with {df.shape[0]} rows.")

    return df

def explore_data(df):
    """
    Explore and visualize the dataset.
    """
    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print("\nSample data:")
    print(df.head())

    print("\nColumns and data types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())

    # Show unique countries
    print(f"\nNumber of countries: {df['Country'].nunique()}")
    print(f"Top 5 countries by order count:")
    print(df['Country'].value_counts().head())

    return df

def clean_data(df):
    """
    Clean and preprocess the dataset.
    """
    print("\n=== Data Preprocessing ===")

    # Make a copy to avoid modifying the original
    df_clean = df.copy()

    # Drop rows with missing CustomerID (we need this for grouping)
    before_count = df_clean.shape[0]
    df_clean = df_clean.dropna(subset=['CustomerID'])
    print(f"Removed {before_count - df_clean.shape[0]} rows with missing CustomerID")

    # Convert CustomerID to integer type
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

    # Remove canceled transactions (negative quantity)
    before_count = df_clean.shape[0]
    df_clean = df_clean[df_clean['Quantity'] > 0]
    print(f"Removed {before_count - df_clean.shape[0]} canceled transactions")

    # Remove rows with UnitPrice <= 0
    before_count = df_clean.shape[0]
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    print(f"Removed {before_count - df_clean.shape[0]} rows with invalid UnitPrice")

    # Convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # Calculate total amount spent
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

    print(f"Final cleaned dataset shape: {df_clean.shape}")

    return df_clean

def create_customer_features(df):
    """
    Aggregate data at customer level and create features for clustering.
    """
    print("\n=== Creating Customer Features ===")

    # Calculate recency
    max_date = df['InvoiceDate'].max()

    # Group by customer and calculate features
    customer_features = df.groupby('CustomerID').agg({
        'InvoiceNo': pd.Series.nunique,  # Number of orders
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency in days
        'Quantity': 'sum',  # Total quantity purchased
        'TotalAmount': 'sum',  # Total monetary value
        'StockCode': pd.Series.nunique  # Number of unique products
    })

    # Rename columns
    customer_features.columns = ['Frequency', 'Recency', 'Quantity', 'Monetary', 'UniqueProducts']

    # Calculate additional metrics
    customer_features['AvgOrderValue'] = customer_features['Monetary'] / customer_features['Frequency']
    customer_features['AvgQuantityPerOrder'] = customer_features['Quantity'] / customer_features['Frequency']

    # Remove extreme outliers (over 99.5 percentile)
    for col in customer_features.columns:
        threshold = customer_features[col].quantile(0.995)
        customer_features = customer_features[customer_features[col] <= threshold]

    print(f"Created features for {customer_features.shape[0]} customers")
    print("Features created:", customer_features.columns.tolist())

    return customer_features

def find_optimal_clusters(data, max_clusters=10):
    """
    Find the optimal number of clusters using the Elbow method and Silhouette scores.
    """
    print("\n=== Finding Optimal Number of Clusters ===")

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Calculate metrics for different numbers of clusters
    inertia_values = []
    silhouette_scores_list = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)

        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores_list.append(silhouette_avg)

        print(f"Clusters: {k}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_avg:.3f}")

    # Create plot for visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot elbow curve
    ax[0].plot(range(2, max_clusters + 1), inertia_values, marker='o')
    ax[0].set_title('Elbow Method')
    ax[0].set_xlabel('Number of Clusters')
    ax[0].set_ylabel('Inertia')
    ax[0].grid(True)

    # Plot silhouette scores
    ax[1].plot(range(2, max_clusters + 1), silhouette_scores_list, marker='o')
    ax[1].set_title('Silhouette Method')
    ax[1].set_xlabel('Number of Clusters')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('optimal_clusters.png')

    # Find optimal k based on silhouette score
    optimal_k = silhouette_scores_list.index(max(silhouette_scores_list)) + 2
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    return optimal_k, scaler, scaled_data

def perform_clustering(data, scaled_data, n_clusters):
    """
    Perform K-means clustering with the optimal number of clusters.
    """
    print(f"\n=== Performing K-means Clustering with {n_clusters} clusters ===")

    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Add clusters to the original data
    clustered_data = data.copy()
    clustered_data['Cluster'] = clusters

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)

    # Add PCA components to dataframe
    clustered_data['PC1'] = principal_components[:, 0]
    clustered_data['PC2'] = principal_components[:, 1]

    print("Clustering complete")
    print(f"Cluster distribution:")
    print(clustered_data['Cluster'].value_counts().sort_index())

    return clustered_data, kmeans, pca

def analyze_clusters(clustered_data):
    """
    Analyze and visualize the resulting clusters.
    """
    print("\n=== Analyzing Customer Segments ===")

    # Calculate cluster statistics
    cluster_stats = clustered_data.groupby('Cluster').mean()

    # Create scatter plot of clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=clustered_data,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='viridis',
        s=70,
        alpha=0.7
    )
    plt.title('Customer Segments Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('customer_segments_pca.png')

    # Create boxplots for key features
    key_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']
    plt.figure(figsize=(16, 12))

    for i, feature in enumerate(key_features):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='Cluster', y=feature, data=clustered_data, palette='viridis')
        plt.title(f'{feature} by Cluster')

    plt.tight_layout()
    plt.savefig('cluster_feature_distribution.png')

    # Create parallel coordinates plot
    plt.figure(figsize=(14, 8))

    # Normalize data for parallel plot
    parallel_data = clustered_data.copy()
    for feature in key_features:
        parallel_data[feature] = (parallel_data[feature] - parallel_data[feature].min()) / (parallel_data[feature].max() - parallel_data[feature].min())

    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(
        parallel_data,
        'Cluster',
        cols=key_features,
        color=plt.cm.viridis.colors
    )
    plt.title('Parallel Coordinates Plot of Customer Segments')
    plt.savefig('parallel_coordinates.png')

    return cluster_stats

def interpret_segments(cluster_stats):
    """
    Interpret and name the customer segments.
    """
    print("\n=== Customer Segment Interpretation ===")

    segment_names = {}
    segment_descriptions = {}

    for cluster in cluster_stats.index:
        stats = cluster_stats.loc[cluster]

        # Define segment based on characteristics
        high_monetary = stats['Monetary'] > cluster_stats['Monetary'].median()
        high_frequency = stats['Frequency'] > cluster_stats['Frequency'].median()
        low_recency = stats['Recency'] < cluster_stats['Recency'].median()
        high_aov = stats['AvgOrderValue'] > cluster_stats['AvgOrderValue'].median()

        # Assign segment name based on characteristics
        if high_monetary and high_frequency and low_recency:
            name = "Loyal High-Value Customers"
            description = "Frequent purchasers who spend a lot and have bought recently"
        elif high_monetary and not high_frequency:
            name = "Big Spenders (Low Frequency)"
            description = "Make large purchases but don't shop often"
        elif high_frequency and not high_monetary:
            name = "Frequent Small Purchasers"
            description = "Shop often but spend less per transaction"
        elif low_recency and not high_monetary and not high_frequency:
            name = "New Customers"
            description = "Recently started shopping but haven't established a clear pattern"
        elif not low_recency and not high_monetary and not high_frequency:
            name = "At-Risk Customers"
            description = "Haven't purchased recently and weren't very active"
        elif high_aov and not high_frequency:
            name = "Occasional High Spenders"
            description = "Make infrequent but large purchases"
        else:
            name = f"Segment {cluster}"
            description = "Mixed characteristics"

        segment_names[cluster] = name
        segment_descriptions[cluster] = description

        # Print segment details
        print(f"\nCluster {cluster}: {name}")
        print(f"  Description: {description}")
        print(f"  Average Monetary Value: ${stats['Monetary']:.2f}")
        print(f"  Average Frequency: {stats['Frequency']:.1f} orders")
        print(f"  Average Recency: {stats['Recency']:.0f} days")
        print(f"  Average Order Value: ${stats['AvgOrderValue']:.2f}")

    return segment_names, segment_descriptions

def create_marketing_recommendations(segment_names, segment_descriptions, cluster_stats):
    """
    Create marketing recommendations for each segment.
    """
    print("\n=== Marketing Recommendations ===")

    recommendations = {}

    for cluster in segment_names.keys():
        name = segment_names[cluster]
        stats = cluster_stats.loc[cluster]

        # Create tailored recommendations based on segment characteristics
        if "Loyal High-Value" in name:
            recommendations[cluster] = [
                "Implement loyalty rewards program",
                "Offer exclusive early access to new products",
                "Create a VIP customer club with special perks",
                "Send personalized thank you notes with orders"
            ]
        elif "Big Spenders" in name:
            recommendations[cluster] = [
                "Encourage more frequent purchases with time-limited offers",
                "Create bundle deals to increase purchase frequency",
                "Send personalized product recommendations based on past purchases",
                "Implement a 'complete the collection' marketing campaign"
            ]
        elif "Frequent Small" in name:
            recommendations[cluster] = [
                "Introduce tiered discounts to encourage larger basket sizes",
                "Cross-sell related products at checkout",
                "Create 'upgrade' offers to higher-priced alternatives",
                "Implement a points system that rewards larger purchases"
            ]
        elif "New Customers" in name:
            recommendations[cluster] = [
                "Send welcome series emails with product education",
                "Offer first-time buyer discounts on second purchase",
                "Share customer testimonials to build trust",
                "Create 'new customer favorite' product bundles"
            ]
        elif "At-Risk" in name:
            recommendations[cluster] = [
                "Implement win-back email campaign with special offers",
                "Ask for feedback to understand why they haven't returned",
                "Offer free shipping or discount on next purchase",
                "Showcase new products that match their previous interests"
            ]
        elif "Occasional High" in name:
            recommendations[cluster] = [
                "Send reminder emails when typical purchase cycle approaches",
                "Create seasonal campaigns aligned with their buying habits",
                "Offer financing options for large purchases",
                "Provide exclusive access to premium or limited edition products"
            ]
        else:
            recommendations[cluster] = [
                "Conduct further analysis to understand behavior",
                "Test different marketing approaches to find what resonates",
                "Segment this group further for more targeted approaches",
                "Gather more data on these customers' preferences"
            ]

        # Print recommendations
        print(f"\nMarketing Recommendations for {name}:")
        for rec in recommendations[cluster]:
            print(f"  • {rec}")

    return recommendations

def main():
    """
    Main function to run the customer segmentation analysis.
    """
    print("Starting Customer Segmentation Analysis")

    # Load and explore data
    df = load_retail_data()
    explore_data(df)

    # Clean data
    df_clean = clean_data(df)

    # Create customer features
    customer_features = create_customer_features(df_clean)

    # Find optimal number of clusters
    optimal_k, scaler, scaled_data = find_optimal_clusters(customer_features)

    # Perform clustering
    clustered_data, kmeans_model, pca = perform_clustering(customer_features, scaled_data, optimal_k)

    # Analyze clusters
    cluster_stats = analyze_clusters(clustered_data)

    # Interpret segments
    segment_names, segment_descriptions = interpret_segments(cluster_stats)

    # Create marketing recommendations
    recommendations = create_marketing_recommendations(segment_names, segment_descriptions, cluster_stats)

    # Save results
    clustered_data.to_csv('customer_segments_results.csv')
    print("\nResults saved to 'customer_segments_results.csv'")

    print("\nCustomer Segmentation Analysis Complete!")

    return {
        'clustered_data': clustered_data,
        'kmeans_model': kmeans_model,
        'segment_names': segment_names,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = main()
