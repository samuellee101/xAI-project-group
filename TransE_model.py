import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from rdflib import Graph
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

# TransE Model Definition
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1.0):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, head, relation, tail):
        h = self.entity_embeddings(head)
        r = self.relation_embeddings(relation)
        t = self.entity_embeddings(tail)
        score = torch.norm(h + r - t, p=2, dim=1)
        return score

# Margin Ranking Loss
class MarginRankingLoss(nn.Module):
    def __init__(self, margin):
        super(MarginRankingLoss, self).__init__()
        self.loss_fn = nn.MarginRankingLoss(margin)

    def forward(self, positive_scores, negative_scores):
        target = torch.tensor([-1.0], dtype=torch.float32)
        return self.loss_fn(positive_scores, negative_scores, target)

# Data Preparation
def load_rdf_data(file_path):
    graph = Graph()
    graph.parse(file_path, format='nt')
    triples = list(graph)
    entities = set()
    relations = set()
    for s, p, o in triples:
        entities.add(str(s))
        entities.add(str(o))
        relations.add(str(p))
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    relation_to_idx = {r: i for i, r in enumerate(relations)}
    indexed_triples = [
        (entity_to_idx[str(s)], relation_to_idx[str(p)],
         entity_to_idx[str(o)]) for s, p, o in triples
    ]
    return indexed_triples, entity_to_idx, relation_to_idx

# Generate Negative Samples
def generate_negative_sample(triple, num_entities, all_triples):
    head, relation, tail = triple
    corrupted = []
    while True:
        new_head = random.randint(0, num_entities - 1)
        if (new_head, relation, tail) not in all_triples:
            corrupted.append((new_head, relation, tail))
            break
    while True:
        new_tail = random.randint(0, num_entities - 1)
        if (head, relation, new_tail) not in all_triples:
            corrupted.append((head, relation, new_tail))
            break
    return corrupted

# Evaluation (Hits@K and MRR)
def evaluate_model(model, triples, entity_to_idx, relation_to_idx, top_k=10):
    model.eval()
    hits = 0
    mrr = 0
    with torch.no_grad():
        for head, relation, tail in random.sample(triples, min(100, len(triples))):
            head_emb = model.entity_embeddings(torch.tensor([head]))
            rel_emb = model.relation_embeddings(torch.tensor([relation]))
            all_entities = model.entity_embeddings.weight
            scores = torch.norm(head_emb + rel_emb - all_entities, dim=1)
            ranked = torch.argsort(scores)
            rank = (ranked == tail).nonzero(as_tuple=True)[0].item() + 1
            mrr += 1.0 / rank
            if rank <= top_k:
                hits += 1
    print(f"\nEvaluation: Hits@{top_k} = {hits/100:.2f}, MRR = {mrr/100:.2f}\n")

# Explanation Function
def explain_prediction(
    head_label, rel_label, model, entity_to_idx, relation_to_idx, top_k=5
):
    h_idx = entity_to_idx.get(head_label)
    r_idx = relation_to_idx.get(rel_label)
    if h_idx is None or r_idx is None:
        print("Head or relation not found in training data.")
        return
    h = model.entity_embeddings.weight[h_idx].detach().numpy()
    r = model.relation_embeddings.weight[r_idx].detach().numpy()
    predicted = h + r
    similarities = cosine_similarity(
        [predicted],
        model.entity_embeddings.weight.detach().numpy()
    )[0]
    top_k_idx = similarities.argsort()[-top_k:][::-1]
    idx_to_entity = {v: k for k, v in entity_to_idx.items()}
    print(f"\nExplanation for: ({head_label}, {rel_label}, ?)")
    for i in top_k_idx:
        print(f"Rank {i}: {idx_to_entity[i]} (Score: {similarities[i]:.4f})")

# Train the TransE Model
def train_transe(model, data, num_entities, epochs=100, learning_rate=0.01):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MarginRankingLoss(model.margin)
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(data)
        for head, relation, tail in tqdm(data, desc=f"Epoch {epoch+1}"):
            negative_samples = generate_negative_sample(
                (head, relation, tail), num_entities, data
            )
            for neg_head, neg_relation, neg_tail in negative_samples:
                pos_score = model(
                    torch.tensor([head]),
                    torch.tensor([relation]),
                    torch.tensor([tail])
                )
                neg_score = model(
                    torch.tensor([neg_head]),
                    torch.tensor([neg_relation]),
                    torch.tensor([neg_tail])
                )
                loss = loss_fn(pos_score, neg_score)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model

# Plotly
def plot_clustered_embeddings_interactive(
    embeddings, labels, num_clusters=10, title="Interactive Clustered Embeddings"
):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    clusters = kmeans.labels_
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'Entity': labels,
        'Cluster': [f"Cluster {c}" for c in clusters]
    })
    fig = px.scatter(
        df, x='x', y='y', color='Cluster',
        hover_data=['Entity'], title=title
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.show()

# Main: Load RDF and Train
if __name__ == "__main__":
    rdf_file = "/Users/pruthvi/Desktop/XAIMP/xaiMP/cleaned_dbpedia_1000.nt"
    embedding_dim = 50
    margin = 1.0
    epochs = 50
    learning_rate = 0.01
    triples, entity_to_idx, relation_to_idx = load_rdf_data(rdf_file)
    num_entities = len(entity_to_idx)
    num_relations = len(relation_to_idx)
    print(
        f"Loaded {len(triples)} triples with "
        f"{num_entities} entities and {num_relations} relations."
    )

    model = TransE(num_entities, num_relations, embedding_dim, margin)
    trained_model = train_transe(
        model, triples, num_entities, epochs, learning_rate
    )

    torch.save(trained_model.state_dict(), "trained_transe_model.pth")
    print("Model training complete and saved.")

    # Save Entity Embeddings to Tables
    entity_embeddings = trained_model.entity_embeddings.weight.detach().numpy()
    entity_df = pd.DataFrame(
        entity_embeddings,
        index=[e for e, idx in sorted(entity_to_idx.items(), key=lambda x: x[1])]
    )
    entity_df.index.name = 'Entity'
    entity_df.columns = [f"Dim_{i}" for i in range(embedding_dim)]
    entity_df.to_csv("entity_embeddings.csv")
    print("Saved entity embeddings to entity_embeddings.csv")

    # Save Relation Embeddings to Tables
    relation_embeddings = trained_model.relation_embeddings.weight.detach().numpy()
    relation_df = pd.DataFrame(
        relation_embeddings,
        index=[r for r, idx in sorted(relation_to_idx.items(), key=lambda x: x[1])]
    )
    relation_df.index.name = 'Relation'
    relation_df.columns = [f"Dim_{i}" for i in range(embedding_dim)]
    relation_df.to_csv("relation_embeddings.csv")
    print("Saved relation embeddings to relation_embeddings.csv")

    print(entity_df.head())
    print(relation_df.head())

    plot_clustered_embeddings_interactive(
        entity_embeddings, list(entity_df.index), title="Entity Embeddings"
    )
    # Calling evaluation
    evaluate_model(trained_model, triples, entity_to_idx, relation_to_idx)
    # Calling explanation with an example
    example_head = list(entity_to_idx.keys())[0]
    example_relation = list(relation_to_idx.keys())[0]
    explain_prediction(
        example_head, example_relation,
        trained_model, entity_to_idx, relation_to_idx
    )
