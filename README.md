# XAI-project

######------------------------A Detailed Description Of How The Approach Works-------------------------------######

We have implemented a Knowledge Graph Embedding pipeline using the **TransE** model to embed RDF triples from DBpedia.

Knowledge Graph Embeddings represent entities and relations in a KG as high-dimensional vectors. These embeddings preserve semantic structure and allow us to perform reasoning, prediction, and clustering directly in vector space.

In **TransE**, each triple (head, relation, tail) is embedded with the assumption that:

**Head vector + Relation vector ≈ Tail vector**

The model uses this geometric relationship to assign a **score** to each triple:

**score = || h + r - t ||**

Where:
- `h`: Head entity embedding
- `r`: Relation embedding
- `t`: Tail entity embedding
- `|| · ||`: Euclidean norm (L2)

This score reflects the plausibility of a triple. **Lower scores = more likely true**.

### Model Training Process
- For every positive (true) triple in the RDF data, we generate **negative samples** by corrupting either the head or tail.
- A **margin ranking loss** function is used to train the model:

**Loss = max(0, score_pos - score_neg + margin)**

The goal is to:
- Minimize the score for **true** triples
- Maximize the score for **false** ones

### Hyperparameters Used
- **Embedding Dimension** = 50
- **Margin** = 1.0
- **Epochs** = 50
- **Learning Rate** = 0.01

---

######-------------------------Clear Instructions On How To Execute The Code----------------------------######

### Steps to Run the Model

1. **Install Python (Recommended: Python 3.10+)**
2. **Install required packages**:
   ```bash
   pip install numpy torch tqdm rdflib pandas scikit-learn plotly
   ```

3. **Put the following files in one folder**:
   - `TransE_Model.py` (the main script)
   - `cleaned_dbpedia_1000.nt` (the RDF input file with 1000 triples)

4. **Open terminal in that folder and run**:
   ```bash
   python TransE_Model.py
   ```

5. The script will:
   - Load RDF data
   - Train a TransE model
   - Save embeddings to:
     - `entity_embeddings.csv`
     - `relation_embeddings.csv`
   - A set of example output is stored in 'Observations' folder
   - Show an **interactive Plotly visualization** of clustered entity embeddings
   - Run **evaluation metrics (Hits@10, MRR)**
   - Print an **explanation of a sample triple prediction**

---

### Evaluation and Explanation

- **Hits@10** and **Mean Reciprocal Rank (MRR)** are printed after training.
- You can **hover over clusters** in the Plotly chart to explore semantic groupings.
- The explanation module uses **cosine similarity** to list the top-5 predicted tail entities for a given (head, relation) pair.

---

### Notes
- You can easily extend the code to 10,000 triples by using `cleaned_dbpedia_10000.nt`.
- Trained model is saved as `trained_transe_model.pth`.

---

### Example Outputs:
- `entity_embeddings.csv`: contains 50D vectors for each unique entity.
- `relation_embeddings.csv`: contains 50D vectors for each relation.
- `interactive scatter plot`: shows clustered 2D embeddings using PCA + KMeans.
- Printed explanation: shows top predictions for a test triple and why.

---

### Useful References

- [TransE paper (Bordes et al., 2013)](https://papers.nips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
- [DBpedia: A Crowd-Sourced Community Effort to Extract Structured Content from Wikipedia](https://wiki.dbpedia.org/)
- [RDFlib Documentation](https://rdflib.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Plotly Scatter Docs](https://plotly.com/python/line-and-scatter/)

---


Thank you for using this TransE pipeline! For questions or contributions, feel free to open an issue or pull request.
