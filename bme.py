import marimo

__generated_with = "0.15.2"
app = marimo.App()


@app.cell
def _():
    from sentence_transformers import SentenceTransformer

    # Load the BGE-M3 model from HuggingFace on CPU
    model = SentenceTransformer('BAAI/bge-m3', device='cpu')
    return (model,)


@app.cell
def _(model):
    # Input sentences for inference
    sentences = [
        "My soul died a long, long time ago. It rotted away in place for cowardice, and this body goes on to witness the scars of my cowardice punishment.",
        "The price of sin is amortized over time",
        "Time is the great equalizer in life",
        "Every saint has a past, every sinner has a future.",
        "Coke Zero is vastly better than coke diet.",
        "Bananas cost about 25 cents a pop"
    ]

    # Generate dense embeddings
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings, sentences


@app.cell
def _(embeddings):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Assuming 'embeddings' is your array of 1024-size vectors
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue', alpha=0.6)

    for i, (x, y) in enumerate(reduced_embeddings):
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title("t-SNE Visualization of Vector Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    import duckdb

    # Connect to a DuckDB database (or use ':memory:' for transient storage)
    con = duckdb.connect('vector_store.db')

    # Install and load the Vector Similarity Search (vss) extension
    con.execute("INSTALL vss;")
    con.execute("LOAD vss;")

    # Example: Create a table with a vector column (e.g., 1536 dimensions for OpenAI embeddings)
    con.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id VARCHAR,
        text TEXT,
        vec FLOAT[1024]
    );
    """)
    return (con,)


@app.function
def insert_embeddings(con, sentences, embeddings):
    con.execute("INSERT INTO embeddings (text, vec) SELECT * FROM (SELECT unnest(?) AS text, unnest(?) AS vec)", [sentences, embeddings])


@app.cell
def _(con, embeddings, sentences):
    insert_embeddings(con, sentences, embeddings)
    return


@app.cell
def _(model):
    class embed_ergonomics:
        """Optional docstring."""
        def __init__(self, parameter):
            self.attribute = parameter

        def vectorize(self):
            return(model.encode(self.attribute, normalize_embeddings=True))
    return (embed_ergonomics,)


@app.cell
def _(embed_ergonomics):
    item = embed_ergonomics("What good is a man without a soul?")
    return (item,)


@app.cell
def _(con, item):
    con.execute("""
        SELECT text
        FROM embeddings
        ORDER BY array_distance(vec, ?::FLOAT[1024]) ASC
        LIMIT 3;
    """, [item.vectorize()]).fetchall()
    return


@app.cell
def _(item):
    item.vectorize()
    return


@app.cell
def _(con):
    con.execute("CREATE OR REPLACE TABLE embeddings AS SELECT DISTINCT * FROM embeddings")
    return


if __name__ == "__main__":
    app.run()
