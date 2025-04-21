# Connect to milvus
from pymilvus import (
    MilvusClient, DataType, Function, FunctionType
)

client = MilvusClient(uri="http://123.60.22.2:19530")
from llama_index.core import SimpleDirectoryReader
# Load data
documents = SimpleDirectoryReader(
        input_files=["./release_notes.md"]
).load_data()

print("Document ID:", documents[0].doc_id)

from llama_index.core.node_parser import SentenceWindowNodeParser

# Create the sentence window node parser
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)

print(len(nodes))

# Extract the raw text from the nodes.
docs = []
for node in nodes:
    docs.append(node.text)

print(docs[0])
print(len(docs))


# Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

dense_dim=3072

# Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=512, enable_analyzer=True)
schema.add_field(field_name="sparse_bm25", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)

bm25_function = Function(
        name="bm25",
        function_type=FunctionType.BM25,
        input_field_names=["text"],
        output_field_names="sparse_bm25",
    )
schema.add_function(bm25_function)

# Prepare index parameters
index_params = client.prepare_index_params()

# Add indexes
index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)

index_params.add_index(
    field_name="sparse_bm25",
    index_name="sparse_bm25_index",
    index_type="SPARSE_WAND",
    metric_type="BM25"
)

collection_name = "hybrid_search_bm25_collection"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

# create collection
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)
