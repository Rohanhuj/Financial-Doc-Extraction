from setup_config import *

# ### Intelligent, Structure-Aware Chunking
# 
# We'll use Pydantic to define the desired JSON structure, ensuring the LLM's output is reliable.

class ChunkMetadata(BaseModel):
    """Structured metadata for a document chunk."""
    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: List[str] = Field(description="A list of 5-7 key topics or entities mentioned.")
    hypothetical_questions: List[str] = Field(description="A list of 3-5 questions this chunk could answer.")
    table_summary: Optional[str] = Field(description="If the chunk is a table, a natural language summary of its key insights.", default=None)

# Ideally use gpt-4o-mini here
#enrichment_llm = ChatOllama(model="gpt-oss:20b", temperature=0).with_structured_output(ChunkMetadata)
enrichment_llm = ChatOllama(model="llama3.1:8b", temperature=0).with_structured_output(ChunkMetadata)

embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
qdrant_instance = qdrant_client.QdrantClient(path=QDRANT_PATH)

def parse_pdf_file(file_path: str) -> List[Dict]:
    """Parses an PDF file using unstructured and returns a list of elements."""
    try:
        #elements = partition_html(filename=file_path, infer_table_structure=True, strategy='fast')
        elements = partition(filename=file_path)
        print(f"Found {len(elements)} elements")
        return [el.to_dict() for el in elements]
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

def get_chunk_samples(parsed_elements):
    # Convert the dictionary elements back to unstructured Element objects for chunking

    elements_for_chunking = elements_from_dicts(parsed_elements)

    # Chunk the elements using the chunk_by_title strategy
    chunks = chunk_by_title(
        elements_for_chunking,
        max_characters=2048,      # Max size of a chunk
        combine_text_under_n_chars=256, # Combine small text elements
        new_after_n_chars=1800  # Start a new chunk if the current one is getting too big
    )

    print(f"Document chunked into {len(chunks)} sections.")

    print("\n--- Sample Chunks ---")

    # Find and print a sample text chunk and a sample table chunk
    text_chunk_sample = None
    table_chunk_sample = None

    for chunk in chunks:
        if 'text_as_html' not in chunk.metadata.to_dict() and text_chunk_sample is None and len(chunk.text) > 500:
            text_chunk_sample = chunk
        if 'text_as_html' in chunk.metadata.to_dict() and table_chunk_sample is None:
            table_chunk_sample = chunk
        if text_chunk_sample and table_chunk_sample:
            break

    if text_chunk_sample:
        print("** Sample Text Chunk **")
        print(f"Content: {text_chunk_sample.text[:500]}...")
        print(f"Metadata: {text_chunk_sample.metadata.to_dict()}")

    if table_chunk_sample:
        print("\n** Sample Table Chunk **")
        # For tables, the HTML representation is often more useful
        print(f"HTML Content: {table_chunk_sample.metadata.text_as_html[:500]}...")
        print(f"Metadata: {table_chunk_sample.metadata.to_dict()}")

    return chunks, text_chunk_sample, table_chunk_sample


def get_enrichment_prompt(chunk_text: str, is_table: bool) -> str:
    """Generates a prompt for the LLM to enrich a chunk."""

    table_instruction = """
    This chunk is a TABLE. The summary should describe the type of information presented, as well as the key data points mentioned in the content.
    """ if is_table else ""

    prompt = f"""
    You are an informatione extraction expert who understands financial information. Analyze the following document chunk and generate the specified metadata fields. The summary metadata field should cover the type of information presented in the content, as well as key highlights and data points mentioned. The keywords metadata field should contain the main entities and topics covered. The hypothetical questions field should be the type of questions that could potentially be answered by the content.
    {table_instruction}
    Chunk Content:
    ---
    {chunk_text}
    ---
    """
    return prompt

def enrich_chunk(chunk) -> Dict[str, Any]:
    """Enriches a single chunk with LLM-generated metadata."""
    is_table = 'text_as_html' in chunk.metadata.to_dict()
    content = chunk.metadata.text_as_html if is_table else chunk.text

    # To avoid overwhelming the LLM, we'll truncate very long chunks
    truncated_content = content[:3000]

    prompt = get_enrichment_prompt(truncated_content, is_table)
    #print(f"Prompt for table={is_table} enrichment_llm is {prompt}")

    try:
        metadata_obj = enrichment_llm.invoke(prompt)
        return metadata_obj.dict()
    except Exception as e:
        #print(f"  - Error enriching chunk: {e}")
        raise e
        #return None

def test_enrich_samples(text_chunk_sample, table_chunk_sample):
    print("--- Testing Enrichment on a Text Chunk ---")
    try:
        if text_chunk_sample:
            enriched_text_meta = enrich_chunk(text_chunk_sample)
            print(json.dumps(enriched_text_meta, indent=2))

        print("\n--- Testing Enrichment on a Table Chunk ---")
        if table_chunk_sample:
            enriched_table_meta = enrich_chunk(table_chunk_sample)
            print(json.dumps(enriched_table_meta, indent=2))
    except Exception as e:
        print(f"Error enriching chunk: {e}")

    # The output should show two JSON objects, one for each chunk type.
    # - For the text chunk, we have a clear summary, relevant keywords, and insightful hypothetical questions.
    # - For the table chunk, the LLM has correctly identified it as a table 
    # and provided a `table_summary` that interprets the data in natural language. 

def enrich_chunks(all_files):
    if os.path.exists(ENRICHED_CHUNKS_PATH):
        print("Loading pre-existing enriched chunks file.")
        with open(ENRICHED_CHUNKS_PATH, 'r') as f:
            all_enriched_chunks = json.load(f)
    else:
        all_enriched_chunks = []
        total_files = len(all_files)
        # Use tqdm from tqdm.notebook to be compatible with Colab
        with tqdm(total=total_files, desc="Processing Files") as pbar_files:
            for i, file_path in enumerate(all_files):
                pbar_files.set_postfix_str(os.path.basename(file_path))
                parsed_elements_dicts = parse_pdf_file(file_path)
                if not parsed_elements_dicts:
                    pbar_files.update(1)
                    continue
                #elements_for_chunking = [element_from_dict(el) for el in parsed_elements_dicts]
                elements_for_chunking = elements_from_dicts(parsed_elements_dicts)
                doc_chunks = chunk_by_title(elements_for_chunking, max_characters=2048, combine_text_under_n_chars=256)
                with tqdm(total=len(doc_chunks), desc=f"Enriching Chunks", leave=False) as pbar_chunks:
                    for chunk in doc_chunks:
                        try:
                            enrichment_data = enrich_chunk(chunk)
                            if enrichment_data:
                                is_table = 'text_as_html' in chunk.metadata.to_dict()
                                content = chunk.metadata.text_as_html if is_table else chunk.text
                                final_chunk_data = {
                                    'source': f"{os.path.basename(os.path.dirname(os.path.dirname(file_path)))}/{os.path.basename(os.path.dirname(file_path))}",
                                    'content': content,
                                    'is_table': is_table,
                                    **enrichment_data
                                }
                                all_enriched_chunks.append(final_chunk_data)
                        except Exception as e:
                            tqdm.write(f"Exception in enrich_chunk: {e}")
                        pbar_chunks.update(1)
                pbar_files.update(1)
        print(f"\n\nCompleted processing. Total enriched chunks: {len(all_enriched_chunks)}")
        with open(ENRICHED_CHUNKS_PATH, 'w') as f:
            json.dump(all_enriched_chunks, f)
        print(f"Enriched chunks saved to '{ENRICHED_CHUNKS_PATH}'.")


def create_embedding_text(chunk: Dict) -> str:
    """Creates a combined text for embedding from an enriched chunk."""
    # The text to be embedded will include the summary and keywords for better retrieval
    return f"""
    Summary: {chunk['summary']}
    Keywords: {', '.join(chunk['keywords'])}
    Content: {chunk['content'][:1000]} 
    """

def get_qdrant_client():
    return qdrant_instance

def get_collection_name(file_name):
    return f"{file_name.upper()}_{COLLECTION_NAME_PREFIX}"

def load_vectorstore(file_name):
    # Set up the Qdrant client
    #client = qdrant_client.QdrantClient(":memory:")
    client = get_qdrant_client()
    collection_name = get_collection_name(file_name)

    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Qdrant collection '{collection_name}' loaded with {collection_info.points_count} points.")
    return client, collection_name


def populate_vectorstore(file_name, recreate=False):
    with open('enriched_chunks.json', 'r') as f:
        all_enriched_chunks = json.load(f)
    print(f"Loaded {len(all_enriched_chunks)} enriched chunks from file.")

    client = get_qdrant_client()
    collection_name = get_collection_name(file_name)

    if recreate or not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense-text": qdrant_client.http.models.VectorParams(
                    size=embedding_model.get_embedding_size(model_name="BAAI/bge-small-en-v1.5"),
                    distance=qdrant_client.http.models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse-text": models.SparseVectorParams()
            }
        )

    print(f"Qdrant collection '{collection_name}' created.")

    points_to_upsert = []
    texts_to_embed = []

    for i, chunk in enumerate(all_enriched_chunks):
        text_to_embed = create_embedding_text(chunk)
        texts_to_embed.append(text_to_embed)

        #dense_embedding = list(embedding_model.embed([text_to_embed]))[0]
        #sparse_embedding = list(sparse_model.embed([text_to_embed]))[0].as_object()
        #points_to_upsert.append(qdrant_client.http.models.PointStruct(
            #id=i,
            #vector={
                #"dense-text": dense_embedding.tolist(),
                #"sparse-text": qdrant_client.http.models.SparseVector(**sparse_embedding)
            #},
            #payload=chunk # The payload contains all the rich metadata
        #))
        dense_embedding = list(embedding_model.embed([text_to_embed]))[0]

        #sparse_embedding = sparse_model.embed(text_to_embed)
        sparse_embedding = list(sparse_model.embed([text_to_embed]))[0]

        #if isinstance(sparse_embedding, list) and len(sparse_embedding) > 0:
            #sparse_embedding_obj = sparse_embedding[0]
        #else:
            #sparse_embedding_obj = None

        # Create the point struct with named vectors
        points_to_upsert.append(models.PointStruct(
            id=i,
            vector={
                "dense-text": dense_embedding.tolist(),
                #"sparse-text": sparse_embedding
                "sparse-text": models.SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist()
                )
            },
            payload=chunk
        ))

    print(f"Prepared {len(points_to_upsert)} points for upsert.")

    print("Upserting into Qdrant...")

    batch_size = 128
    for i in range(0, len(points_to_upsert), batch_size):
        batch_points = points_to_upsert[i: i+batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch_points,
            wait=True
        )
        print(f"Upserted batch {i // batch_size + 1}")

    print("\nUpsert complete!")
    collection_info = client.get_collection(collection_name=collection_name)
    print(f"Points in collection: {collection_info.points_count}")
    return client, collection_name


def run_ingestion(file_path=INPUT_DATAFILE_PATH, recreate=False):
    # TODO: Check that file exists

    parsed_elements = parse_pdf_file(file_path)

    print(ChunkMetadata.schema_json(indent=2))

    chunks, text_chunk_sample, table_chunk_sample = get_chunk_samples(parsed_elements)
    test_enrich_samples(text_chunk_sample, table_chunk_sample)

    enrich_chunks(all_files=[file_path])

    vectorstore, collection_name = populate_vectorstore(os.path.basename(file_path), recreate)

    # Note : recreate csv and db since this is quick
    return vectorstore, collection_name
    
if __name__ == "__main__":
    run_ingestion(INPUT_DATAFILE_PATH)
