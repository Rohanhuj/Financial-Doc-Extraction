from setup_config import *
from ingest_vectorize import *

query_optimizer_llm = ChatOllama(model="mistral-small3.2", temperature=0)
#cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
cross_encoder_model = CrossEncoder('BAAI/bge-reranker-base')
TOP_K = 40
sql_agent_llm = ChatOllama(model="mistral-small3.2", temperature=0)


def optimize_query(query: str) -> str:
    """Uses an LLM to rewrite a query for better retrieval."""
    
    prompt = f"""
    You are a query optimization expert. Rewrite the following user query to be more specific and effective for searching through corporate financial documents (e.g. quarterly financial results).

    User Query: {query}

    Optimized Query:"
    """
    optimized_query = query_optimizer_llm.invoke(prompt).content
    return optimized_query


@tool
def librarian_rag_tool(query: str, file_path=INPUT_DATAFILE_PATH, vectorstore_client=None, collection_name=None) -> List[Dict[str, Any]]:
    """ Finding and retrieving information from a vector database """
    print(f"\n-- Librarian Tool Called with query: '{query}' --")

    # Optimize/rewrite Query
    optimized_query = optimize_query(query)
    print(f"  - Optimized query: '{optimized_query}'")

    # Vector Search (Initial Retrieval)
    if not vectorstore_client:
        vectorstore_client, collection_name = load_vectorstore(os.path.basename(file_path))
    #query_embedding = list(embedding_model.embed([optimized_query]))[0]
    #search_results = vectorstore_client.search(
        #collection_name=collection_name,
        #query_vector=query_embedding,
        #limit=50, # Get more results initially for the re-ranker
        #with_payload=True
    #)
    query_dense_vector = list(embedding_model.embed([optimized_query]))[0].tolist()
    # FIXME : Some local versions of qdrant have problems dealing with hybrid searches... Even though I
    # hacked thru it, hybrid search doesn't really give me better results. Still had to include 50 results
    # for it to be useful. Other versions e.g. remote version, and/or other vector store versions that
    # support hybrid search should be explored here.
    """
    dense_results = vectorstore_client.search(
        collection_name=collection_name,
        query_vector={
            "name": "dense-text",
            "vector": query_dense_vector
        },
        limit=50,
        with_payload=True
    )

    # Run sparse search
    sparse_embedding_obj = list(sparse_model.embed([optimized_query]))
    sparse_results = vectorstore_client.search(
        collection_name=collection_name,
        query_vector={
            "name": "sparse-text",
            "vector": models.SparseVector(
                indices=sparse_embedding_obj.indices.tolist(),
                values=sparse_embedding_obj.values.tolist()
            )
        },
        limit=50,
        with_payload=True
    )

    print(f"Dense search retrieved {len(dense_results)} results.")
    print(f"Sparse search retrieved {len(sparse_results)} results.")
    search_results = list(dense_results) + list(sparse_results)

    """
    try:
        sparse_embedding_obj = list(sparse_model.embed(optimized_query))[0]
        query_sparse_vector_instance = models.SparseVector(
            indices=sparse_embedding_obj.indices,
            values=sparse_embedding_obj.values
        )
    except IndexError:
        print("Warning: Sparse model returned no embeddings. Proceeding with dense search only.")
        query_sparse_vector_instance = None

    prefetch_list = []
    if query_sparse_vector_instance:
        prefetch_list.append(models.Prefetch(
            query=query_sparse_vector_instance,
            using="sparse-text",
            limit=50
        ))
    prefetch_list.append(models.Prefetch(
        query=query_dense_vector,
        using="dense-text",
        limit=50
    ))

    search_results_generator = vectorstore_client.query_points(
        collection_name=collection_name,
        #prefetch=[
            #models.Prefetch(
                #query=models.SparseVector(**sparse_embedding_obj),
                #using="sparse-text",
                #limit=20
            #),
            #models.Prefetch(
                #query=query_dense_vector,
                #using="dense-text",
                #limit=20
            #)
        #],
        prefetch=prefetch_list,
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        #query=models.FusionQuery(fusion=models.Fusion.DBSF, prefetches=prefetch_list),
        limit=50,
        with_payload=True
    )
    search_results = list(search_results_generator)
    #for result in search_results:
        #print(f"  - Retrieved {result} candidate chunk from vector store.")
    if len(search_results) == 1:
        print(f"  - Type of results is {type(search_results[0])}")
        scored_points = search_results[0][1]
        rerank_pairs = [[optimized_query, scored_point.payload['content']] for scored_point in scored_points]
        scores = cross_encoder_model.predict(rerank_pairs)
        for i, score in enumerate(scores):
            scored_points[i].score = score
        reranked_results = sorted(scored_points, key=lambda x: x.score, reverse=True)
    else:
        rerank_pairs = [[optimized_query, result.payload['content']] for result in search_results]
        scores = cross_encoder_model.predict(rerank_pairs)
        for i, score in enumerate(scores):
            search_results[i].score = score
        reranked_results = sorted(search_results, key=lambda x: x.score, reverse=True)
    print("  - Re-ranked the results using Cross-Encoder.")

    # Format and Return Top Results
    final_results = []
    for result in reranked_results[:TOP_K]:
        final_results.append({
            #'source': result.payload['source'],
            'content': result.payload['content'],
            'summary': result.payload['summary'],
            'keywords': result.payload['keywords'],
            #'rerank_score': float(result.score)
        })

    print(f"  - Returning top {TOP_K} re-ranked chunks.")
    return final_results

tools = [librarian_rag_tool]
tool_map = {tool.name: tool for tool in tools}

print()
for tool in tools:
    print(f"- Tool: {tool.name}")
    print(f"  Description: {tool.description.strip()}\n")


