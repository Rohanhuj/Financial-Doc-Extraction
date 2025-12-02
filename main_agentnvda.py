from setup_config import *
from ingest_vectorize import *
from rag_tools import *
from agent_graph import *
#INPUT_DATAFILE_PATH = "/home/roroa/extraction-poc/input/NVIDIAAn.pdf"

if __name__ == "__main__":
    file_path = input(f"Enter document path <default={INPUT_DATAFILE_PATH}>: ")
    # FIXME : Validate the file path entered
    if not file_path:
        file_path = INPUT_DATAFILE_PATH

    ingestion = input("Would you like to run ingestion again? [Y/N] <default=N>: ")
    print(f"Processing {file_path}")

    if ingestion and ingestion.upper() in ['Y','YES']:
        # Optional TODO : Turn off recreation once vectorstore is populated
        #db, vectorstore, collection_name = run_ingestion(file_path)
        vectorstore, collection_name = run_ingestion(file_path, recreate=True)
    else:
        print(f"Skipping ingestion for {file_path}")
        vectorstore, collection_name = load_vectorstore(os.path.basename(file_path))

    if file_path == INPUT_DATAFILE_PATH: # Use default for running some unit-tests
        pass
    # Build the graph
    graph_workflow = build_graph()
    save_graph_to_png(graph_workflow, "./extraction_agent_graph.png")

    if file_path == INPUT_DATAFILE_PATH: # Use known eval_questions for default doc's basic evaluation
        # TODO : Enhance this query for the sageview document that we are working with
        run_state = run_graph(graph_workflow, "Extract the following information for the quarterly financial results of this company - company name, Revenue, CEO, which quarter, Gaming and AI PC Revenue highlights")
        print("\n" + "="*80 + "\n")
        with open(OUTPUT_JSON_FILE_PATH,'w') as json_f:
            json.dump(run_state['final_response'], json_f, indent=2)
    else:
        # TODO : Come up with some other query based on attributes that need extraction
        run_state = run_graph(graph_workflow, "Extract the following information for the quarterly financial results of this company - company name, Revenue, CEO, which quarter, Gaming and AI PC Revenue highlights")
        print("\n" + "="*80 + "\n")
        with open(OUTPUT_JSON_FILE_PATH,'w') as json_f:
            json.dump(run_state['final_response'], json_f, indent=2)

