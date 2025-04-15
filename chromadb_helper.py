import chromadb

CHROMADB_COLLECTION_NAME = "gad245-g1-chromadb-embedding"

def add_dataframe( df, function, collection_name = CHROMADB_COLLECTION_NAME ):

    chroma_client = chromadb.PersistentClient( path = "./chroma" )
    collection = chroma_client.get_or_create_collection( 
                                        name = collection_name,
                                        embedding_function = function, #  Chroma will use sentence transformer as a default. 
                               )
    
    n = df.shape[ 0 ]
    print( "n =", n )

    embeddings = df[ "embedding" ].tolist()[ :n ]
    documents = df[ "document" ].tolist()[ :n ]
    ids = df[ "id" ].tolist()[ :n ]
    metadatas = df.drop( columns = [ "id", "document", "embedding" ] ).to_dict( orient = "records" )[  :n ]


    collection.add(
        documents = documents,
        embeddings = embeddings,
        ids = ids,
        metadatas = metadatas,
    )
    
    return chroma_client, collection

def delete_collection( client, collection_name = CHROMADB_COLLECTION_NAME ):
    client.delete_collection( collection_name )