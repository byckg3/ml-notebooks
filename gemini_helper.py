import os
import time
from google import genai
from google.genai import types
from chromadb import Documents, EmbeddingFunction, Embeddings

# https://ai.google.dev/gemini-api/docs/embeddings
class GenAIEmbeddingFunction( EmbeddingFunction[ Documents ] ):

    def __init__( self, api_key: str = None, 
                  model_name: str = "gemini-embedding-exp-03-07", 
                  task_type = "SEMANTIC_SIMILARITY" ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv( "GEMINI_API_KEY" )
        self.client = genai.Client( api_key = self.api_key )
        self.model_name = model_name
        self.task_type = task_type

    def __call__( self, input: Documents ) -> Embeddings:
       
        result = self.client.models.embed_content( model = self.model_name,
                                                   contents = input,
                                                   config = types.EmbedContentConfig( task_type = self.task_type )
                                    )
       
        return [ embedding.values for embedding in result.embeddings ]
    

def to_embeddings( contents, n = -1, type = "SEMANTIC_SIMILARITY" ):
    embedding_function = GenAIEmbeddingFunction( task_type = type )

    current = 0
    offset = 5
    total = len( contents ) if n == -1 else n
    embeddings = []
    while len( embeddings ) < total:
        
        try:
            result = embedding_function( contents[ current:( current + offset ) ] )
            if isinstance( result, list ):
                embeddings.extend( result )

            if current % 10 == 0:
                print( f"current progress: { current }" )
            current = current + offset
            
        except Exception as e:
            print( f"Error: { e }" )
        
        time.sleep( 3 )

    print( "len:", len( embeddings ) )
    print( "last index:", current )

    assert len( embeddings ) == total

    return embeddings