import torch
from Config import allcreds

data=[]


# Get Creds from config.py

es_client = allcreds["es"]
groq_client = allcreds["client"]
colbert_creds=allcreds["colbert"]
tokenizer_creds=allcreds["tokenizer"]



class Chat_api:

    def __init__(self):
        pass
    
    def gpt_query(self,Question:str,File_name:str): 
        try:                                                  # Pass Question and filename as input query                                  

            tokens = tokenizer_creds(Question, padding=True, truncation=True, return_tensors="pt")        # Get embeddings for the question using colbert model    
            with torch.no_grad():
                embeddings = colbert_creds(**tokens).last_hidden_state.mean(dim=1).numpy()[0]

            query = es_client.search( index="chatbot_index", body={                                       # Query to fetch first 3 document from elastic search index 
                    "_source": [
                        "docstring"],
                        'size':3, 
                    'query': { 'script_score': { 'query': {'match_all': {}                                # Match all documents in the index
                            }, 
                    'script': { 'source': "cosineSimilarity(params.query_vector, 'embeddings') + 1.0",    # Script to calculate the cosine similarity score
                            'params': {'query_vector': embeddings                                      # Pass the query vector (embeddings) as a parameter to the script
                                } 
                            }
                        }
                    }
                } )
        
            hits = query['hits']['hits']
            for hit in hits:
                data.append(hit)                                                                          # append top 3 document into list pass to llm 

            chat_completion = groq_client.chat.completions.create(                                        # calling llama -3 model (groq cloud ai) passing user question and elastic document to get response.
                messages=[
                    {
                        "role": "system",
                       "content": f"""You are a helpful assistant who answers users' questions based on the given data and also suggests some questions for the next search based on the user's interest as separate fields called suggestions. Strictly return the response in the following JSON format:
                            {{
                                "answer": "<your_answer_here>",
                                "suggestions": [
                                    "<suggestion_1>",
                                    "<suggestion_2>",
                                    "<suggestion_3>"
                                ]
                            }}
                        Data:{data}
        """
                    },
                    {
                        "role": "user",
                        "content": f'{str(Question)}',
                    },
                ],
                model="llama3-8b-8192",
            )

            filename=f"citation:{File_name}"
            print(chat_completion.choices)
            response = chat_completion.choices[0].message.content
            # suggestions = chat_completion.choices[0].message.content['suggestions']  # assuming suggestions are included in metadata

            
            return response,filename 
        
        except Exception as e:
            raise e
    
Chat_bot=Chat_api()

