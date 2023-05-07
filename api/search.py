import openai
import os
import numpy as np
from langchain.embeddings import OpenAIEmbeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

folder = 'api/database/'
files = os.listdir(folder)
files = [f for f in files if f.endswith('.npy')]

files = sorted(files)
print(files)


db = {}
global_idx = 0

# if not os.path.exists('api/database/papers_all.npy'):
for f in files:
    print(f"Loading file {f}...")
    papers = np.load(folder + f, allow_pickle=True).item()
    papers = {global_idx + idx: papers[idx] for idx in papers}
    db.update(papers)
    global_idx += len(papers)

assert len(global_idx)>0, "No papers found in the database! Please run get_data.py first"
# np.save('api/database/papers_all.npy', db)

# else:
#     db = np.load('api/database/papers_all.npy', allow_pickle=True).item()   

# db = np.load('api/papers_all.npy', allow_pickle=True).item()   

# print(f"Database size: {os.path.getsize('api/papers_all.npy') / 1024 / 1024} MB")

emb_matrix = np.zeros((len(db), 1536))
for idx in db:
    emb_matrix[idx] = db[idx]['embedding']
    
print(emb_matrix.shape) # (24000, 1536)

# query_title = "Learning to Solve Combinatorial Optimization Problems on Real-World Graphs in Linear Time"
# query_abstract = "Combinatorial optimization algorithms for graph problems are usually designed afresh for each new problem with careful attention by an expert to the problem structure. In this work, we develop a new framework to solve any combinatorial optimization problem over graphs that can be formulated as a single player game defined by states, actions, and rewards, including minimum spanning tree, shortest paths, traveling salesman problem, and vehicle routing problem, without expert knowledge. Our method trains a graph neural network using reinforcement learning on an unlabeled training set of graphs. The trained network then outputs approximate solutions to new graph instances in linear running time. In contrast, previous approximation algorithms or heuristics tailored to NP-hard problems on graphs generally have at least quadratic running time. We demonstrate the applicability of our approach on both polynomial and NP-hard problems with optimality gaps close to 1, and show that our method is able to generalize well: (i) from training on small graphs to testing on large graphs; (ii) from training on random graphs of one type to testing on random graphs of another type; and (iii) from training on random graphs to running on real world graphs."

def get_relevant_papers(query_title, query_abstract, key, num_papers=15):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
    query = f"Title: {query_title}\n ===== Abstract: {query_abstract}\n\n"
    emb = np.array(embeddings.embed_query(query)).reshape(1, -1) # (1, 1536_)
    id_list = np.argsort(-np.dot(emb, emb_matrix.T), axis=1)[0][:num_papers]
    # create a dict with titles, authors, year, and link:
    return id_list

def generate_literature(query_title, query_abstract, id_list, key, model="gpt-3.5-turbo"):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=key)
    citations = []
    
    context = ""
    for id, p in enumerate(id_list):
        
        db[p]['abstract'] = db[p]['abstract'].replace('\n', ' ')
        if model != 'gpt-4':
            if len(db[p]['abstract']) > 1500:
                db[p]['abstract'] = db[p]['abstract'][:1500] + " ..."
        
        context += "====="
        context += f"ID: {id+1}\n"
        context += f"-Title: {db[p]['title']}\n-Abstract: {db[p]['abstract']}\n\n"
        
        citations.append({'id': str(id+1),
                          'title': db[p]['title'], 
                          'authors': [x.name for x in db[p]['authors']], 
                          'date': str(db[p]['date'].date()), 
                          'link': db[p]['link']})
        
    context += "=====\n"

    paper = f"-Title: {query_title}\n-Abstract: {query_abstract}\n\n"
    
    
    prompt = "I want you to write the related work section of a paper, using a list of related papers. You can only use these papers listed. Use this format \"apples are red [1,7]\" where numbers are paper IDs."
    prompt += "Here is the main paper:\n\n"
    prompt += "=====\n" + paper + "\n=====\n"
    prompt += " Here are the papers:\n\n"
    prompt += context
    prompt += "Just generate the related work section, not the references themselves."


    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "user", "content": prompt},
            ]
    )
    
    
    res = response["choices"][0]["message"]["content"]
    
    return res, citations
