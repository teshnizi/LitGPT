from flask import Flask
from api.search import get_relevant_papers, generate_literature
from flask import render_template
from flask import jsonify
from flask import request
from flask import current_app
import numpy as np
import openai

# def init_app_context(app):
    # with app.app_context():
    #     app.paper_ids = np.load('api/data/paper_ids.npy', allow_pickle=True).item()
    #     app.emb_matrix = np.load('api/data/emb_matrix.npy', allow_pickle=True)
    
app = Flask(__name__)
# init_app_context(app)
        
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def about():
    # query_title = "Learning to Solve Combinatorial Optimization Problems on Real-World Graphs in Linear Time"
    # query_abstract = "Combinatorial optimization algorithms for graph problems are usually designed afresh for each new problem with careful attention by an expert to the problem structure. In this work, we develop a new framework to solve any combinatorial optimization problem over graphs that can be formulated as a single player game defined by states, actions, and rewards, including minimum spanning tree, shortest paths, traveling salesman problem, and vehicle routing problem, without expert knowledge. Our method trains a graph neural network using reinforcement learning on an unlabeled training set of graphs. The trained network then outputs approximate solutions to new graph instances in linear running time. In contrast, previous approximation algorithms or heuristics tailored to NP-hard problems on graphs generally have at least quadratic running time. We demonstrate the applicability of our approach on both polynomial and NP-hard problems with optimality gaps close to 1, and show that our method is able to generalize well: (i) from training on small graphs to testing on large graphs; (ii) from training on random graphs of one type to testing on random graphs of another type; and (iii) from training on random graphs to running on real world graphs."

    data = request.get_json()
    query = data['input']
    key = data['key']
    model = data['model']   
    
    print(model)
    openai.api_key = key
    
    query = query.split('\n')
    if len(query) <= 1:
        return jsonify({'generated_text': "Please enter both title and abstract in the given format!", 'citations': []})
    else:
        query_title = query[0]
        query_abstract = ' '.join(query[1:])
    
    print(query_title, '----',  query_abstract, key) 

    num_papers = 15
    if model == "gpt-4":
        num_papers = 25
        
    try:
        id_list = get_relevant_papers(query_title, query_abstract, key, emb_matrix=current_app.emb_matrix, num_papers=num_papers)
    except Exception as e:
        return jsonify({'generated_text': "One of the inputs is invalid! Is your API key correct?", 'citations': []})
    
    lit, citations = generate_literature(query_title, query_abstract, id_list, key, paper_ids=current_app.paper_ids, model=model)
    return jsonify({'generated_text': lit, 'citations': citations})
    

if __name__ == '__main__':
    app.run(debug=True)
    