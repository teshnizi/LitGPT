
# 🔥 LitGPT

The "Literature Review" part of academic papers can take a while to write, often misses some related sources, and doesn't keep up with new publications. LitGPT is a proof of concept showing how AI can change the way we do literature reviews. Simply provide your paper's title and abstract to see the results!

- 📚 Right now LitGPT only covers ML and AI papers submitted on arxiv from Jan 2011 to May 2023 (281035 papers). 🗓️
- 🚀 The output quality is better with GPT-4 if your API_Key has access to it. 🔑💸
- 📝 LitGPT only uses titles and abstracts. The output quality is significantly better with the whole paper text, but the data must be extracted from PDF files and will take up much more space and memory. 💾
- 🧢 The number of papers is capped to 15 for ChatGPT and 25 for GPT-4.

## Running Locally

- To get the paper data (might take a while, I can't upload the data because of github size limits)
```bash
python api/get_data.py
```
- To run the app
```bash
export FLASK_APP=api/index.py
flask run
```
