# Grid Testing on Truporch RAG w/ Advanced Features
Attached are the latest results of a configuration grid test that I had to run on OpenWebUI's gemma3:27b model, testing the LLM at different model parameters to optimize answer quality metrics.

Much like in the previous repository, I used a Python RAG script, this time with advanced context retrieval strategies, and imported the RAG into a grid test script with improved answer quality measurements.
I used the embeddings from the previous scrape I conducted, so the scraper script used here is the same as before. ('notion_scraper_final.py')

## Updated features
To optimize the RAG for answer quality, I put several methods to the test:

- Meta Tensor Protection System: fixed PyTorch meta tensor issues using aggressive environment variable settings, forced CUDA initialization, and fallback mechanisms
- Hybrid Retrieval Architecture: keyword-based retrieval with TF-IDF-like scoring and inverted indexing
- Cross-encoder Reranking: used a separate cross-encoder model (ms-marco-MiniLM-L-6-v2) to score query-document pairs more precisely
- Enhanced Query Processing: uses real estate domain synonyms to expand queries with related terms
- Dynamic Quality Thresholding: adjusts minimum similarity requirements based on query type and score distribution
- Context Organization: relevance-based passage extraction identifies the most relevant sections within retrieved documents
- Quality-based caching: only caches high-quality responses
- Intelligent fallback and error handling/ retry mechanisms for failed queries

To test the updated RAG for answer quality as rigously as I could, my grid test script measures: 

- Similarity variance: how well retrieved documents match the query, lower variance = better
- Technical Content Analysis: counts domain-specific terms (cap rate, NOI, cash flow, etc.)
- Numerical Data Density: presence of percentages, dollar amounts, ratios
- Information Density: technical terms + numbers per 100 words
- Concrete Examples

I ran five real-estate related queries (computation, evaluation, comparison based queries) through twelve LLM configurations: 

- Temperature/Top-P Matrix (Configs 1-9): Tests 3 temperature levels (0.1, 0.2, 0.4) × 3 top-p values (0.8, 0.9, 0.95)
- High Creativity (Config 10): Temperature 0.6, top-p 0.95 for more creative responses
- Maximum Precision (Config 11): Temperature 0.05, top-p 0.7 for factual accuracy
- Extended Context (Config 12): Larger context window (4096 tokens) for comprehensive answers

Lastly, my grid test includes broken pipe recovery, memory management, and a retry mechanism that restarts up to three times per failed query request. 

##Recommendations

The configuration where temperature = 0.1 and top-p = 0.9 (all other parameters kept to default) returned the best results, with all queries returning an advanced quality score of 0.8. 
The default temperature and top-p parameters were 0.2 and 0.85, respectively. 

While a successful test, the testing schema employed in the grid test was admittedly convoluted, and tested too many unimportant performance metrics without directly capturing relevance and accuracy scores.
In subsequent trials, I'll have to tinker with the testing script once more to include more robust human evaluation metrics. Also, expanding the configuration matrix to include mirostat could've
also improved test scores. 
