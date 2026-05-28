import RAG_QA
from RAG_QA import load_documents

docs = load_documents('pdf_data')
print('docs', len(docs))
