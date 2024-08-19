import streamlit as st
import boto3
from utils import opensearch, secret
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import BedrockChat
from loguru import logger
import sys
import os

# logger
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client

def create_opensearch_vector_search_client(index_name, username, opensearch_password, bedrock_embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(username, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch

def create_bedrock_llm(bedrock_client, model_version_id):
    bedrock_llm = BedrockChat(
        model_id=model_version_id,
        client=bedrock_client,
        model_kwargs={'temperature': 0}
        )
    return bedrock_llm

def initialize_retrieval_chain(index_name, region, bedrock_model_id, bedrock_embedding_model_id, username):
    bedrock_client = get_bedrock_client(region)
    bedrock_llm = create_bedrock_llm(bedrock_client, bedrock_model_id)
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
    opensearch_endpoint = opensearch.get_opensearch_endpoint(index_name, region)
    opensearch_password = secret.get_secret(index_name, region)
    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, username, opensearch_password, bedrock_embeddings_client, opensearch_endpoint)

    prompt = ChatPromptTemplate.from_template("""If the context is not relevant, please answer the question by using your own knowledge about the topic. If you don't know the answer, just say that you don't know, don't try to make up an answer. don't include harmful content

    {context}

    Question: {input}
    Answer:""")

    docs_chain = create_stuff_documents_chain(bedrock_llm, prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=opensearch_vector_search_client.as_retriever(),
        combine_docs_chain = docs_chain
    )
    return retrieval_chain

def main():
    st.title("RAG Application")

    question = st.text_input("Ask a Question:", "Can we combine life and car insurance?")

    if st.button("Submit"):
        logger.info(f"Question provided: {question}")

        # Parameters
        region = 'us-east-1'
        index_name = 'rag'
        bedrock_model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        bedrock_embedding_model_id = 'amazon.titan-embed-text-v1'
        username = 'osmaster'

        # Initialize retrieval chain
        retrieval_chain = initialize_retrieval_chain(index_name, region, bedrock_model_id, bedrock_embedding_model_id, username)

        # Get response
        response = retrieval_chain.invoke({"input": question})
        answer = response.get('answer')

        st.subheader("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()