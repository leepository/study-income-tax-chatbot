import os
import time

from dotenv import load_dotenv
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

def run_bot():
    load_dotenv()

    # 문서 spliter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    # 문서의 내용을 읽고 splitter로 쪼갠다
    loader = Docx2txtLoader('./tax_with_markdown.docx')
    document_list = loader.load_and_split(text_splitter=text_splitter)
    print("document length : ", len(document_list))

    # embedding : 사람이 사용하는 자연어를 기계가 이해할 수 있는 숫자 형태의 Vector로 변경하는 과정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=pinecone_api_key)

    query = "연봉 5천만원인 거주자의 소득세는 얼마인가요?"

    # 처음 벡터 데이터베이스에 데이터를 넣어야 할 때 사용
    # database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)

    # 기존에 입력된 데이터를 사용하는 경우
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    # retriever: 지정된 벡터 데이터베이스에서 유사도를 측정하여 유사성이 높은 데이터를 추출하는 함수
    retriever = database.as_retriever(search_kwargs={'k': 4})
    retriever.invoke(query)

    llm = ChatOpenAI(model='gpt-4o')

    prompt = hub.pull('rlm/rag-prompt')
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # ai_message = qa_chain.invoke({'query': query})

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    dictionary_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단되면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}

        질문: {{question}}
    """)
    dictionary_chain = dictionary_prompt | llm | StrOutputParser()

    tax_chain = {"query": dictionary_chain} | qa_chain

    ai_message = tax_chain.invoke({"question": query})

    print(ai_message['result'])
