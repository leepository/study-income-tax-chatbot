from dotenv import load_dotenv
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def run_bot():
    load_dotenv()

    llm = ChatOpenAI()

    ai_message = llm.invoke("인프런에는 어떤 강의가 있나요?")

    print(ai_message.content)

def run_rag():
    # 1. 문서의 내용을 읽는다.
    # 2. 문서를 쪼갠디 -> 토큰수 초과로 답변을 생성하지 못할 수 있고 문서가 길면 (인풋이 길면) 답변 생성이 오래걸림
    # 3. 임베딩 -> 벡터 데이터베이스에 저장
    # 4. 질문이 있을 때, 벡터 데이터베이스에 유사도 검색
    # 5. 유사도 검색으로 가져온 문서를 LLM에 질문과 같이 전달

    # 환경 변수를 load 한다.
    load_dotenv()

    # 문서 splitter 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # 하나의 chunk가 가지는 token수
        chunk_overlap=200
    )

    # 문서의 내용을 읽고 splitter로 쪼갠다.
    loader = Docx2txtLoader('./tax.docx')
    document_list = loader.load_and_split(text_splitter=text_splitter)

    # 임베딩 -> 벡터 데이터베이스에 저장
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    database = Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name='chroma-tax',
        persist_directory='./chroma'
    )

    # 질문
    query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
    # 벡터 데이터베이스에서 유사도 검색
    retrieved_docs = database.similarity_search(query, k=3)
    # 유사도 검색으로 가져온 문서를 LLM에 질문과 같이 전달
    llm = ChatOpenAI(model='gpt-4o')

    prompt = f"""[Identity]
    - 당신은 한국 최고의 소득세 전문가입니다.
    - [Context]를 참고해서 질문에 답변해 주세
    
    [Context]
    {retrieved_docs}
    
    Question: {query}
    """

    ai_message = llm.invoke(prompt)

    print(ai_message.content)

def run_retrievalqa():
    load_dotenv()
    llm = ChatOpenAI()

    # 기존에 문서를 벡터 데이터베이스에 저장했다면 아래와 같이 사용한다.
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    database = Chroma(
        embedding_function=embeddings,
        collection_name='chroma-tax',
        persist_directory='./chroma'
    )

    prompt = hub.pull('rlm/rag-prompt')

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=database.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'
    ai_message = qa_chain.invoke({"query": query})

    print(ai_message)

def run_without_langchain():
    import os
    import tiktoken
    import chromadb
    from docx import Document
    from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
    from openai import OpenAI

    def split_document(full_text, chunk_size):
        # LLM 모델에 따라 처리 가능한 token 수의 한계가 존재한다.
        # tiktoken을 통한 문서 전체 토큰수 계산값이 model의 토큰 한계치를 넘을 수 있기 때문에
        # 우리가 지정한 chunk_size 별로 별도로 그룹핑하여 그룹핑된 text list를 반환한다.
        encoder = tiktoken.encoding_for_model('gpt-4o')
        encoding = encoder.encode(full_text)
        total_token_count = len(encoding)
        text_list = []
        for i in range(0, total_token_count, chunk_size):
            chunk = encoding[i: i+chunk_size]
            decoded = encoder.decode(chunk)
            text_list.append(decoded)
        return text_list

    document = Document('./tax.docx')
    # Document는 문단 단위로 읽어들이므로 전체 텍스트를 별도로 만든다.
    full_text = ''
    for paragraph in document.paragraphs:
        full_text += f'{paragraph.text}\n'

    # 문서를 쪼갠다. -> 문서의 토큰수를 계산한다.
    chunk_list = split_document(full_text=full_text, chunk_size=1500)

    # 임베딩
    chroma_client = chromadb.Client()
    collection_name = 'tax_collection'

    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_embedding = OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name='text-embedding-3-large'
    )
    tax_collection = chroma_client.get_or_create_collection(collection_name, embedding_function=openai_embedding)

    id_list = []
    for index in range(len(chunk_list)):
        id_list.append(f'{index}')

    tax_collection.add(documents=chunk_list, ids=id_list)

    # Query -> 유사도 검색
    query = "연봉 5천만원 직장인의 소득세는 얼마인가요?"
    retrieved_doc = tax_collection.query(query_texts=query)

    # LLM 질의
    openai_client = OpenAI()
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {'role': 'system', 'content': f'당신은 한국의 소득세 전문가입니다. 아래 내용을 참고하여 질문에 답변하세요 {retrieved_doc["documents"][0]}'},
            {'role': 'user', 'content': query}
        ]
    )

    print("response : ", response.choices[0].message.content)
