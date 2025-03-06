from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate, \
    FewShotChatMessagePromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import answer_examples


load_dotenv()

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm():
    llm = ChatOpenAI(model='gpt-4o')
    return llm

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    # 사용자의 질문 History를 참고하여 입력된 사용자의 질문을 reformaulate 하라는 system prompt
    # ref: python.langchain.com
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user questions "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def get_dictionary_chain():
    llm = get_llm()

    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    dictionary_prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}

        질문: {{question}}
    """)
    dictionary_chain = dictionary_prompt | llm | StrOutputParser()
    return dictionary_chain

def get_rag_chain():
    llm = get_llm()

    # 답변의 정확도를 높이기 위해 few shot을 사용한다.
    # few shot은 예상되는 질문에 대한 답변 형식을 미리 지정하여 학습시키는 방법이다.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}")
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples
    )


    # QA에 대한 system prompt
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise."
    #     "\n\n"
    #     "{context}"
    # )

    # 한글로 작성하는 system prompt -> 영어로 작성하면 token을 조금 아낄 수 있다.
    # 하지만 한글로 작성해도 충분히 동작한다.
    system_prompt = (
        "당신은 소득세볍 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요. "
        "아래에 제공된 문서를 활용해서 답변해주시고 "
        "답변을 할 수 없다면 모른다고 답변해 주세요. "
        "답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해 주시고"
        "2-3 문장 정도의 짧은 내용의 답변을 원합니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_rag_chain

def get_ai_response(user_message: str):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_rag_chain()

    tax_chain = {"input": dictionary_chain} | qa_chain
    ai_response = tax_chain.stream(
        {"question": user_message},
        config={
            "configurable": {"session_id": "chatbot"}
        }
    )

    return ai_response



