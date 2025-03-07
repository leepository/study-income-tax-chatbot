# [Study] Income tax chatbot

소득세 관련 질문에 답변을 하는 간단한 chatbot.
 - Chatbot 구동에 필요한 파일 다음과 같다
	 - chat.py : streamlit으로 구성된 web client 모듈
	 - llm.py:  사용자 입력 쿼리를 이용한 AI 답변 생성 모듈
 - 나머지 파일들은 AI Study 참고용으로 생성해 둔 것.

## Requirements

소득세 챗봇 구성을 위해서는 다음과 같은 사항이 필요하다. (환경변수로 설정)
 - OPENAI_API_KEY
 - PINECONE_API_KEY
 - LANGSMITH_TRACING
 - LANGSMITH_ENDPOINT
 - LANGSMITH_API_KEY
 - LANGSMITH_PROJECT

소득세 학습을 위한 Markup된 문서: Repository에 포함되어 있음.
 

## Packages

게시판 구현에 사용된 python version은 3.11.9이며 python package는 다음과 같다.

     
	chromadb==0.6.3    
	docx2txt==0.8    
	langchain==0.3.20  
	langchain-chroma==0.2.2  
	langchain-community==0.3.18  
	langchain-core==0.3.41  
	langchain-openai==0.3.7  
	langchain-pinecone==0.2.3  
	langchain-tests==0.3.13  
	langchain-text-splitters==0.3.6  
	langchainhub==0.1.21  
	langsmith==0.3.11   
	numpy==1.26.4   
	openai==1.65.2   
	pandas==2.2.3  
	pinecone==5.4.2    
	requests==2.32.3  
	streamlit==1.43.0  

## Run application

Application은 다음과 같이 구동한다. 

    $ source .venv/bin/activate
    $(.venv) streamlit run chat.py

## Evaluation
Income tax chatbot의 사용자 질문에 대한 답변의 정확성은 다음과 같은 방법으로 평가한다.
	
	$ source .venv/bin/activate
	$(.venv) python evaluator.py

평가 결과는 langsmith의 Experiments에 정리된다. 

