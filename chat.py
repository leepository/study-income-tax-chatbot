import streamlit as st
from llm import get_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("🤖 소득세 챗봇")
st.caption("소득세에 관한 모든것을 답변해 드립니다.")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []
# streamlit은 매번 입력값이 전송될때마다 페이지를 새로고침한다.
# session_state는 session 기준으로 상태를 저장하는 역할을 한다.
# session_state에 message_list 항목을 넣어놓고 입력값을 저장해 둔다면 이전 입력값에 대해 History를 유지할 수 있다.
# 페이지 새로고침이 일어나면 message_list에서 입력했던 질문들의 목록을 불러와 화면에 출력한다.
# 매번 새로고침할때마다 그리는 것이 아니라 streamlit이 react 기반이고 react에서 DOM 변경 사항을 확인하여 변경 사항만
# 추가하는 방식으로 동작한다.
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요."):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner("답변을 생성중입니다."):
        ai_response = get_ai_response(user_message=user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({'role': 'ai', 'content': ai_message})