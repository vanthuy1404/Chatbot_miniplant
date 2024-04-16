import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from pyvi import ViPosTagger
import requests
import json

# Khởi tạo cho query search api
GOOGLE_SEARCH_KEY = "AIzaSyCR9GdJQN3Ss9po0YW2WEZBfUGOqiyHqhM"
SEARCH_ENGINE_ID = "0746a5016259e495b"
num_results = 5
urls = []


# Hàm tìm kiếm urls
def search_google(query, api_key, cse_id, num=5):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cse_id}&num={num}"
    response = requests.get(url)
    results = json.loads(response.text)
    items = results['items']
    count = 0
    urls = []
    if not items:
        return None
    for item in items:
        link = item['link']
        if "facebook.com" not in link and "youtube.com" not in link and ".pdf" not in link:
            urls.append(link)
            count = count + 1
        if count == 2:
            break
    return urls


# Hàm lấy keyword từ câu hỏi
def get_keyword(question):
    tokens_with_pos = ViPosTagger.postagging(question)
    # keyword mặc định cho lĩnh vực cây cảnh mini
    keyword = ['cây', 'cảnh', "mini", 'xanh', 'trồng', 'chăm sóc', 'lá', 'thân', 'rễ', 'cành', 'chậu', 'đất']
    for token, pos_tag in zip(tokens_with_pos[0], tokens_with_pos[1]):
        if pos_tag.startswith('N') or pos_tag.startswith('V'):
            keyword.append(token.lower())
    return keyword


# Hàm clean data
def clean_data(texts, keywords):
    clean_data1 = ''
    clean_data2 = ''
    for p in texts:
        sentences = p.split('\n')
        new_texts = []
        for sentence in sentences:
            if sentence.strip() != '':
                new_text = sentence.strip() + '.'
                new_texts.append(new_text)
        clean_data1 += ' '.join(new_texts)
        new_sentences = clean_data1.split('.')
        for new_sentence in new_sentences:
            for keyword in keywords:
                if keyword.lower() in new_sentence.lower():
                    clean_data2 += new_sentence.strip() + '. '
                    break  # Thoát khỏi vòng lặp từ khóa khi tìm thấy từ khóa trong câu
    return clean_data2


st.title("Chatbot cây cảnh mini")
# Tạo lịch sử chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Hiển thị nội dung chat:
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

if question := st.chat_input("Hãy hỏi tôi điều gì đó"):

    with st.chat_message('user'):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    #Lấy urls
    urls = search_google(question, GOOGLE_SEARCH_KEY, SEARCH_ENGINE_ID, num_results)
    # Lay tu khóa
    keywords = get_keyword(question)
    # load data từ website
    loader = WebBaseLoader(urls)
    data = loader.load()
    # Xử lí data + chia thành các chunk nhỏ để embedding
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=10000,
                                          chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    context = "\n\n".join(str(p.page_content) for p in docs)
    texts = text_splitter.split_text(context)
    context = "".join(clean_data(texts, keywords))
    clean_data_final = text_splitter.split_text(context)
    # Tạo model
    google_api_key = "AIzaSyAizcUR_wSwPiPMzZ7Lnbd24AxSF0zOMPM"
    model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                                   temperature=0.4, convert_system_message_to_human=True)

    # embedding và lưu vào Chromadb
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_index = Chroma.from_texts(clean_data_final, embeddings).as_retriever(search_kwargs={"k": 3})

    # Tạo prompt template để truy vấn
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.Answer in detail and completely as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)  # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    result = qa_chain({"query": question})

    with st.chat_message('assistant'):
        response = result['result']
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

