import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Acceder a las variables de entorno
ruta = os.getenv("RUTA_INDEX")
api_key = os.getenv("GOOGLE_API_KEY")

# Función para manejar el historial de los mensajes. Cada sesión tiene su historial.
def get_session_history(session_id: str = ''):
    if session_id not in st.session_state['histories']:
        st.session_state['histories'][session_id] = []

    # Obtener el historial de la sesión
    return st.session_state['histories'][session_id]

#Funcion para cambiar de conversación
def switch_conversation(session_id: str):
    st.session_state['current_session'] = session_id
    st.rerun()

#Titulo de la pagina
st.title('Wiki Chatbot')

#Si no hay sesión, se crea una por defecto
if 'current_session' not in st.session_state:
    #Conversacion por defecto
    st.session_state['current_session'] = "Chat principal"

session_id = st.session_state["current_session"]

#Bool para mostrar o no mostrar el input para nueva conversacion
if 'show_input' not in st.session_state:
    st.session_state['show_input'] = False

#Se inicializan los historiales de conversación
if 'histories' not in st.session_state:
    st.session_state['histories'] = {}

#Si la sesión no tiene historial, se crea uno vacío.
if session_id not in st.session_state['histories']:
    st.session_state['histories'][session_id] = []

#Sidebar para las conversaciones
with st.sidebar:
    st.title("Chats")
    
    if st.button("Nuevo chat"):
        st.session_state['show_input']=True
    
    if st.session_state['show_input']==True:
        new_session_id = st.text_input("Ingresa un nombre para el chat:")

        if new_session_id and new_session_id not in st.session_state['histories']:
            #Inicializar el chat
            st.session_state['histories'][new_session_id] = []
            st.session_state['show_input'] = False
            switch_conversation(new_session_id)

    #Se muestran los chats existentes
    for session in st.session_state['histories']:
        if st.button(session, key=session, type='tertiary'):
            switch_conversation(session)

#Se crea la cadena
if 'qa_chain' not in st.session_state:
    # Definir el modelo de embedding utilizado
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Cargar el índice FAISS desde el archivo
    vector_db = FAISS.load_local(ruta, embeddings=embed_model, allow_dangerous_deserialization=True)  #
    
    # Cargar el modelo de lenguaje
    llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key=api_key)

    # Crear el retriever
    retriever = vector_db.as_retriever()

    # Crear el prompt para contextualizar preguntas
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dado un historial de conversación y una pregunta del usuario, "
                "formula una pregunta independiente que se pueda entender sin el historial. "
                "NO respondas la pregunta, solo reformúlala si es necesario."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Crear el retriever que usa el historial
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextualize_q_prompt
    )

    # Crear el prompt para responder preguntas
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente de preguntas y respuestas llamado Javi. Usa el contexto recuperado para responder. Si no sabes la respuesta, solo di 'No lo sé'. {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Pregunta: {input}"),
    ])

    # Crear la cadena de respuesta
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Crear la cadena de recuperación y respuesta final
    qa_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    # Definir el qa chain utilizado para Streamlit
    st.session_state['qa_chain'] = qa_chain


# Obtener el historial para esta sesión
history = st.session_state['histories'][session_id]

# Mostrar mensajes previos en la interfaz
for message in history:
    # `message` es un diccionario con claves 'role' y 'content'
    if isinstance(message, HumanMessage):
        role = "user"
        with st.chat_message(role):
            st.markdown(message.content)
    else:
        role = "assistant"
        with st.chat_message(role):
            st.markdown(message)


qa_chain = st.session_state['qa_chain']

#Crear la interfaz
if prompt := st.chat_input("¡Pregunta lo que quieras!"):

    #Se muestra el mensaje del usuario
    with st.chat_message('user'):
        st.markdown(prompt)

    #Obtener respuesta del modelo RAG
    with st.chat_message('assistant'):
        # Obtener la respuesta
        with st.spinner("Esperando respuesta..."):

            answer = qa_chain.invoke({"chat_history": st.session_state['histories'][session_id], "input": prompt})   

        #Se muestra el mensaje del asistente
        response = st.markdown(answer['answer'])
    
    #Se agrega el prompt y respuesta al historial de conversación para la sesión
    st.session_state['histories'][session_id].extend([HumanMessage(content=prompt), answer["answer"]])

