
import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# --- Configuración de la página ---
st.set_page_config(
    page_title="Pregúntale a tus PDFs",
    page_icon="❓",
    layout="wide"
)

st.title("Chatea con tus Documentos 💬")
st.markdown("""
Sube tus archivos PDF, haz una pregunta y obtén respuestas basadas en su contenido.
""")

# --- Constantes y configuración ---
PDFS_DIR = "temp_docs"
VECTORSTORE_DIR = "vector_store"
if not os.path.exists(PDFS_DIR):
    os.makedirs(PDFS_DIR)

# --- Lógica de la aplicación ---

@st.cache_resource
def get_embeddings_model():
    """Carga el modelo de embeddings una sola vez."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdfs(pdf_files_paths):
    """
    Procesa los PDFs subidos: extrae texto, lo divide y crea una base de datos de vectores.
    """
    all_docs = []
    for path in pdf_files_paths:
        try:
            loader = PyPDFLoader(path)
            documents = loader.load()
            all_docs.extend(documents)
        except Exception as e:
            st.error(f"Error al leer el archivo {os.path.basename(path)}: {e}")
            return None

    if not all_docs:
        st.warning("No se pudo extraer texto de los PDFs.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    embeddings = get_embeddings_model()
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    return vectorstore

def get_answer(question, vectorstore):
    """
    Obtiene una respuesta a la pregunta del usuario usando la base de datos de vectores.
    """
    if not st.session_state.get('groq_api_key'):
        st.error("Por favor, introduce tu API Key de Groq para continuar.")
        return None

    llm = ChatGroq(temperature=0, groq_api_key=st.session_state.groq_api_key, model_name="llama3-8b-8192")
    
    retriever = vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    response = qa_chain.invoke(question)
    return response['result']

# --- Interfaz de Usuario ---

with st.sidebar:
    st.header("Configuración")
    groq_api_key = st.text_input("Introduce tu API Key de Groq", type="password")
    if groq_api_key:
        st.session_state.groq_api_key = groq_api_key

    st.header("1. Sube tus PDFs")
    uploaded_files = st.file_uploader(
        "Arrastra y suelta tus PDFs aquí",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Procesar Documentos"):
        if uploaded_files:
            with st.spinner("Analizando documentos..."):
                saved_files_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(PDFS_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files_paths.append(file_path)

                st.session_state.db = process_pdfs(saved_files_paths)
                
                if st.session_state.db:
                    st.success("¡Documentos procesados con éxito!")
                    st.balloons()
        else:
            st.warning("Por favor, sube al menos un archivo PDF.")

st.header("2. Haz una pregunta")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre los documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            if "db" in st.session_state and st.session_state.db:
                response = get_answer(prompt, st.session_state.db)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "Por favor, primero procesa los documentos en el panel de la izquierda."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
