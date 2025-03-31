# Wiki-Chatbot
Creación de un chatbot en Streamlit utilizando el framework Langchain y el dataset de Wikimedia.

# 📌 Dataset utilizado
Se utilizó el dataset de Wikimedia en español, específicamente la versión [20231101](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.es). Para ejecutar el código que genera la base de datos vectorial, es necesario que estos archivos estén en el mismo directorio que el código.

# 📌 Ejecución del archivo carga_datos.ipynb
En este archivo se encuentra el procedimiento utilizado para generar los datos en la base de datos vectorial FAISS. Si se desea descargar directamente el archivo FAISS, se encuentra en este [enlace](https://drive.google.com/file/d/1DEGFLW81RfkzR_Uiizu5hxaSj0CCDpKD/view?usp=sharing), el cual es necesario descomprimit.

## TextSplitter
Para la segmentación del texto, se utilizó RecursiveCharacterTextSplitter de LangChain, ya que permite dividir documentos largos en fragmentos más pequeños, manteniendo la coherencia del contenido.

## Embedding
Para la conversión de texto a vectores se utilizó el modelo sentence-transformers de Hugging Face. Esto permite representar documentos y consultas en un espacio vectorial, facilitando la búsqueda de información relevante. Una de las principales razones para utilizar este embedding es su uso gratuito.

## Generación del faiss.index
Debido a la cantidad de datos, en esta ocasión no fue posible guardar todos los datos del dataset en el archivo de FAISS. Se utilizó un método de carga por lotes para realizar los embeddings y guardarlos en la base de datos vectorial.

# 📌 Ejecución del chatbot
Para la ejecución del chatbot, es necesario ejecutar ` streamlit run chatbot.py ` y tener configurada la GOOGLE_API_KEY y la ruta del archivo faiss.index en las variables de entorno (archivo .env)

## Cadena utilizada
Los pasos para generar una respuesta son los siguientes:

1. Se define el modelo de embedding utilizado
2. Se carga el archivo faiss.index
3. Se carga el modelo de lenguaje a utilizar, en este caso, gemini-2.0 utilizando la API
4. Se crea el retriever a partir del archivo de base de datos vectorial
5. Se crea el prompt para contextualizar las preguntas. Esto es útil si la pregunta del usuario es ambigua y depende del contexto previo.
6. Se crea un retriever mejorado con contexto, que reformula la pregunta usando el historial antes de buscar en FAISS.
7. Se define el prompt principal que el LLM usará para generar respuestas.
8. Se crea una cadena que toma el contexto recuperado y lo pasa al LLM junto con la pregunta del usuario.
9. Se combina el retriever mejorado y la cadena de generación de respuestas en una sola estructura.
10. A partir del prompt y el historial de mensajes, se llama a esta cadena final con el método ` invoke `

# 📌 Funcionalidades
1. El chatbot tiene un historial de conversación utilizando el método que se menciono antes.
2. Se pueden tener varios chats.

# 📌 Ejemplos

## Contexto de la conversación

![Wiki Chatbot recuerda el nombre](/memoria.jpeg)

## Ejemplo de preguntas que se pueden hacer al chatbot
![Pregunta sobre agujeros negros](/pregunta1.jpeg)

![Pregunta sobre Pablo Neruda](/pregunta2.jpeg)
   


