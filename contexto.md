Voy a desarrollar proyecto, cuyas directrices son:
Proyecto de la asignatura
Directrices

Idioma • El proyecto debe desarrollarse en castellano o euskera. • Todo el corpus, análisis y resultados deben estar en uno de estos idiomas.
Recopilación del corpus • Es obligatorio recopilar un corpus adecuado al objetivo del proyecto. • El corpus debe ser amplio y representativo del problema que se va a abordar. • Al menos una parte del corpus debe obtenerse mediante técnicas de web scraping vistas en clase.
Etiquetado (opcional) • Si el proyecto incluye un componente de anotación o etiquetado: o Se deben definir pautas claras y detalladas para la anotación, que se describan en la memoria del proyecto. o Se debe calcular el acuerdo entre anotadores (inter-annotator agreement), proporcionando métricas que respalden su consistencia.
Análisis exploratorio de datos (EDA) • Realizar un EDA del corpus con el objetivo de comprender su estructura, características y posibles problemas antes de aplicar cualquier modelo. • El EDA debe incluir visualizaciones, estadísticas descriptivas y cualquier análisis que ayude a entender los datos.
Preprocesamiento • Todas las operaciones de preprocesamiento de texto realizadas deben estar justificadas, explicando cómo se relacionan con los objetivos del proyecto.
Identificación de sesgos • Analizar posibles sesgos en el corpus, por ejemplo: género, edad, ubicación geográfica, clase social, etc. • Si se identifican sesgos, proponer medidas para mitigarlos y documentarlas en el proyecto.
Métricas de evaluación • Revisar la literatura existente para seleccionar las métricas de evaluación más adecuadas para el proyecto. MUCSI – Curso 2025-2026 Minería de texto y procesamiento del lenguaje natural • Justificar la elección de una métrica estándar o, si es necesario, desarrollar una métrica propia adaptada al proyecto. • Evaluar cómo las métricas seleccionadas impactan en los resultados y asegurarse de que estén alineadas con los objetivos del proyecto.
Proyectos conjuntos con Deep Learning (DL) • Si el proyecto combina PLN con DL: o Las tareas relacionadas con PLN y DL deben documentarse y separarse claramente en la memoria y en la presentación. o Cada parte debe incluir objetivos, métodos y resultados específicos. ¿dónde encontrar fuentes para obtener ideas? • Presentación del primer día de clase. • Github • Kaggle • HuggingFace • Towards Data Science • Medium • Papers: arXiv, ACL Antology, Google Scholar, etc. MUCSI – Curso 2025-2026 Minería de texto y procesamiento del lenguaje natural Hitos
Hito 0 – Propuesta inicial o Fecha: 27 de febrero de 2026 o Confirmación de los integrantes del grupo o Completar plantilla y entregarla en ALUD.
Hito 1 - Entrega intermedia (EC) o Fecha: 26 de marzo de 2026 o Breve documento explicativo (de 2 a 5 páginas) o Contenido ▪ Breve estado del arte sobre el tema a resolver ▪ Listado de corpus y recursos identificados. ▪ Arquitecturas u otros recursos identificados. ▪ Grado de avance en el desarrollo de las tareas ▪ Cambios en la propuesta original.
Hito 2 - Entrega o Fecha: por confirmar, a coordinar con DL. Tentativa: 17 de mayo de 2026 o Contenido: ▪ Memoria: ● Introducción ● Breve estado del arte de la problemática planteada. ● Corpus y recopilación de datos ● EDA ● Identificación de sesgos ● Metodología ➢ Preprocesamiento ➢ Arquitectura ➢ Métricas ● Resultados ● Discusión ● Referencias ▪ Si el proyecto es conjunto con DL, la memoria debe tener dos partes diferenciadas. ▪ Código: ● No olvidéis aportar los archivos más grandes (y que no se puedan adjuntar en la tarea) a través de enlaces a Drive usando para ello la función de comentarios de ALUD. o Además, es necesario realizar una presentación para su defensa la última semana de clase. ▪ Si el proyecto es conjunto con DL, la presentación debe tener dos partes diferenciadas
He pensado en realizar un RAG con recetas webscrappeadas de blogs que asista en la cocina con consejos de recetas. Puesto que tengo una cuenta de google colab pro plus, he pensado en desarrollar el proyecto desde visual studio conectando mi kernel a colab para aprovehcar las tarjetas gráficas disponibles. Los apartados inciales de webscrapping, EDA, preocesamiento, etc. no me preocupan tanto ya que los he realizado en actividades anteriores. Me gustaría disctuir la implementació del RAG y su evaluación, crees que esto sería sencillo desde el set up que te he descrito, o hay alternativas mejores? Debería de escribir el código en archivos .py o en cuaderno .ipynb? Es posible implementar una interfaz con Streamlit por ejemplo? Que modelos recomiendas para el RAG?


Ingeteam Minimal Icon
Tu idea del RAG con recetas web‑scrapeadas encaja bastante bien con los requisitos del proyecto: corpus propio, scraping, EDA, sesgos (tipo cocina regional, género de autores, etc.), y evaluación de un sistema de PLN. No es una mala elección en absoluto. El punto crítico será cómo planteas la evaluación, porque un asistente tipo RAG conversacional puede ser difícil de medir si no defines bien la tarea.

Voy por partes.

1. ¿Tu setup (VS Code + Colab GPU) es razonable?
Sí, es perfectamente viable.

Arquitectura típica que puedes usar:

VS Code como entorno principal.
Kernel remoto de Colab para ejecutar entrenamiento/embeddings.
Vector DB local (FAISS o Chroma).
Modelo LLM vía:
API (OpenAI, Mistral, etc.)
o local ligero.
Lo único a tener en cuenta:

Colab no es ideal para servicios persistentes (vector DB corriendo mucho tiempo).
Lo normal es:
generar embeddings en Colab
guardar FAISS/Chroma
luego usarlo localmente.
Flujo práctico:

Scraping + limpieza → local
EDA → local
Generación de embeddings → Colab GPU
Crear vector index (FAISS)
Pipeline RAG
Evaluación
2. .py vs .ipynb
Recomendación práctica para este tipo de proyecto:

Usa ambos.

Notebook (.ipynb)
Para:

Web scraping exploratorio
EDA
pruebas de embeddings
experimentos
Te sirve para generar gráficas para la memoria.

Scripts .py
Para el sistema final:

pipeline RAG
evaluación
interfaz
Estructura razonable:

css
project/ │ ├─ data/ │ └─ recetas_raw.json │ ├─ notebooks/ │ ├─ scraping.ipynb │ ├─ eda.ipynb │ └─ embedding_tests.ipynb │ ├─ src/ │ ├─ scrape.py │ ├─ preprocess.py │ ├─ embed.py │ ├─ rag_pipeline.py │ └─ evaluation.py │ ├─ vector_db/ │ └─ faiss_index │ └─ app/ └─ streamlit_app.py
Esto queda mucho más profesional para entregar.

3. ¿Streamlit es viable?
Sí, y de hecho queda muy bien para la defensa.

Una interfaz simple podría tener:

caja de texto: "¿Qué quieres cocinar?"
resultados:
receta sugerida
pasos
consejos
Ejemplo de queries:

"Tengo pollo y arroz, ¿qué puedo cocinar?"
"Cómo evitar que el arroz quede pastoso"
"Recetas rápidas con garbanzos"
Arquitectura:

java

Usuario ↓ Streamlit UI ↓ RAG pipeline ↓ Vector DB (recetas) ↓ LLM ↓ Respuesta + fuentes

Además puedes mostrar:

recetas recuperadas
score de similitud
Eso queda muy bien para explicar retrieval + generation.

4. Modelos recomendados para el RAG
Para un proyecto académico no necesitas algo enorme.

Embeddings (muy importante)
Opciones buenas:

BAAI/bge-small-en-v1.5
intfloat/e5-base-v2
sentence-transformers/all-MiniLM-L6-v2
Si el corpus está en español, mejor:

intfloat/multilingual-e5-base
bge-m3
Recomendación equilibrada:

multilingual-e5-base

Vector DB
Las más simples:

FAISS → muy común en papers
ChromaDB → muy fácil de usar
Para un proyecto:

FAISS suele verse más "académico".

LLM para generación
Opciones:

Si usas API (muy fácil)
GPT‑4o mini
GPT‑4.1 mini
Si quieres modelo abierto
Mistral 7B Instruct
Llama 3 8B Instruct
Qwen2.5 7B Instruct
Para Colab GPU:

Llama‑3‑8B‑Instruct o Qwen2.5‑7B funcionan bien.

5. Pipeline RAG recomendado
Pipeline típico:

css
1 Scraping recetas 2 Limpieza 3 Chunking 4 Embeddings 5 Vector DB 6 Query 7 Retrieval (top-k) 8 Prompt + contexto 9 LLM genera respuesta
Ejemplo de chunk:

json
{ "titulo": "Tortilla de patatas", "ingredientes": "...", "pasos": "...", "tips": "...", "origen": "blog X" }
Chunk size típico:

text
200–400 tokens
6. La parte más importante: Evaluación
Aquí es donde muchos proyectos RAG fallan.

Puedes evaluarlo de varias formas.

1️⃣ Evaluación del retrieval
Métricas:

Recall@k
Precision@k
MRR
Ejemplo:

Query:

text
receta con pollo y limón
Si la receta correcta aparece en top‑5 → cuenta.

2️⃣ Evaluación del sistema completo
Opciones:

BLEU / ROUGE
Comparar con respuestas de referencia.

LLM-as-judge
Muy usado en papers recientes.

Prompt:

css
Evalúa si la respuesta es útil para cocinar la receta. Score 1–5
3️⃣ Evaluación humana
Puedes pedir a 3 personas que evalúen:

utilidad
claridad
precisión
Y calcular:

Inter‑annotator agreement (Cohen's Kappa)

Esto encaja perfectamente con las directrices del curso.

7. Posibles sesgos (muy importante para tu memoria)
Ejemplos que puedes analizar:

cocina dominante (española vs latinoamericana)
recetas más masculinas/femeninas (blogs)
ingredientes caros vs baratos
recetas saludables vs calóricas
Puedes analizar:

frecuencia ingredientes
origen de blogs
dificultad recetas
Esto cumple muy bien el apartado bias analysis.

8. Ideas para mejorar el proyecto (si quieres que destaque)
Una mejora simple pero potente:

RAG + filtros estructurados
Ejemplo:

makefile
Usuario: "Recetas vegetarianas con menos de 30 minutos"
Filtras por metadata antes de retrieval.

O RAG con "cooking tips"
Separar:

recetas
consejos culinarios
y recuperar ambos.

✅ Conclusión honesta:

Tu idea es muy viable y académicamente sólida si enfocas bien:

corpus
retrieval evaluation
sesgos
y no solo el chatbot.

Si quieres, también puedo enseñarte algo que muchos estudiantes hacen mal en proyectos RAG:

cómo definir una tarea evaluable clara, que es lo que más miran los profesores.


<img class