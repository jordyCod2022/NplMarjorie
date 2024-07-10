import re
import spacy
import unicodedata
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

nlp = spacy.load("es_core_news_sm")

def preprocess_text(text):
    if isinstance(text, str):  # Verificar si es una cadena de texto
        # Convertir el texto a minúsculas
        text = text.lower()
        # Eliminar tildes
        text = ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))
        # Eliminar caracteres especiales
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Aplicar el modelo de spaCy al texto de entrada
        doc = nlp(text)
        # Eliminar stopwords
        tokens = [token.text for token in doc if not token.is_stop and token.text.isalpha()]
        # Reconstruir el texto sin stopwords
        text = ' '.join(tokens)
    elif isinstance(text, dict):  # Verificar si es un diccionario
        if 'msg' in text:
            text = text['msg']
            # Podemos aplicar el mismo procesamiento que hacemos para las cadenas de texto
            text = preprocess_text(text)
    return text


# Leer el archivo CSV en un DataFrame
df = pd.read_csv('backend-avi/Servidor/api/train.csv', sep=';', encoding='utf-8')


# Acceder a los textos y etiquetas
textos = df['Textos'].tolist()
etiquetas = df['Etiquetas'].tolist()

# Verificar los datos cargados
#print("Textos preprocesados:", textos)
#print("Etiquetas:", etiquetas)


# Preprocesamiento de los textos
textos_preprocesados = [preprocess_text(texto) for texto in textos]

# Codificación de etiquetas
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(etiquetas)

# Entrenamiento del clasificador SVM con TF-IDF
svm_classifier = Pipeline([
    #Parametros = min_df=2, max_df=0.5, ngram_range=(1, 2)
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear'))
])

svm_classifier.fit(textos_preprocesados, y_encoded)

# Función para predecir y obtener la mejor respuesta
def predecir_y_responder(texto):
    texto_preprocesado = preprocess_text(texto)
    predicciones = svm_classifier.predict([texto_preprocesado])
    mejor_etiqueta = encoder.inverse_transform(predicciones)[0]
    datos={'Mejor etiqueta':mejor_etiqueta,'texto desp. preproc':texto_preprocesado,'corpus desp. preproc':textos_preprocesados}
    """ print("Texto del usuario (después del preprocesamiento):", texto_preprocesado)
    print("Texto del corpus (después del preprocesamiento):", textos_preprocesados) """
    return datos






















# Ejemplo de uso
""" texto_a_clasificar = "me gusta el deporte y la programación, ya que son muy interesantes"
texto_usuario_preprocesado = preprocess_text(texto_a_clasificar)
mejor_respuesta = predecir_y_responder(texto_a_clasificar)

# Imprimir texto del usuario y texto del corpus después del preprocesamiento
print("Texto del usuario (después del preprocesamiento):", texto_usuario_preprocesado)
print("Texto del corpus (después del preprocesamiento):", textos_preprocesados)
print("La mejor etiqueta clasificada para el texto es:", mejor_respuesta)
 """



""" # Crear un archivo CSV para almacenar los textos preprocesados y las etiquetas
df = pd.DataFrame({'Textos': textos, 'Etiquetas': etiquetas})

# Guardar el DataFrame en un archivo CSV
df.to_csv('train.csv', index=False, sep=';', encoding='utf-8')
 """