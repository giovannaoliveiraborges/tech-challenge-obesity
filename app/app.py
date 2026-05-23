import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title='Sistema Preditivo de Obesidade',
    page_icon='🩺',
    layout='wide'
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Obesity.csv')


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def train_model(df):
    df_model = df.copy()

    # Criação da variável de IMC
    df_model['BMI'] = df_model['Weight'] / (df_model['Height'] ** 2)

    X = df_model.drop(columns=['Obesity'])
    y = df_model['Obesity']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return pipeline, accuracy


df = load_data()
model, accuracy = train_model(df)

if 'BMI' not in df.columns:
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)


st.title('🩺 Sistema Preditivo de Nível de Obesidade')

st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para estimar o nível de obesidade de uma pessoa com base em informações demográficas, alimentares, comportamentais e de estilo de vida.

A solução tem como objetivo apoiar a equipe médica na identificação de perfis de risco e na tomada de decisão clínica.
""")

st.info(f'Acurácia do modelo no conjunto de teste: {accuracy:.2%}')

tab1, tab2, tab3 = st.tabs([
    'Sistema Preditivo',
    'Dashboard Analítico',
    'Sobre o Projeto'
])


with tab1:
    st.header('Previsão individual')

    st.sidebar.header('Informações do paciente')

    gender = st.sidebar.selectbox('Gênero', ['Female', 'Male'])
    age = st.sidebar.slider('Idade', 14, 61, 25)
    height = st.sidebar.number_input('Altura em metros', min_value=1.40, max_value=2.10, value=1.70, step=0.01)
    weight = st.sidebar.number_input('Peso em kg', min_value=35.0, max_value=180.0, value=70.0, step=0.5)

    family_history = st.sidebar.selectbox('Histórico familiar de excesso de peso?', ['yes', 'no'])
    favc = st.sidebar.selectbox('Consome alimentos calóricos com frequência?', ['yes', 'no'])

    fcvc = st.sidebar.slider('Frequência de consumo de vegetais', 1.0, 3.0, 2.0, 1.0)
    ncp = st.sidebar.slider('Número de refeições principais por dia', 1.0, 4.0, 3.0, 1.0)

    caec = st.sidebar.selectbox('Come entre as refeições?', ['no', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.sidebar.selectbox('Fuma?', ['yes', 'no'])

    ch2o = st.sidebar.slider('Consumo diário de água', 1.0, 3.0, 2.0, 1.0)
    scc = st.sidebar.selectbox('Monitora calorias diariamente?', ['yes', 'no'])

    faf = st.sidebar.slider('Frequência de atividade física semanal', 0.0, 3.0, 1.0, 1.0)
    tue = st.sidebar.slider('Tempo usando dispositivos eletrônicos', 0.0, 2.0, 1.0, 1.0)

    calc = st.sidebar.selectbox('Consumo de álcool', ['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.sidebar.selectbox(
        'Meio de transporte habitual',
        ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']
    )

    bmi = weight / (height ** 2)

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history': [family_history],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans],
        'BMI': [bmi]
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric('Idade', age)

    with col2:
        st.metric('IMC estimado', round(bmi, 2))

    with col3:
        st.metric('Peso', f'{weight} kg')

    st.subheader('Dados informados')
    st.dataframe(input_data)

    if st.button('Gerar previsão'):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data).max()

        st.success(f'Nível previsto: {prediction}')
        st.info(f'Confiança estimada do modelo: {prediction_proba:.2%}')

        st.warning('Este resultado deve ser utilizado como apoio à decisão e não substitui avaliação médica individualizada.')


with tab2:
    st.header('Dashboard Analítico')

    st.subheader('Distribuição dos níveis de obesidade')
    obesity_counts = df['Obesity'].value_counts()
    st.bar_chart(obesity_counts)

    st.subheader('IMC médio por nível de obesidade')
    bmi_by_obesity = df.groupby('Obesity')['BMI'].mean().sort_values()
    st.bar_chart(bmi_by_obesity)

    st.subheader('Peso médio por nível de obesidade')
    weight_by_obesity = df.groupby('Obesity')['Weight'].mean().sort_values()
    st.bar_chart(weight_by_obesity)

    st.subheader('Idade média por nível de obesidade')
    age_by_obesity = df.groupby('Obesity')['Age'].mean().sort_values()
    st.bar_chart(age_by_obesity)

    st.subheader('Histórico familiar de excesso de peso')
    family_table = pd.crosstab(df['family_history'], df['Obesity'])
    st.dataframe(family_table)

    st.subheader('Consumo frequente de alimentos calóricos')
    favc_table = pd.crosstab(df['FAVC'], df['Obesity'])
    st.dataframe(favc_table)

    st.subheader('Atividade física por nível de obesidade')
    faf_table = df.groupby('Obesity')['FAF'].mean().sort_values()
    st.bar_chart(faf_table)

    st.subheader('Principais insights para a equipe médica')

    st.markdown("""
    - O IMC médio aumenta conforme o nível de obesidade se torna mais elevado.
    - O histórico familiar de excesso de peso aparece como uma variável relevante para análise de risco.
    - O consumo frequente de alimentos calóricos contribui para diferenciar perfis comportamentais.
    - A frequência de atividade física ajuda a compor uma visão mais completa sobre o estilo de vida do paciente.
    - O modelo preditivo pode apoiar uma triagem inicial, ajudando a equipe médica a identificar perfis que merecem maior atenção.
    """)


with tab3:
    st.header('Sobre o Projeto')

    st.markdown("""
    Este projeto foi desenvolvido para o Tech Challenge da Pós Tech em Data Analytics.

    O objetivo foi construir uma solução de Machine Learning capaz de prever o nível de obesidade de uma pessoa a partir de dados demográficos, alimentares, comportamentais e de estilo de vida.

    A solução contempla:

    - análise exploratória dos dados;
    - criação da variável de IMC;
    - pipeline de machine learning;
    - tratamento de variáveis numéricas e categóricas;
    - treinamento de modelo Random Forest;
    - sistema preditivo em Streamlit;
    - dashboard analítico com principais insights;
    - visão de negócio voltada ao apoio à equipe médica.

    O modelo foi desenvolvido para fins educacionais e deve ser interpretado como apoio à decisão, não como substituto de avaliação médica profissional.
    """)
