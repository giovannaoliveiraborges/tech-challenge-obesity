
import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_obesidade.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Obesity.csv')

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

st.set_page_config(
    page_title='Sistema Preditivo de Obesidade',
    page_icon='🩺',
    layout='wide'
)

st.title('🩺 Sistema Preditivo de Nível de Obesidade')

st.markdown("""
Esta aplicação utiliza um modelo de Machine Learning para estimar o nível de obesidade de uma pessoa com base em informações demográficas, alimentares, comportamentais e de estilo de vida.

A solução tem como objetivo apoiar a equipe médica na identificação de perfis de risco e na tomada de decisão clínica.
""")

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

    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)

    bmi_by_obesity = df.groupby('Obesity')['BMI'].mean().sort_values()
    st.bar_chart(bmi_by_obesity)

    st.subheader('Peso médio por nível de obesidade')
    weight_by_obesity = df.groupby('Obesity')['Weight'].mean().sort_values()
    st.bar_chart(weight_by_obesity)

    st.subheader('Histórico familiar de excesso de peso')
    family_table = pd.crosstab(df['family_history'], df['Obesity'])
    st.dataframe(family_table)

    st.subheader('Consumo frequente de alimentos calóricos')
    favc_table = pd.crosstab(df['FAVC'], df['Obesity'])
    st.dataframe(favc_table)

    st.subheader('Principais insights para a equipe médica')

    st.markdown("""
    - O IMC médio aumenta de forma consistente conforme o nível de obesidade se torna mais elevado.
    - O histórico familiar de excesso de peso pode ser usado como um sinal importante para triagem de risco.
    - Hábitos alimentares, prática de atividade física, consumo de água e meio de transporte ajudam a compor uma visão mais ampla do perfil do paciente.
    - O modelo preditivo pode apoiar a equipe médica em uma análise inicial, priorizando acompanhamento preventivo para perfis de maior risco.
    """)

with tab3:
    st.header('Sobre o Projeto')

    st.markdown("""
    Este projeto foi desenvolvido para o Tech Challenge da Pós Tech em Data Analytics.

    O objetivo foi construir uma solução de Machine Learning capaz de prever o nível de obesidade de uma pessoa a partir de dados demográficos, alimentares, comportamentais e de estilo de vida.

    A solução contempla:

    - pipeline de machine learning;
    - engenharia de atributos;
    - treinamento e avaliação de modelo;
    - sistema preditivo em Streamlit;
    - dashboard analítico com principais insights;
    - visão de negócio voltada ao apoio à equipe médica.
    """)
