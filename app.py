import streamlit as st
from corretor import processar_gabarito

st.set_page_config(page_title="Corretor OMR", layout="wide")

st.title("📸 Corretor Inteligente de Gabaritos (91-180)")
st.subheader("Módulo de Leitura e Validação Geométrica")

uploaded_file = st.file_uploader("Envie a foto do Cartão-Resposta...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    
    with st.spinner('Processando matriz de marcações...'):
        respostas, img_resultado, status = processar_gabarito(bytes_data)
        
    if respostas:
        st.success(status)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(img_resultado, channels="BGR", use_column_width=True, 
                     caption="Grid de Alinhamento Corrigido sobre as Questões")
            
        with col2:
            st.header("📊 Respostas Extraídas")
            
            dados_tabela = [{"Questão": k, "Marcado": v} for k, v in sorted(respostas.items())]
            
            sub_col1, sub_col2 = st.columns(2)
            meio = len(dados_tabela) // 2
            
            sub_col1.dataframe(dados_tabela[:meio], hide_index=True, use_container_width=True)
            sub_col2.dataframe(dados_tabela[meio:], hide_index=True, use_container_width=True)
    else:
        st.error(status)