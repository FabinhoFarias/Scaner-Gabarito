import streamlit as st
import numpy as np
import cv2
from corretor import executar_pipeline_omr
import json


# Configuração inicial da página Web
st.set_page_config(page_title="Corretor de Gabaritos OMR", layout="centered")
st.title("📝 Corretor Inteligente de Gabaritos")
st.write("Tire uma foto nítida do cartão-resposta alinhando os 4 cantos em 'L'.")

# Opção de entrada: Câmera ativa ou Upload de foto pronta
metodo_entrada = st.radio("Escolha o método de entrada:", ("Tirar Foto pela Câmera", "Carregar Arquivo de Imagem"))

imagem_bytes = None

if metodo_entrada == "Tirar Foto pela Câmera":
    # Cria um componente de câmera web direto no navegador (funciona no celular também!)
    imagem_bytes = st.camera_input("Posicione o gabarito no fundo escuro e clique em Tirar Foto")
else:
    imagem_bytes = st.file_uploader("Selecione a foto do gabarito (Formato JPG/PNG)", type=["jpg", "jpeg", "png"])

# Se o usuário forneceu uma imagem (tirou foto ou fez upload)
if imagem_bytes is not None:
    
    # Converter os bytes recebidos pelo Streamlit em uma matriz de imagem do OpenCV (BGR)
    file_bytes = np.frombuffer(imagem_bytes.getvalue(), np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.info("🔄 Processando gabarito, aguarde...")
    
    # Chamar a função purificada que criamos no passo anterior
    imagem_processada, respostas, status, sucesso = executar_pipeline_omr(frame_bgr)
    
    # Exibir a imagem com os feedbacks visuais na tela do navegador
    st.image(imagem_processada, caption="Visualização do Scanner", use_container_width=True)
    
    # Exibir status do processamento
    if sucesso:
        st.success(status)
        
        # Criar duas colunas para mostrar os resultados organizados
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Total de Questões Lidas", value=f"{len(respostas)}/90")
        with col2:
            # Permite o download do JSON gerado direto pelo navegador
            st.download_button(
                label="📥 Baixar JSON de Respostas",
                data=json.dumps(respostas, indent=4),
                file_name="respostas_detectadas.json",
                mime="application/json"
            )
            
        # Mostrar um preview das respostas em tabela na interface web
        with st.expander("Ver mapa de respostas detectadas"):
            st.json(respostas)
    else:
        st.error(status)