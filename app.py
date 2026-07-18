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
    
    # AJUSTE AQUI: Agora recebemos 5 variáveis (adicionada a 'imagem_debug_ancoras')
    imagem_processada, imagem_debug_ancoras, respostas, status, sucesso = executar_pipeline_omr(frame_bgr)
    
    # Exibir a imagem com os feedbacks visuais na tela do navegador
    st.image(imagem_processada, caption="Visualização do Scanner", use_container_width=True)
    
    # --- NOVO: BLOCO DE DEBUG DAS 90 ÂNCORAS ---
    with st.expander("🔍 Ver Debug de Validação (90 Âncoras Pequenas)"):
        if imagem_debug_ancoras is not None:
            st.image(imagem_debug_ancoras, caption="Rastreamento Individual das Margens", use_container_width=True)
        else:
            st.warning("Imagem de debug das âncoras não disponível.")
            
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
            
        # --- NOVO: MONTAGEM DOS DOIS BLOCOS DE 45 PARA PLANILHA ---
        st.markdown("---")
        st.subheader("📋 Copiar Alternativas para a Planilha")
        st.write("Clique no ícone de cópia no canto direito de cada bloco para colar na sua planilha:")
        
        # Construindo a string linear baseada na ordem correta das chaves numéricas (1 a 90)
        string_gabarito = ""
        for i in range(1, 91):
            # Garante que chaves strings ou inteiras funcionem e substitui falhas por 'x'
            res_atual = respostas.get(i) or respostas.get(str(i)) or 'X'
            string_gabarito += str(res_atual)
            
        bloco_1 = string_gabarito[:45].lower()
        bloco_2 = string_gabarito[45:].lower()
        
        # Exibição com o botão "Copy" nativo do Streamlit
        st.markdown("**Questões 01 a 45:**")
        st.code(bloco_1, language="text")
        
        st.markdown("**Questões 46 a 90:**")
        st.code(bloco_2, language="text")
        # -----------------------------------------------------------
        
        # Mostrar um preview das respostas em tabela na interface web
        with st.expander("Ver mapa completo em formato JSON"):
            st.json(respostas)
    else:
        st.error(status)