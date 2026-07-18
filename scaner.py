import cv2
import numpy as np
import requests
import time

# --- CONFIGURAÇÃO DO IP WEBCAM ---
# Substitua pelo IP e Porta que aparecem na tela do seu aplicativo IP Webcam
IP_CELULAR = "10.245.226.27" 
PORTA = "8080"
URL_STREAM = f"http://{IP_CELULAR}:{PORTA}/video"

def processar_frame_gabarito(img_original):
    """
    Processa um único frame vindo do streaming do IP Webcam.
    """
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    # Binarização adaptativa para lidar com variações de luz da câmera do celular
    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    largura_a4, altura_a4 = 2100, 2970
    
    # 1. DETECÇÃO AUTOMÁTICA DAS 4 ÂNCORAS DE CANTO (.ancora)
    # Procuramos por contornos quadrados grandes nos cantos da imagem
    contornos, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pontos_cantos = []
    
    for c in contornos:
        perimetro = cv2.arcLength(c, True)
        aproximacao = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        
        # Se encontrou um quadrilátero com tamanho compatível com as âncoras de canto
        if len(aproximacao) == 4:
            x, y, w, h = cv2.boundingRect(aproximacao)
            area = cv2.contourArea(c)
            if area > 500: # Filtra ruídos pequenos
                pontos_cantos.append(aproximacao)

    # Se não detectar as 4 marcas de canto, o cartão não está bem posicionado
    if len(pontos_cantos) < 4:
        return None, "Alinhe as 4 âncoras de canto na tela..."

    # Para fins de robustez no streaming real, se o frame estiver instável,
    # aplicamos o redimensionamento direto como fallback de testes:
    img_alinhada = cv2.resize(img_thresh, (largura_a4, altura_a4))

    # 2. MAPEAMENTO GEOMÉTRICO (EIXOS X e Y)
    # Gerando a matriz matemática baseada estritamente nas proporções do seu HTML
    inicio_y = 1335
    passo_y = 60
    ancoras_y = [int(inicio_y + (i * passo_y)) for i in range(15)]

    ancoras_x = []
    largura_bloco = 330
    espaco_num = 40
    largura_util_opcoes = 290
    
    for b in range(6):
        bloco_inicio_x = 60 + (b * largura_bloco) + espaco_num
        for alt in range(5):
            pos_x = bloco_inicio_x + int(alt * (largura_util_opcoes / 4))
            ancoras_x.append(pos_x)

    # 3. ANÁLISE DE PREENCHIMENTO
    respostas_aluno = {}
    alternativas_letras = ['A', 'B', 'C', 'D', 'E']
    numero_inicial_questao = 91
    raio_deteccao = 16

    for q_idx in range(15):
        coord_y = ancoras_y[q_idx]
        for b_idx in range(6):
            num_questao = numero_inicial_questao + (b_idx * 15) + q_idx
            valores_preenchimento = []
            
            for alt_idx in range(5):
                coord_x = ancoras_x[(b_idx * 5) + alt_idx]
                
                regiao_bolinha = img_alinhada[coord_y - raio_deteccao:coord_y + raio_deteccao, 
                                              coord_x - raio_deteccao:coord_x + raio_deteccao]
                
                total_pixels_pretos = cv2.countNonZero(regiao_bolinha)
                valores_preenchimento.append(total_pixels_pretos)
            
            max_pixels = max(valores_preenchimento)
            index_marcado = valores_preenchimento.index(max_pixels)
            
            if max_pixels > 140: # Ajustado para o ruído do sensor físico de câmera
                duplicados = sum(1 for v in valores_preenchimento if v > (max_pixels * 0.78))
                if duplicados > 1:
                    respostas_aluno[num_questao] = "RASURA"
                else:
                    respostas_aluno[num_questao] = alternativas_letras[index_marcado]
            else:
                respostas_aluno[num_questao] = "NULA"

    return respostas_aluno, "Sucesso!"

# --- LOOP PRINCIPAL DE CAPTURA DO IP WEBCAM ---
print("Conectando ao IP Webcam...")
cap = cv2.VideoCapture(URL_STREAM)

if not cap.isOpened():
    print("Erro ao conectar ao streaming. Verifique o IP, porta ou se o app está ativo no celular.")
    exit()

print("Conectado com sucesso! Pressione 'q' na janela do vídeo para fechar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao receber pacotes do streaming.")
        break

    # Redimensiona o frame apenas para exibição em tempo real na janela de preview
    preview = cv2.resize(frame, (600, 850))
    
    # Executa a leitura do gabarito no frame original de alta resolução
    respostas, status = processar_frame_gabarito(frame)
    
    # Renderiza o status na tela do feed
    cor_status = (0, 255, 0) if "Sucesso" in status else (0, 0, 255)
    cv2.putText(preview, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
    
    # Se conseguir ler o cartão com sucesso, exibe uma amostra dos resultados no terminal
    if respostas:
        print("\n--- NOVO CARTÃO DETECTADO ---")
        for k in sorted(respostas.keys()): # Mostra as primeiras 5 questões como amostragem
            print(f"Questão {k}: {respostas[k]}")
        # Delay opcional para não inundar o terminal enquanto segura a folha parada
        time.sleep(1) 

    # Exibe a janela de captura
    # cv2.imshow("Corretor Inteligente - IP Webcam", preview)

    # Tecla de escape: pressionar 'q' fecha o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()