import sys
import json
import cv2
import numpy as np

# ----------------------------------------------------------------------
# FUNÇÃO DE DETECÇÃO DA FOLHA (RETÂNGULO)
# ----------------------------------------------------------------------

def detectar_folha_retangular(frame_colorido):
    # 1. PRÉ-PROCESSAMENTO: Isolar a folha branca do fundo escuro
    imagem_cinza = cv2.cvtColor(frame_colorido, cv2.COLOR_BGR2GRAY)
    imagem_borrada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)
    
    # Binarização Global de Otsu: Como o fundo é preto e o papel é branco,
    # isso isola a folha perfeitamente sem gerar o ruído do threshold adaptativo.
    _, thresh = cv2.threshold(imagem_borrada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. ENCONTRAR O MAIOR CONTORNO (A FOLHA)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
        
    # O maior contorno disparado será a folha de papel
    maior_contorno = max(contornos, key=cv2.contourArea)
    area_folha = cv2.contourArea(maior_contorno)
    
    altura_img, largura_img = frame_colorido.shape[:2]
    # Filtro de segurança: a folha deve ocupar pelo menos 10% da imagem
    if area_folha < (altura_img * largura_img * 0.10):
        return None
        
    # 3. ENCONTRAR OS 4 CANTOS APROXIMADOS DA FOLHA
    perimetro = cv2.arcLength(maior_contorno, True)
    poligono = cv2.approxPolyDP(maior_contorno, 0.02 * perimetro, True)
    
    # Se a aproximação direta não retornar exatamente 4 pontos, extraímos via Convex Hull
    if len(poligono) != 4:
        hull = cv2.convexHull(maior_contorno)
        poligono = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
        if len(poligono) != 4:
            # Fallback seguro usando o retângulo delimitador rotacionado
            rect = cv2.minAreaRect(maior_contorno)
            box = cv2.boxPoints(rect)
            poligono = np.int32(box).reshape(-1, 1, 2)
            
    # 4. ORDENAR OS CANTOS DA FOLHA (Superior Esquerdo, Superior Direito, etc.)
    pts = poligono.reshape(4, 2)
    somas = pts.sum(axis=1)
    dif = np.diff(pts, axis=1)
    
    cantos_folha = np.zeros((4, 2), dtype="float32")
    cantos_folha[0] = pts[np.argmin(somas)]       # Top-Left
    cantos_folha[2] = pts[np.argmax(somas)]       # Bottom-Right
    cantos_folha[1] = pts[np.argmin(dif)]         # Top-Right
    cantos_folha[3] = pts[np.argmax(dif)]         # Bottom-Left
    
    # 5. REFINAMENTO DOS CANTOS COM DETECTOR DE CANTOS DO OPENCV
    cantos_precisos = []
    tamanho_roi = 80  # Define uma janela de busca de 80x80 pixels ao redor de cada canto
    metade = tamanho_roi // 2
    
    # Direções para mover as janelas de busca ligeiramente para DENTRO do papel
    # para garantir que a marca "L" esteja totalmente visível na janela recortada.
    direcoes = [
        [1, 1],   # Top-Left: move para direita e baixo
        [-1, 1],  # Top-Right: move para esquerda e baixo
        [-1, -1], # Bottom-Right: move para esquerda e cima
        [1, -1]   # Bottom-Left: move para direita e cima
    ]
    
    # Parâmetros de parada para o algoritmo de refinamento subpixel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    
    for i in range(4):
        cx, cy = cantos_folha[i]
        
        # Deslocamento estratégico de 32 pixels para focar exatamente no "L"
        shift_pixels = 32
        roi_cx = int(cx + direcoes[i][0] * shift_pixels)
        roi_cy = int(cy + direcoes[i][1] * shift_pixels)
        
        # Coordenadas de corte do ROI com limites físicos da imagem respeitados
        x1 = max(0, roi_cx - metade)
        y1 = max(0, roi_cy - metade)
        x2 = min(largura_img, roi_cx + metade)
        y2 = min(altura_img, roi_cy + metade)
        
        roi_recortado = imagem_cinza[y1:y2, x1:x2]
        
        # Executa o Detector de Cantos de Shi-Tomasi do OpenCV
        # O canto externo do "L" possui uma resposta matemática de canto extremamente alta
        cantos_detectados = cv2.goodFeaturesToTrack(
            roi_recortado, 
            maxCorners=4, 
            qualityLevel=0.08, 
            minDistance=10
        )
        
        if cantos_detectados is not None and len(cantos_detectados) > 0:
            pts_locais = cantos_detectados.reshape(-1, 2)
            
            # Escolhe o canto detectado mais próximo do canto original do papel
            canto_local_papel = np.array([cx - x1, cy - y1])
            distancias = np.linalg.norm(pts_locais - canto_local_papel, axis=1)
            melhor_idx = np.argmin(distancias)
            
            local_x, local_y = pts_locais[melhor_idx]
            
            # Refina a coordenada para precisão Subpixel (abaixo de 1 pixel)
            ponto_refinado = np.array([[local_x, local_y]], dtype=np.float32).reshape(-1, 1, 2)
            try:
                cv2.cornerSubPix(roi_recortado, ponto_refinado, (5, 5), (-1, -1), criteria)
                local_x, local_y = ponto_refinado[0][0]
            except:
                pass # Caso falte contraste para subpixel, usa a detecção padrão
            
            # Reconverte as coordenadas locais do ROI de volta para globais da imagem
            global_x = x1 + local_x
            global_y = y1 + local_y
            cantos_precisos.append([global_x, global_y])
        else:
            # Caso o detector falhe por completo, usa o canto aproximado do papel como segurança
            cantos_precisos.append([cx, cy])
            
    return np.array(cantos_precisos, dtype="float32").reshape((4, 1, 2))


def ordenar_cantos(cantos):
    """
    Ordena os 4 pontos de um quadrilátero em uma ordem consistente:
    (Superior Esquerdo, Superior Direito, Inferior Direito, Inferior Esquerdo).
    """
    # CORRIGIDO: A forma correta para 4 pontos (X, Y) é (4, 2).
    # O OpenCV espera 4 pontos de origem e 4 pontos de destino no formato (4, 2).
    cantos = cantos.reshape((4, 2))
    
    # CORRIGIDO: O array de saída também deve ser (4, 2).
    cantos_ordenados = np.zeros((4, 2), dtype="float32")

    # 1. Soma das coordenadas (x + y):
    # axis=1 garante que (X + Y) de cada ponto seja somado.
    somas = cantos.sum(axis=1) 
    
    # Ponto de menor soma (Top-Left)
    cantos_ordenados[0] = cantos[np.argmin(somas)]  
    # Ponto de maior soma (Bottom-Right)
    cantos_ordenados[2] = cantos[np.argmax(somas)]  

    # 2. Diferença das coordenadas (x - y):
    # np.diff(cantos, axis=1) calcula (X - Y) para cada ponto.
    diferencas = np.diff(cantos, axis=1)
    
    # Ponto de menor diferença (x - y) -> Top-Right (Geralmente X é grande, Y é pequeno)
    cantos_ordenados[1] = cantos[np.argmin(diferencas)] 
    # Ponto de maior diferença (x - y) -> Bottom-Left (Geralmente X é pequeno, Y é grande)
    cantos_ordenados[3] = cantos[np.argmax(diferencas)] 

    return cantos_ordenados

def transformar_perspectiva_e_cortar(imagem, cantos_folha):
    """
    Aplica a Transformação de Perspectiva para endireitar e cortar o gabarito.

    Args:
        imagem (np.array): O frame de vídeo original (BGR).
        cantos_folha (np.array): Os 4 pontos do contorno detectado.

    Returns:
        np.array: A folha de papel endireitada e cortada.
    """
    # 1. ORDENAR OS CANTOS DE ENTRADA
    cantos_ordenados = ordenar_cantos(cantos_folha)
    (tl, tr, br, bl) = cantos_ordenados

    # 2. DEFINIR AS DIMENSÕES DO NOVO RETÂNGULO
    # Calcula a maior largura (distância entre tr e br, ou tl e bl)
    larguraA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    larguraB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    largura_max = max(int(larguraA), int(larguraB))

    # Calcula a maior altura (distância entre tr e tl, ou br e bl)
    alturaA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)) # Distância entre TR e BR
    alturaB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)) # Distância entre TL e BL
    altura_max = max(int(alturaA), int(alturaB))
    
    # 3. DEFINIR OS PONTOS DE DESTINO
    # Os pontos no novo retângulo, com a mesma largura e altura
    pontos_destino = np.array([
        [0, 0],                            # Superior Esquerdo (0, 0)
        [largura_max - 1, 0],              # Superior Direito
        [largura_max - 1, altura_max - 1],  # Inferior Direito
        [0, altura_max - 1]],             # Inferior Esquerdo
        dtype="float32")

    # 4. CALCULAR A MATRIZ DE PERSPECTIVA (M)
    # Mapeia os cantos_ordenados para os pontos_destino
    M = cv2.getPerspectiveTransform(cantos_ordenados, pontos_destino)

    # 5. APLICAR A TRANSFORMAÇÃO E CORTAR
    # cv2.warpPerspective aplica a transformação M
    imagem_cortada_endireitada = cv2.warpPerspective(imagem, M, (largura_max, altura_max))

    return imagem_cortada_endireitada

def cortar_imagem(imagem, x_inicial, y_inicial, largura, altura):
    """
    Corta uma Região de Interesse (ROI) da imagem.
    
    Args:
        imagem (np.array): A imagem de entrada (frame).
        x_inicial (int): Coordenada X (coluna) de onde o corte deve começar.
        y_inicial (int): Coordenada Y (linha) de onde o corte deve começar.
        largura (int): A largura do corte desejada.
        altura (int): A altura do corte desejada.
        
    Returns:
        np.array: A imagem cortada (ROI).
    """
    
    # 1. Calcular as coordenadas finais
    x_final = largura
    y_final = altura
    
    # 2. Aplicar o fatiamento do NumPy:
    # [y_inicial : y_final, x_inicial : x_final]
    # (Lembre-se que em arrays NumPy, a altura (Y) vem primeiro, depois a largura (X))
    imagem_cortada = imagem[y_inicial:y_final, x_inicial:x_final]
    
    return imagem_cortada




def detectar_respostas_somente_dicionario(lista_blocos):
    """
    Detecta as respostas em uma lista de blocos (imagens binárias) iterando 
    fixamente (sem detecção de picos) para garantir a contagem de 90 questões.
    
    A detecção da linha de questão (eixo Y) é fixada usando o pitch (espaçamento) 
    e um ponto de partida inicial fixo.
    
    Args:
        lista_blocos (list): Lista de numpy.ndarrays (blocos binários, fundo 0, marca 255).

    Returns:
        dict: Dicionário de respostas detectadas no formato {Questão: Resposta}.
    """
    
    # --- VARIÁVEIS DE MAPEAMENTO E CONFIGURAÇÃO ---
    respostas_finais = {} 
    numero_global_questao = 1
    MAPA_ALTERNATIVA = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    # Parâmetros de calibração de detecção 
    FATOR_DUPLA_MARCACAO = 0.90   # Se 2ª Marca > 90% da 1ª Marca, é dupla.
    TAMANHO_AMOSTRAGEM_Y = 15     # Altura da linha de pixels para somar marcas.
    EXPECTED_Q_PER_BLOCK = 15    # Número Fixo de questões por bloco (15*6 = 90).
    
    # CALIBRAÇÃO DE PONTO DE PARTIDA FIXO (NOVO)
    # Assuma que a primeira questão começa após uma margem superior.
    # Ex: A primeira questão começa 1/4 da altura do pitch da primeira questão.
    FATOR_MARGEM_SUPERIOR = 0.25 
    
    for bloco_idx, Bloco in enumerate(lista_blocos):
        
        # 0. VERIFICAÇÃO INICIAL E CÁLCULOS FIXOS
        if Bloco.size == 0:
            print(f"ATENÇÃO: Bloco {bloco_idx + 1} está vazio e foi ignorado.")
            for _ in range(EXPECTED_Q_PER_BLOCK):
                respostas_finais[numero_global_questao] = 'X'
                numero_global_questao += 1
            continue
            
        print(f"Processando bloco: {bloco_idx + 1}")
        
        # CÁLCULOS DE PITCH E LARGURA
        altura_bloco = Bloco.shape[0]
        altura_pitch_q = altura_bloco // EXPECTED_Q_PER_BLOCK
        LARGURA_BLOCO = Bloco.shape[1]
        LARGURA_CELULA_X = LARGURA_BLOCO // 5
        
        # 1. GERAÇÃO DAS 15 POSIÇÕES Y (Iteração Fixa - SUBSTITUI A LÓGICA DE PICO)
        
        # O centro da primeira questão é a margem superior (baseado no FATOR_MARGEM_SUPERIOR)
        # Mais metade da altura de amostragem Y, ou simplesmente um fator do pitch.
        y_margem_superior = int(altura_pitch_q * FATOR_MARGEM_SUPERIOR)
        y_centro_inicial = y_margem_superior + (altura_pitch_q // 2)
        
        # Se as imagens são bem controladas, o centro da primeira questão
        # pode ser fixado com base no fator de margem superior.
        
        indices_y_questoes_fixos = []
        for i in range(EXPECTED_Q_PER_BLOCK):
            y_pos = y_centro_inicial + i * altura_pitch_q
            if y_pos < altura_bloco:
                indices_y_questoes_fixos.append(y_pos)
        
        # TRATAMENTO DE ERRO: Se a calibração de pitch falhar (o que é improvável se o bloco foi cortado corretamente)
        if len(indices_y_questoes_fixos) < EXPECTED_Q_PER_BLOCK:
             print(f"AVISO: Bloco {bloco_idx + 1} encontrou {len(indices_y_questoes_fixos)} questões, esperado {EXPECTED_Q_PER_BLOCK}.")
             # O loop principal abaixo processará apenas as questões encontradas.

        # 2. LOOP PRINCIPAL PARA CADA CÉLULA (Soma de Marca)
        MARCA_MAX_POSSIVEL = TAMANHO_AMOSTRAGEM_Y * LARGURA_CELULA_X * 255 
        LIMITE_MARCA_MINIMO = MARCA_MAX_POSSIVEL * 0.05 # 15% de preenchimento mínimo (Ajustável!)

        for y_centro in indices_y_questoes_fixos:
            
            valores_de_marca = []
            
            # 2.1. EXTRAIR A LINHA DE AMOSTRAGEM DA QUESTÃO
            y_start = y_centro - TAMANHO_AMOSTRAGEM_Y // 2
            y_end = y_start + TAMANHO_AMOSTRAGEM_Y
            
            # Garantindo que os índices não saiam dos limites do Bloco
            y_start = max(0, y_start)
            y_end = min(Bloco.shape[0], y_end)

            linha_questao = Bloco[y_start:y_end, :] 

            # 2.2. DIVIDIR EM 5 COLUNAS E SOMAR AS MARCAS (255)
            for i in range(5):
                x_start = i * LARGURA_CELULA_X
                x_end = (i + 1) * LARGURA_CELULA_X
                
                regiao_alternativa = linha_questao[:, x_start:x_end]
                soma_marcas = np.sum(regiao_alternativa) 
                valores_de_marca.append(soma_marcas)

            # 3. DETERMINAÇÃO DA RESPOSTA (MAIOR MARCA = Resposta Válida)
            if not valores_de_marca:
                alternativa_marcada = 'X' # Fallback para não detectada
            else:
                # Encontra o índice (0 a 4) com a MAIOR soma de marca.
                indice_resposta = np.argmax(valores_de_marca)
                if numero_global_questao == 81:
                    print(valores_de_marca ,indice_resposta)
                if numero_global_questao == 82:
                    print(valores_de_marca ,indice_resposta)
                if numero_global_questao == 83:
                    print(valores_de_marca ,indice_resposta)
                if numero_global_questao == 84:
                    print(valores_de_marca ,indice_resposta)

                marca_na_resposta = valores_de_marca[indice_resposta]
                
                # A. VERIFICAÇÃO DE MARCAÇÃO FRACA
                if marca_na_resposta < LIMITE_MARCA_MINIMO:
                    alternativa_marcada = 'X' # Não respondida
                else:
                    # B. VERIFICAÇÃO DE DUPLA MARCAÇÃO
                    valores_ordenados = sorted(valores_de_marca, reverse=True)
                    
                    if len(valores_ordenados) >= 2 and (valores_ordenados[1] / valores_ordenados[0]) > FATOR_DUPLA_MARCACAO:
                        alternativa_marcada = 'X' # Dupla Marcação (Anulada)
                    else:
                        # C. RESPOSTA VÁLIDA
                        alternativa_marcada = MAPA_ALTERNATIVA[indice_resposta]

            # 4. Armazena o resultado no dicionário final
            respostas_finais[numero_global_questao] = alternativa_marcada
            numero_global_questao += 1
            
    return respostas_finais

def mostrar_detecao_90_ancoras(gabarito_endireitado, topo, base, esq, dir):
    """
    Exibe uma janela de debug desenhando individualmente cada uma das 90 âncoras
    com códigos de cores separados por margem para facilitar a validação.
    """
    img_visual = gabarito_endireitado.copy()
    
    # 1. Âncoras do Topo (30) - Vermelho
    for idx, (x, y) in enumerate(topo):
        cv2.circle(img_visual, (x, y), 4, (0, 0, 255), -1)
        if idx % 5 == 0:  # Rótulo simples para não poluir
            cv2.putText(img_visual, f"T{idx}", (x - 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
    # 2. Âncoras da Base (30) - Verde
    for idx, (x, y) in enumerate(base):
        cv2.circle(img_visual, (x, y), 4, (0, 255, 0), -1)
        if idx % 5 == 0:
            cv2.putText(img_visual, f"B{idx}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
    # 3. Âncoras da Esquerda (15) - Azul
    for idx, (x, y) in enumerate(esq):
        cv2.circle(img_visual, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(img_visual, f"L{idx+1}", (x + 8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
    # 4. Âncoras da Direita (15) - Amarelo
    for idx, (x, y) in enumerate(dir):
        cv2.circle(img_visual, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(img_visual, f"R{idx+1}", (x - 22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
    cv2.imshow("DEBUG: 90 Ancoras Separadas", img_visual)

def obter_intersecao_ancoras(pt_topo, pt_base, pt_esq, pt_dir):
    """
    Calcula matematicamente o ponto de interseção exato de duas retas formadas
    pelas âncoras correspondentes. Resolve qualquer distorção de rotação ou inclinação.
    """
    x1, y1 = pt_topo
    x2, y2 = pt_base
    x3, y3 = pt_esq
    x4, y4 = pt_dir
    
    denominador = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominador == 0:
        # Fallback de segurança se as retas forem perfeitamente paralelas
        return int((x1 + x2) / 2), int((y3 + y4) / 2)
        
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominador
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominador
    return int(px), int(py)

def detectar_90_ancoras(gabarito_endireitado):
    """
    Detecta individualmente as 90 âncoras físicas (30 topo, 30 base, 15 esquerda, 15 direita)
    utilizando análise de contornos e proporção nas margens do papel.
    """
    altura, largura = gabarito_endireitado.shape[:2]
    
    # 1. Pré-processamento focado nas marcas pretas das margens
    gabarito_cinza = cv2.cvtColor(gabarito_endireitado, cv2.COLOR_BGR2GRAY)
    gabarito_borrado = cv2.GaussianBlur(gabarito_cinza, (3, 3), 0)
    
    # Binarização adaptativa para destacar os quadradinhos pretos das margens (viram branco = 255)
    thresh = cv2.adaptiveThreshold(
        gabarito_borrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 7
    )
    
    # Encontrar contornos externos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    topo_cands = []
    base_cands = []
    esq_cands = []
    dir_cands = []
    
    # Limites dinâmicos de área para as pequenas marcas
    area_total = altura * largura
    area_min = area_total * 0.00004  # Cerca de 10 a 15 pixels de área
    area_max = area_total * 0.0015   # No máximo um pequeno bloco
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area_min < area < area_max:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            proporcao = w / float(h)
            
            # Classificar de acordo com a região da margem
            # Margem Superior (30 Âncoras - Quadradas)
            if cy < altura * 0.12 and 0.6 <= proporcao <= 1.6:
                topo_cands.append((cx, cy))
                
            # Margem Inferior (30 Âncoras - Quadradas)
            elif cy > altura * 0.88 and 0.6 <= proporcao <= 1.6:
                base_cands.append((cx, cy))
                
            # Margem Esquerda (15 Âncoras - Retângulos Horizontais)
            elif cx < largura * 0.08 and 1.0 <= proporcao <= 4.0:
                esq_cands.append((cx, cy))
                
            # Margem Direita (15 Âncoras - Retângulos Horizontais)
            elif cx > largura * 0.92 and 1.0 <= proporcao <= 4.0:
                dir_cands.append((cx, cy))

    # Função interna para limpar ruídos redundantes/próximos
    def filtrar_e_ordenar(candidatos, por_x=True, dist_min=8):
        if not candidatos:
            return []
        candidatos = sorted(candidatos, key=lambda p: p[0] if por_x else p[1])
        filtrados = [candidatos[0]]
        for p in candidatos[1:]:
            last = filtrados[-1]
            dist = abs(p[0] - last[0]) if por_x else abs(p[1] - last[1])
            if dist > dist_min:
                filtrados.append(p)
        return filtrados

    topo = filtrar_e_ordenar(topo_cands, por_x=True)
    base = filtrar_e_ordenar(base_cands, por_x=True)
    esquerda = filtrar_e_ordenar(esq_cands, por_x=False)
    direita = filtrar_e_ordenar(dir_cands, por_x=False)

    # 2. SISTEMA DE SEGURANÇA (INTERPOLAÇÃO)
    # Se alguma âncora falhou devido à impressão ou rasura, reconstrói linearmente
    def interpolar_ancoras(pontos, n_esperado, por_x=True, tamanho_total=100):
        if len(pontos) == n_esperado:
            return pontos
        
        # Caso extremo de falha total de detecção na margem
        if len(pontos) < 2:
            margem = tamanho_total * 0.05
            passo = (tamanho_total - 2 * margem) / (n_esperado - 1)
            coord_fixa = int(tamanho_total * (0.05 if n_esperado != 15 else 0.95))
            if por_x:
                return [(int(margem + i * passo), coord_fixa) for i in range(n_esperado)]
            else:
                return [(coord_fixa, int(margem + i * passo)) for i in range(n_esperado)]
                
        # Reconstrução linear baseada nos extremos detectados
        coords = np.array([p[0] if por_x else p[1] for p in pontos])
        outra_coord_media = int(np.mean([p[1] if por_x else p[0] for p in pontos]))
        min_val, max_val = coords[0], coords[-1]
        
        pontos_finais = []
        if n_esperado == 30: # Topo/Base: 6 blocos de 5 colunas
            # Distribuição teórica proporcional das colunas do gabarito
            pesos_colunas = []
            for bloco in range(6):
                for col in range(5):
                    pesos_colunas.append(bloco * 6.5 + col * 1.0)
            pesos_colunas = np.array(pesos_colunas)
            pesos_normalizados = (pesos_colunas - pesos_colunas[0]) / (pesos_colunas[-1] - pesos_colunas[0])
            coords_reconstruidas = min_val + pesos_normalizados * (max_val - min_val)
            
            for val in coords_reconstruidas:
                pontos_finais.append((int(val), outra_coord_media) if por_x else (outra_coord_media, int(val)))
        else: # Esquerda/Direita: 15 linhas lineares
            passo = (max_val - min_val) / 14.0
            for i in range(15):
                val = min_val + i * passo
                pontos_finais.append((int(val), outra_coord_media) if por_x else (outra_coord_media, int(val)))
                
        return pontos_finais

    topo_finais = interpolar_ancoras(topo, 30, por_x=True, tamanho_total=largura)
    base_finais = interpolar_ancoras(base, 30, por_x=True, tamanho_total=largura)
    esq_finais = interpolar_ancoras(esquerda, 15, por_x=False, tamanho_total=altura)
    dir_finais = interpolar_ancoras(direita, 15, por_x=False, tamanho_total=altura)

    return topo_finais, base_finais, esq_finais, dir_finais

def obter_centros_segmentos_1d(vetor_somas, threshold_ratio=0.15):
    """ Encontra os centros das âncoras agrupando índices contíguos de picos de sinal. """
    threshold = np.max(vetor_somas) * threshold_ratio
    indices = np.where(vetor_somas > threshold)[0]
    
    if len(indices) == 0:
        return []
        
    centros = []
    grupo = [indices[0]]
    for idx in indices[1:]:
        if idx - grupo[-1] <= 4:  # Tolerância de pixels contíguos para o mesmo bloco
            grupo.append(idx)
        else:
            centros.append(int(np.mean(grupo)))
            grupo = [idx]
    centros.append(int(np.mean(grupo)))
    return centros

def detectar_grid_ancoras(imagem_binaria):
    """
    Usa projeções horizontais e verticais para mapear as 30 colunas de alternativas
    e as 15 linhas de questões usando as marcas físicas de ancoragem.
    """
    altura, largura = imagem_binaria.shape[:2]
    
    # 1. Projeção Vertical para achar as colunas (X)
    # Analisamos a faixa superior (0% a 10%) e inferior (90% a 100%) onde os quadrados pretos estão
    zona_topo = imagem_binaria[0:int(altura * 0.10), :]
    zona_base = imagem_binaria[int(altura * 0.90):, :]
    
    soma_x_total = np.sum(zona_topo, axis=0) + np.sum(zona_base, axis=0)
    colunas_x = obter_centros_segmentos_1d(soma_x_total, threshold_ratio=0.15)
    
    # 2. Projeção Horizontal para achar as linhas das questões (Y)
    # Analisamos a faixa da esquerda (0% a 8%) e direita (92% a 100%) para achar os tracinhos horizontais
    zona_esquerda = imagem_binaria[:, 0:int(largura * 0.08)]
    zona_direita = imagem_binaria[:, int(largura * 0.92):]
    
    soma_y_total = np.sum(zona_esquerda, axis=1) + np.sum(zona_direita, axis=1)
    linhas_y = obter_centros_segmentos_1d(soma_y_total, threshold_ratio=0.15)
    
    # --- SISTEMA DE CONTROLE DE SEGURANÇA (FALLBACK) ---
    # Se alguma âncora falhar por causa de uma rasura ou sombra, geramos a grade matematicamente
    if len(colunas_x) != 30:
        print(f"⚠️ Aviso: Detectou {len(colunas_x)} colunas (esperado: 30). Aplicando grade linear de segurança.")
        colunas_x = []
        largura_util = largura - 30
        for bloco in range(6):
            x_bloco_inicio = 15 + bloco * (largura_util / 6.0)
            for alt in range(5):
                colunas_x.append(int(x_bloco_inicio + alt * (largura_util / 35.0)))
                
    if len(linhas_y) != 15:
        print(f"⚠️ Aviso: Detectou {len(linhas_y)} linhas (esperado: 15). Aplicando grade linear de segurança.")
        linhas_y = [int(22 + i * ((altura - 44) / 14)) for i in range(15)]
        
    return sorted(colunas_x), sorted(linhas_y)

def detectar_respostas_grid(imagem_binaria, colunas_x, linhas_y):
    """ Varre a grade de coordenadas gerada pelas âncoras para ler as marcações. """
    respostas_finais = {}
    MAPA_ALTERNATIVA = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    
    TAMANHO_AMOSTRAGEM = 14  # Janela quadrada de amostragem de pixels (14x14)
    FATOR_DUPLA_MARCACAO = 0.85
    
    MAX_VALOR_MARCA = TAMANHO_AMOSTRAGEM * TAMANHO_AMOSTRAGEM * 255
    LIMITE_MARCA_MINIMO = MAX_VALOR_MARCA * 0.20  # Pelo menos 20% do miolo preenchido
    
    for q in range(1, 91):
        bloco_idx = (q - 1) // 15
        linha_idx = (q - 1) % 15
        
        valores_alternativas = []
        y_centro = linhas_y[linha_idx]
        y_start = max(0, y_centro - TAMANHO_AMOSTRAGEM // 2)
        y_end = min(imagem_binaria.shape[0], y_start + TAMANHO_AMOSTRAGEM)
        
        # Avaliar as 5 alternativas da questão atual
        for alt_idx in range(5):
            col_idx = bloco_idx * 5 + alt_idx
            x_centro = colunas_x[col_idx]
            
            x_start = max(0, x_centro - TAMANHO_AMOSTRAGEM // 2)
            x_end = min(imagem_binaria.shape[1], x_start + TAMANHO_AMOSTRAGEM)
            
            # Medir o nível de preenchimento (pixels brancos na imagem invertida)
            regiao_bolha = imagem_binaria[y_start:y_end, x_start:x_end]
            valores_alternativas.append(np.sum(regiao_bolha))
            
        # Determinar qual foi preenchida
        indice_resposta = np.argmax(valores_alternativas)
        marca_na_resposta = valores_alternativas[indice_resposta]
        
        if marca_na_resposta < LIMITE_MARCA_MINIMO:
            alternativa_marcada = 'X'  # Branco/Não respondido
        else:
            valores_ordenados = sorted(valores_alternativas, reverse=True)
            # Validar dupla marcação
            if len(valores_ordenados) >= 2 and (valores_ordenados[1] / valores_ordenados[0]) > FATOR_DUPLA_MARCACAO:
                alternativa_marcada = 'X'  # Dupla marcação inválida
            else:
                alternativa_marcada = MAPA_ALTERNATIVA[indice_resposta]
                
        respostas_finais[q] = alternativa_marcada
        
    return respostas_finais

def desenhar_visualizacao_grid(imagem_bgr, colunas_x, linhas_y, respostas_finais):
    """ Desenha o grid das âncoras e as respostas lidas para validação em tempo real. """
    imagem_debug = imagem_bgr.copy()
    
    # 1. Desenhar as linhas guia verticais (magenta)
    for x in colunas_x:
        cv2.line(imagem_debug, (x, 0), (x, imagem_debug.shape[0]), (255, 0, 255), 1)
        
    # 2. Desenhar as linhas guia horizontais (ciano)
    for y in linhas_y:
        cv2.line(imagem_debug, (0, y), (imagem_debug.shape[1], y), (255, 255, 0), 1)
        
    # 3. Desenhar círculos verdes nas respostas detectadas
    for q, resp in respostas_finais.items():
        if resp != 'X':
            bloco_idx = (q - 1) // 15
            linha_idx = (q - 1) % 15
            alt_idx = ['A', 'B', 'C', 'D', 'E'].index(resp)
            
            x_centro = colunas_x[bloco_idx * 5 + alt_idx]
            y_centro = linhas_y[linha_idx]
            
            # Círculo verde neon no alvo lido
            cv2.circle(imagem_debug, (x_centro, y_centro), 8, (0, 255, 0), 2)
            
    return imagem_debug

def processar_e_detectar_respostas(gabarito_endireitado):
    # 1. Detectar as 90 âncoras separadamente
    topo, base, esq, dir_anc = detectar_90_ancoras(gabarito_endireitado)
    
    # Mostrar visualização das 90 âncoras separadas em tempo real
    mostrar_detecao_90_ancoras(gabarito_endireitado, topo, base, esq, dir_anc)
    
    # 2. Pré-processar imagem para ler o preenchimento das bolhas
    gabarito_cinza = cv2.cvtColor(gabarito_endireitado, cv2.COLOR_BGR2GRAY)
    gabarito_borrado = cv2.GaussianBlur(gabarito_cinza, (5, 5), 0)
    thresh_respostas = cv2.adaptiveThreshold(
        gabarito_borrado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 19, 9
    )
    
    respostas_finais = {}
    MAPA_ALTERNATIVA = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    TAMANHO_AMOSTRAGEM = 12  # Tamanho do quadrado de leitura de pixels da bolha
    
    # 3. Varredura Inteligente por Interseção de Retas
    for q in range(1, 91):
        bloco_idx = (q - 1) // 15
        linha_idx = (q - 1) % 15
        
        valores_alternativas = []
        pontos_alternativas = []
        
        # Obter a coordenada exata de cada uma das 5 alternativas da questão atual
        for alt_idx in range(5):
            col_idx = bloco_idx * 5 + alt_idx
            
            # Interseção da linha vertical (topo-base) com a linha horizontal (esquerda-direita)
            cx, cy = obter_intersecao_ancoras(topo[col_idx], base[col_idx], esq[linha_idx], dir_anc[linha_idx])
            pontos_alternativas.append((cx, cy))
            
            # Amostrar a região da bolha na imagem binária
            y1 = max(0, cy - TAMANHO_AMOSTRAGEM // 2)
            y2 = min(gabarito_endireitado.shape[0], y1 + TAMANHO_AMOSTRAGEM)
            x1 = max(0, cx - TAMANHO_AMOSTRAGEM // 2)
            x2 = min(gabarito_endireitado.shape[1], x1 + TAMANHO_AMOSTRAGEM)
            
            valores_alternativas.append(np.sum(thresh_respostas[y1:y2, x1:x2]))
            
        # Decidir qual alternativa foi marcada (Análise de intensidade de preenchimento)
        idx_marcado = np.argmax(valores_alternativas)
        max_preenchimento = valores_alternativas[idx_marcado]
        limiar_minimo = (TAMANHO_AMOSTRAGEM ** 2) * 255 * 0.25 # Pelo menos 25% preenchida
        
        if max_preenchimento < limiar_minimo:
            respostas_finais[q] = 'X'  # Não preenchida
        else:
            # Validar dupla marcação
            valores_ordenados = sorted(valores_alternativas, reverse=True)
            if len(valores_ordenados) >= 2 and (valores_ordenados[1] / valores_ordenados[0]) > 0.85:
                respostas_finais[q] = 'X'  # Dupla marcação inválida
            else:
                respostas_finais[q] = MAPA_ALTERNATIVA[idx_marcado]
                
    return respostas_finais


def salvar_respostas_json(respostas_dict, caminho_json='respostas_detectadas.json'):
    """Salva o dicionário de respostas detectadas em um arquivo JSON."""
    try:
        # Garante que as chaves (números de questão) são strings para compatibilidade JSON
        respostas_para_salvar = {str(k): v for k, v in respostas_dict.items()}
        with open(caminho_json, 'w') as f:
            json.dump(respostas_para_salvar, f, indent=4)
        print(f"\n✅ RESPOSTAS SALVAS COM SUCESSO: {caminho_json}")
    except Exception as e:
        print(f"\n❌ ERRO ao salvamr JSON: {e}")


# MAIN
def executar_pipeline_omr(frame_bgr):
    """
    Processa uma única foto do gabarito.
    Retorna: (frame_anotado_rgb, respostas_dict, mensagem_status, sucesso_bool)
    """
    # Criamos uma cópia para desenhar os feedbacks visuais sem alterar o original
    frame_exibicao = frame_bgr.copy()
    respostas_finais = None
    sucesso = False
    
    # --- DETECÇÃO DA FOLHA ---
    cantos_folha = detectar_folha_retangular(frame_bgr)

    if cantos_folha is not None:
        try:
            # 1. Alinhamento de Perspectiva
            gabarito_endireitado = transformar_perspectiva_e_cortar(frame_bgr, cantos_folha)
            gabarito_endireitado = cv2.resize(gabarito_endireitado, (400, 600)) 

            # 2. Definição da Região de Interesse (ROI) das questões
            x_start = 1
            y_start = int(600 * 0.4) 
            corte_largura = 400
            corte_altura = int(600 * 0.775)

            ROI = cortar_imagem(gabarito_endireitado, 
                                x_start, y_start, 
                                corte_largura, corte_altura)
        
            # 3. Processamento OMR das Alternativas
            respostas_finais = processar_e_detectar_respostas(ROI) 
            
            if respostas_finais:
                salvar_respostas_json(respostas_finais, caminho_json='respostas_detectadas.json')
                texto_status = "GABARITO ALINHADO! JSON GERADO COM SUCESSO."
                sucesso = True
            else:
                texto_status = "Folha encontrada, mas falha ao ler as alternativas."
        except Exception as e:
            texto_status = f"Erro no processamento interno: {str(e)}"
            
        # ==========================================
        # DESENHO DOS FEEDBACKS VISUAIS (CANTOS L)
        # ==========================================
        # Como é uma foto estática, desenhamos direto na escala original para não perder resolução
        cantos_desenho = cantos_folha.astype(np.int32).reshape(4, 2)
        
        # Linhas guia verdes
        cv2.polylines(frame_exibicao, [cantos_desenho], isClosed=True, color=(0, 255, 0), thickness=4)
        
        # Alvos nos cantos
        for i, ponto in enumerate(cantos_desenho):
            cx, cy = ponto
            cv2.circle(frame_exibicao, (cx, cy), 15, (255, 255, 0), 3)
            cv2.circle(frame_exibicao, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame_exibicao, f"L {i+1}", (cx - 20, cy - 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        texto_status = "Procurando Folha... Certifique-se de que os 4 cantos em 'L' estão visíveis."

    # ALERTA IMPORTANTE: O OpenCV trabalha em BGR, mas o Streamlit/Web trabalha em RGB.
    # Precisamos converter a imagem antes de exportar!
    frame_final_rgb = cv2.cvtColor(frame_exibicao, cv2.COLOR_BGR2RGB)
    
    return frame_final_rgb, respostas_finais, texto_status, sucesso