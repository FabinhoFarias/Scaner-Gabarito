import cv2
import numpy as np

LARGURA_ALVO = 1200

def processar_gabarito(imagem_bytes):
    nparr = np.frombuffer(imagem_bytes, np.uint8)
    img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_original is None:
        return None, None, "Erro ao decodificar a imagem."

    proporcao = LARGURA_ALVO / float(img_original.shape[1])
    ALTURA_ALVO = int(img_original.shape[0] * proporcao)
    img = cv2.resize(img_original, (LARGURA_ALVO, ALTURA_ALVO))
    img_debug = img.copy()

    # 1. ABORDAGEM DE MITIGAÇÃO DE ILUMINAÇÃO (Preto e Branco Puro via Otsu Binarization)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu calcula automaticamente o melhor threshold para a folha ignorando sombras suaves
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Detectar todas as âncoras candidatas
    contornos, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    ancoras = []
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0: continue
        
        razao_aspecto = w / float(h)
        # Filtro geométrico baseado estritamente na caixa externa (w, h)
        if (12 <= w <= 35) and (12 <= h <= 35) and (0.7 < razao_aspecto < 1.4):
            ancoras.append((x + w // 2, y + h // 2))
            # Desenha marcação azul nas detectadas para validação visual no Streamlit
            cv2.rectangle(img_debug, (x, y), (x + w, y + h), (255, 0, 0), 1)

    if len(ancoras) < 20:
        return None, None, f"Falha ao isolar marcas em P&B. Detectadas apenas {len(ancoras)}."

    # 3. Separar as âncoras em 4 quadrantes anatômicos
    limite_topo = int(ALTURA_ALVO * 0.20)
    limite_base = int(ALTURA_ALVO * 0.80)
    limite_esquerda = int(LARGURA_ALVO * 0.15)
    limite_direita = int(LARGURA_ALVO * 0.85)

    topo_pts = []
    base_pts = []
    esquerda_pts = []
    direita_pts = []

    for pt in ancoras:
        cx, cy = pt
        if cy < limite_topo:
            topo_pts.append(pt)
        elif cy > limite_base:
            base_pts.append(pt)
        elif cx < limite_esquerda:
            esquerda_pts.append(pt)
        elif cx > limite_direita:
            direita_pts.append(pt)

    # Ordenação dos eixos de leitura
    topo_pts = sorted(topo_pts, key=lambda p: p[0])
    base_pts = sorted(base_pts, key=lambda p: p[0])
    esquerda_pts = sorted(esquerda_pts, key=lambda p: p[1])
    direita_pts = sorted(direita_pts, key=lambda p: p[1])

    # 4. RECONSTRUÇÃO ESTREITA DAS TRAVES (30 colunas no topo/base, 18 linhas nas laterais)
    if len(topo_pts) != 30 or len(base_pts) != 30:
        min_tx, max_tx = (topo_pts[0][0], topo_pts[-1][0]) if topo_pts else (int(LARGURA_ALVO*0.13), int(LARGURA_ALVO*0.87))
        min_bx, max_bx = (base_pts[0][0], base_pts[-1][0]) if base_pts else (int(LARGURA_ALVO*0.13), int(LARGURA_ALVO*0.87))
        ty = topo_pts[0][1] if topo_pts else int(ALTURA_ALVO*0.04)
        by = base_pts[0][1] if base_pts else int(ALTURA_ALVO*0.96)
        
        topo_pts = [(int(x), ty) for x in np.linspace(min_tx, max_tx, 30)]
        base_pts = [(int(x), by) for x in np.linspace(min_bx, max_bx, 30)]

    # Correção crucial: As laterais possuem 18 marcas na folha física real, não 15
    if len(esquerda_pts) != 18 or len(direita_pts) != 18:
        min_ly, max_ly = (esquerda_pts[0][1],供esquerda_pts[-1][1]) if esquerda_pts else (int(ALTURA_ALVO*0.09), int(ALTURA_ALVO*0.91))
        min_ry, max_ry = (direita_pts[0][1], direita_pts[-1][1]) if direita_pts else (int(ALTURA_ALVO*0.09), int(ALTURA_ALVO*0.91))
        lx = esquerda_pts[0][0] if esquerda_pts else int(LARGURA_ALVO*0.04)
        rx = direita_pts[0][0] if direita_pts else int(LARGURA_ALVO*0.96)
        
        esquerda_pts = [(lx, int(y)) for y in np.linspace(min_ly, max_ly, 18)]
        direita_pts = [(rx, int(y)) for y in np.linspace(min_ry, max_ry, 18)]

    # Desenha as linhas verticais (30 colunas)
    for i in range(30):
        cv2.line(img_debug, topo_pts[i], base_pts[i], (0, 0, 255), 1)
        
    # Desenha as linhas horizontais (18 linhas)
    for i in range(18):
        cv2.line(img_debug, esquerda_pts[i], direita_pts[i], (0, 0, 255), 1)

    # 5. CRUZAMENTO DE INTERSEÇÕES E VARREDURA DAS QUESTÕES
    # Ignoramos as primeiras 3 linhas do topo (cabeçalhos) para mapear exatamente as 15 de questões
    respostas = {}
    letras = ['A', 'B', 'C', 'D', 'E']
    raio_analise = 6

    for q_idx in range(15):
        # Deslocamos o mapeamento para ignorar os blocos iniciais vazios de cima (linhas 0, 1, 2)
        p_esq = esquerda_pts[q_idx + 3]
        p_dir = direita_pts[q_idx + 3]
        
        for b_idx in range(6):
            nq = 91 + (b_idx * 15) + q_idx
            valores_preenchimento = []
            
            for alt_idx in range(5):
                col_idx = (b_idx * 5) + alt_idx
                p_top = topo_pts[col_idx]
                p_bas = base_pts[col_idx]
                
                # Interseção matemática das retas concorrentes
                A1 = p_dir[1] - p_esq[1]
                B1 = p_esq[0] - p_dir[0]
                C1 = A1 * p_esq[0] + B1 * p_esq[1]
                
                A2 = p_bas[1] - p_top[1]
                B2 = p_top[0] - p_bas[0]
                C2 = A2 * p_top[0] + B2 * p_top[1]
                
                det = A1 * B2 - A2 * B1
                if det != 0:
                    cx = int((B2 * C1 - B1 * C2) / det)
                    cy = int((A1 * C2 - A2 * C1) / det)
                else:
                    cx, cy = p_top[0], p_esq[1]
                
                cv2.circle(img_debug, (cx, cy), 3, (0, 255, 0), -1)
                
                recorte = thresh[cy-raio_analise : cy+raio_analise, cx-raio_analise : cx+raio_analise]
                if recorte.size == 0:
                    valores_preenchimento.append(0)
                    continue
                valores_preenchimento.append(cv2.countNonZero(recorte))
            
            max_p = max(valores_preenchimento)
            idx = valores_preenchimento.index(max_p)
            media = np.mean(valores_preenchimento)
            
            # Ajuste de sensibilidade pós-Otsu
            if max_p > 35 and max_p > (media * 1.4):
                duplicados = sum(1 for v in valores_preenchimento if v > (max_p * 0.75))
                respostas[nq] = "RASURA" if duplicados > 1 else letras[idx]
            else:
                respostas[nq] = "NULA"

    return respostas, img_debug, "Gabarito decodificado com threshold global estável."