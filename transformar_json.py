import json
import os

# --- CONFIGURAÇÃO ---
NUM_QUESTOES = 90
QUESTOES_POR_ABA = 15
ALTERNativas = ['A', 'B', 'C', 'D', 'E', 'X']

# --- DADOS DE EXEMPLO (Substitua pela leitura real do seu arquivo JSON) ---
# Este dicionário será populado a partir do JSON de entrada
caminho_arquivo_json = 'respostas_detectadas.json'

def carregar_json(caminho):
    """
    Abre e lê o conteúdo de um arquivo JSON.
    
    Args:
        caminho (str): O caminho completo para o arquivo JSON.
        
    Returns:
        Dict[int, Any] | None: O dicionário carregado (com chaves convertidas para int), 
                               ou None em caso de erro.
    """
    if not os.path.exists(caminho):
        print(f"❌ Erro: Arquivo JSON não encontrado no caminho: {caminho}")
        return None
        
    try:
        with open(caminho, 'r') as f:
            # json.load() é o método correto para LER (carregar) dados de um arquivo.
            data = json.load(f)
            
            # É comum que as chaves do JSON sejam strings ("1", "2", ...)
            # Convertemos as chaves para inteiros para facilitar o loop e o acesso:
            respostas_carregadas = {int(k): v for k, v in data.items()}
            
            print(f"✅ Arquivo JSON carregado com sucesso: {caminho}")
            return respostas_carregadas
            
    except json.JSONDecodeError:
        print(f"❌ Erro de Formato: O arquivo '{caminho}' não é um JSON válido.")
        return None
        
    except Exception as e:
        print(f"❌ Erro inesperado ao ler o JSON: {e}")
        return None

gabarito = carregar_json(caminho_arquivo_json)
STRINGGABARITO = ''

if gabarito is not None:
    # Use .items() para obter a chave (q_num) e o valor (resposta) em cada iteração
    for q_num, resposta in gabarito.items():
        STRINGGABARITO += resposta
        print(f"Questão {q_num}: Resposta -> {resposta}")

else:
    print("Não foi possível carregar o gabarito para iteração.")


print(STRINGGABARITO[:45].lower())
print(STRINGGABARITO[45:].lower())

print(len("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"))