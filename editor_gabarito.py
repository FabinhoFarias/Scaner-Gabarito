import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import sys

# --- CONFIGURAÇÃO ---
NUM_QUESTOES = 90
QUESTOES_POR_ABA = 15
ALTERNativas = ['A', 'B', 'C', 'D', 'E', 'X']

# --- DADOS DE EXEMPLO (Substitua pela leitura real do seu arquivo JSON) ---
# Este dicionário será populado a partir do JSON de entrada
respostas_carregadas = {}
caminho_arquivo_json = 'respostas_detectadas.json'

# Função para criar um arquivo JSON de exemplo se ele não existir
def criar_json_exemplo(caminho):
    respostas = {str(i): 'A' if i % 5 == 1 else 'X' for i in range(1, NUM_QUESTOES + 1)}
    with open(caminho, 'w') as f:
        json.dump(respostas, f, indent=4)
    print(f"Arquivo de exemplo criado: {caminho}")

# --- FUNÇÕES DE LÓGICA DO FORMULÁRIO ---

def carregar_respostas(caminho):
    """Carrega as respostas do arquivo JSON."""
    global respostas_carregadas
    try:
        with open(caminho, 'r') as f:
            data = json.load(f)
            # Converte chaves de string para int, se necessário
            respostas_carregadas = {int(k): v for k, v in data.items()}
            return True
    except FileNotFoundError:
        messagebox.showerror("Erro", f"Arquivo JSON não encontrado: {caminho}")
        criar_json_exemplo(caminho)
        return carregar_respostas(caminho) # Tenta carregar o exemplo
    except json.JSONDecodeError:
        messagebox.showerror("Erro", "Formato JSON inválido.")
        return False
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao carregar o JSON: {e}")
        return False

def salvar_respostas(caminho):
    """Salva as respostas editadas de volta para o arquivo JSON."""
    global respostas_carregadas
    
    # Garantir que todas as 90 questões estão no dicionário, mesmo as não editadas
    respostas_para_salvar = {str(q): respostas_carregadas.get(q, 'X') for q in range(1, NUM_QUESTOES + 1)}
    
    try:
        with open(caminho, 'w') as f:
            json.dump(respostas_para_salvar, f, indent=4)
        messagebox.showinfo("Sucesso", f"Respostas salvas com sucesso em: {caminho}")
    except Exception as e:
        messagebox.showerror("Erro de Salvamento", f"Não foi possível salvar as respostas: {e}")

def criar_aba(notebook, root, start_q, end_q, frame_vars):
    """Cria uma aba de formulário para um intervalo de questões."""
    frame = ttk.Frame(notebook, padding="10")
    notebook.add(frame, text=f'Q{start_q}-Q{end_q}')

    # Configurar cabeçalho da tabela
    ttk.Label(frame, text="Questão", width=8, font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5)
    for i, alt in enumerate(ALTERNativas):
        ttk.Label(frame, text=alt, width=3, font=('Arial', 10, 'bold')).grid(row=0, column=i+1, padx=5, pady=5)

    # Criar campos de rádio para cada questão na aba
    for q_num in range(start_q, end_q + 1):
        if q_num > NUM_QUESTOES:
            break
            
        row_idx = q_num - start_q + 1
        
        # 1. Label da Questão
        ttk.Label(frame, text=f"Q{q_num}:", width=8, anchor='w').grid(row=row_idx, column=0, padx=5, pady=2)
        
        # 2. Variável de Controle
        # Armazena a alternativa selecionada para a questão atual
        var = tk.StringVar(root)
        
        # 3. Define o valor inicial baseado nas respostas carregadas
        valor_inicial = respostas_carregadas.get(q_num, 'X')
        var.set(valor_inicial)
        
        # 4. Função de rastreamento para atualizar o dicionário global
        def update_resposta(var_trace, q_num=q_num):
            # Obtém o valor mais recente (a alternativa selecionada)
            respostas_carregadas[q_num] = var_trace.get()

        # Configura o rastreamento da variável (listener)
        var.trace_add('write', lambda name, index, mode, var=var, q=q_num: update_resposta(var, q))

        # 5. Cria Radiobuttons para as Alternativas
        for i, alt in enumerate(ALTERNativas):
            rb = ttk.Radiobutton(frame, 
                                 variable=var, 
                                 value=alt, 
                                 text="", # O texto é a label do cabeçalho
                                 width=3)
            rb.grid(row=row_idx, column=i+1, padx=2, pady=2)
            
        # Armazena a variável para evitar que o garbage collector a exclua
        frame_vars.append(var)


def iniciar_formulario():
    """Configura e inicia a janela principal do formulário."""
    # 1. Carrega os dados
    if not os.path.exists(caminho_arquivo_json):
        criar_json_exemplo(caminho_arquivo_json)
        
    if not carregar_respostas(caminho_arquivo_json):
        return

    # 2. Configuração da Janela
    root = tk.Tk()
    root.title(f"Revisão de Gabarito OMR ({NUM_QUESTOES} Questões)")

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, padx=10, expand=True, fill="both")
    
    # Lista para manter referências às variáveis (evita GC)
    frame_vars = [] 

    # 3. Criação das Abas (15 Questões por Aba)
    for i in range(0, NUM_QUESTOES, QUESTOES_POR_ABA):
        start_q = i + 1
        end_q = i + QUESTOES_POR_ABA
        criar_aba(notebook, root, start_q, end_q, frame_vars)

    # 4. Botão de Salvar
    btn_salvar = ttk.Button(root, text="Salvar Alterações no JSON", 
                            command=lambda: salvar_respostas(caminho_arquivo_json))
    btn_salvar.pack(pady=10)
    
    # 5. Loop Principal
    root.mainloop()

# --- EXECUÇÃO ---
iniciar_formulario()