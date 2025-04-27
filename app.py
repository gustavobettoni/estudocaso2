import gradio as gr
from transformers import pipeline
import re
import ast
import operator
import math
from functools import lru_cache

# Dicionário de operadores seguros
SAFE_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '^': operator.pow,
    '√': math.sqrt
}

# Cache para o modelo
@lru_cache(maxsize=1)
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device="cpu"
    )

math_solver = load_model()

def safe_calculate(operation):
    """Calcula tradicionalmente quando possível"""
    try:
        # Pré-processamento
        operation = operation.replace('^', '**').replace('√', 'math.sqrt')
        
        # Verificação de segurança
        allowed_chars = set("0123456789+-*/.()%^√ math")
        if not all(c in allowed_chars for c in operation):
            raise ValueError("Caracteres inválidos")
            
        # Cálculo seguro
        result = eval(operation, {"__builtins__": None}, SAFE_OPERATORS)
        return str(round(result, 4)) if isinstance(result, float) else str(result)
        
    except:
        return None  # Indica que deve usar o modelo

def solve_with_ai(operation):
    """Usa o modelo quando o cálculo tradicional falha"""
    try:
        prompt = f"Resolva passo a passo e dê apenas o número final: {operation}"
        result = math_solver(prompt, max_length=30)[0]['generated_text']
        match = re.search(r"[-+]?\d*\.?\d+", result.replace(",", ""))
        return match.group(0) if match else "Erro: Formato inválido"
    except:
        return "Erro: Não foi possível calcular"

def solve_math(operation):
    # Primeiro tenta cálculo tradicional
    traditional_result = safe_calculate(operation)
    if traditional_result is not None:
        return traditional_result
        
    # Se falhar, usa o modelo AI
    ai_result = solve_with_ai(operation)
    
    # Verificação cruzada
    try:
        if abs(float(ai_result) - float(safe_calculate(ai_result))) > 0.1:
            return "Erro: Resultado inconsistente"
        return ai_result
    except:
        return ai_result

# Interface otimizada
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🔢 Assistente Matématico com IA")
    gr.Markdown("Combina IA com cálculos tradicionais para máxima precisão de resultados")
    
    with gr.Row():
        operation = gr.Textbox(label="Digite a operação", 
                             placeholder="Ex: (3+5)*2, √16, 2^5")
        result = gr.Textbox(label="Resultado")
    
    solve_btn = gr.Button("Calcular", variant="primary")
    solve_btn.click(fn=solve_math, inputs=operation, outputs=result)
    
    # Exemplos testados
    examples = gr.Examples(
        examples=[
            ["3 + 3"],
            ["5 * 4"],
            ["10 / 3"],
            ["(4 + 6) * 2.5"]
        ],
        inputs=operation,
        outputs=result,
        fn=solve_math,
        cache_examples=True
    )

app.launch()
