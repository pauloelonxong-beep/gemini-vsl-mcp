import os
import json
import time
import tempfile
import httpx
import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────
# PROMPT FIXO DE ANÁLISE FORENSE DE VSL
# ─────────────────────────────────────────────
VSL_ANALYSIS_PROMPT = """
Você é um analista forense especializado em VSLs (Video Sales Letters) de direct response,
com foco em desconstrução visual completa de campanhas de nutraceuticos e saúde.

Antes de preencher qualquer campo, assista o vídeo completo pelo menos uma vez.
Depois preencha TODOS os campos abaixo com máxima precisão visual.

REGRAS ABSOLUTAS:
- Não invente. Se não conseguir identificar algo, escreva "Não identificado".
- Separe rigorosamente o que você VÊ do que você ouve/escuta.
- Se um elemento aparece múltiplas vezes, registre cada ocorrência.
- Timestamps são obrigatórios onde solicitado.
- Responda em PORTUGUÊS. Mantenha o formato exato.

---

## BLOCO 0 — ANÁLISE INICIAL (faça isso antes de preencher o resto)

Antes de categorizar qualquer elemento, responda:

1. Qual é a SENSAÇÃO GERAL do vídeo nos primeiros 10 segundos?
2. Qual é o FORMATO PREDOMINANTE da VSL?
3. Qual é a QUALIDADE DE PRODUÇÃO percebida? (1-10, sendo 10 = campanha de $50M+)
4. Qual é a DURAÇÃO TOTAL do vídeo?
5. Existe algum elemento visual imediatamente incomum ou inesperado?

---

## BLOCO 1 — PERSONAGENS E FIGURAS DE AUTORIDADE

### 1.1 Personagem Principal (Herói/Avatar)
- Nome (se mencionado):
- Gênero:
- Idade aparente:
- Etnia aparente:
- Roupa/aparência física:
- Ambiente em que aparece:
- Papel na narrativa (paciente, médico, pesquisador, mãe, consumidora, etc.):
- Energia/emoção transmitida (calmo, urgente, esperançoso, sofrendo, etc.):
- Aparece em câmera ou é só voz? [ ] Câmera [ ] Só voz [ ] Animação/avatar

### 1.2 Figura de Autoridade Principal
- Nome completo (se mencionado):
- Título/Credencial exata como aparece na tela:
- Aparência física (jaleco, terno, roupa casual, etc.):
- Aparece fisicamente no vídeo? [ ] Sim [ ] Não — só mencionado
- Se aparece: timestamps exatos de cada cena com essa figura:
- Como é introduzida? (corte direto, mencionada antes de aparecer, aparece sem apresentação):
- Parece gravação exclusiva para a VSL ou clipe reutilizado/extraído de outro conteúdo?
  [ ] Gravação exclusiva  [ ] Clipe externo  [ ] Não identificado
- Se clipe externo: de onde parece vir? (programa de TV, entrevista, palestra, etc.):

### 1.3 Outras Figuras de Autoridade ou Celebridades
Para CADA figura adicional, preencha:

**Figura #___:**
- Nome:
- Papel/título:
- Contexto de aparição (foto, vídeo, clipe de TV, menção em texto na tela):
- Timestamp:
- É celebridade reconhecível? [ ] Sim — quem: ___ [ ] Não [ ] Não identificado
- Parece clipe extraído sem autorização? [ ] Sim [ ] Não [ ] Não identificado
- Aparece junto a logotipo de programa/canal/instituição? [ ] Sim — qual: ___ [ ] Não

### 1.4 Depoimentos de Clientes/Pacientes
- Existem depoimentos? [ ] Sim [ ] Não
- Quantidade total:
- Formato de cada depoimento:
  [ ] Vídeo real com pessoa falando
  [ ] Foto + texto sobreposto
  [ ] Voz sobreposta sobre imagem (sem rosto)
  [ ] Print de review/comentário (Amazon, redes sociais, etc.)
  [ ] Texto animado na tela
- Para CADA depoimento em vídeo real:
  - Timestamp:
  - Gênero/idade aparente:
  - Ambiente do depoimento (casa, consultório, ao ar livre):
  - Parece gravação espontânea ou produzida?
  - Aparece nome/credencial na tela?

---

## BLOCO 2 — FORMATO E CENÁRIO DA VSL

### 2.1 Formato Geral
Marque o formato PREDOMINANTE e descreva se há combinação:

- [ ] Solo — uma pessoa falando diretamente para câmera
- [ ] Entrevista — duas ou mais pessoas conversando
- [ ] News-style — âncora + estilo de reportagem jornalística
- [ ] Podcast-style — conversa casual entre duas ou mais pessoas
- [ ] Slides com narração — sem rosto aparente
- [ ] Slides com picture-in-picture — rosto pequeno no canto
- [ ] Documental — narrador + imagens de apoio + entrevistados
- [ ] Animação / Motion graphics
- [ ] Misto — descreva:

Se o formato MUDA ao longo do vídeo, registre:
| Timestamp | Formato nesse trecho |
|---|---|
| | |

### 2.2 Cenário Principal e Mudanças
Para CADA cenário diferente que aparece:

**Cenário #___:**
- Tipo (consultório, cozinha, laboratório, estúdio de notícias, exterior, etc.):
- Parece set produzido ou local real?
- Timestamp de entrada e saída:
- Iluminação (natural, artificial estúdio, dramática, íntima):
- Elementos de fundo visíveis (livros, equipamentos, natureza, etc.):

### 2.3 Estilo de Câmera e Edição

**Ritmo de edição:**
- [ ] Cortes muito rápidos (estilo TikTok/Reels — menos de 3 segundos por plano)
- [ ] Cortes médios (3-8 segundos por plano)
- [ ] Tomadas longas (mais de 8 segundos — ritmo conversacional)
- [ ] Misto — descreva onde acelera e onde desacelera:

**Movimento de câmera:**
- [ ] Câmera estática
- [ ] Handheld (câmera na mão — sensação casual/documental)
- [ ] Zoom dramático
- [ ] Traveling (câmera em movimento)
- [ ] Misto:

---

## BLOCO 3 — INSERTS, B-ROLL E IMAGENS DE APOIO

> Esta seção é CRÍTICA. Liste CADA insert que aparece no vídeo, mesmo que brevemente.

### 3.1 Classificação de B-roll

Para cada grupo de imagens de apoio, identifique:

**Insert #___:**
- Timestamp:
- O que mostra (descreva com precisão):
- Tipo:
  [ ] Banco de imagens genérico (iStock, Shutterstock, Getty — parece comercial/artificial)
  [ ] Imagem/vídeo de estudo científico real
  [ ] Clipe extraído de programa de TV / noticiário / documentário
  [ ] Clipe de entrevista ou palestra externa
  [ ] Animação/infográfico produzido para a VSL
  [ ] Antes/depois fotográfico
  [ ] Screenshot de redes sociais / reviews / site
  [ ] Print de paper científico / journal / publicação
  [ ] Imagem de produto / embalagem
  [ ] Imagem de celebridade / figura pública
  [ ] Captação exclusiva (parece produzido especificamente para a VSL)
  [ ] Não identificado
- Parece INCOMUM para o nicho? Explique:

### 3.2 News Clips e Clipes de Programas de TV
- Existem clipes de noticiários ou programas? [ ] Sim [ ] Não
- Para cada um:
  - Canal/programa identificável (CNN, Fox, NBC, BBC, etc.):
  - Timestamp:
  - Conteúdo do clipe (o que está sendo dito/mostrado):
  - Parece uso autorizado ou extração não autorizada?
  - Logotipo do canal aparece? [ ] Sim [ ] Não

### 3.3 Uso de Celebridades ou Figuras Públicas em Inserts
- Aparecem celebridades, políticos, médicos famosos, influencers em B-roll/inserts? [ ] Sim [ ] Não
- Para cada aparição:
  - Nome/identidade:
  - Timestamp:
  - Contexto (clipe de entrevista, foto, menção em texto, etc.):
  - A VSL sugere endosso dessa figura? [ ] Sim [ ] Não
  - Parece endosso real ou uso não autorizado?

### 3.4 Imagens Científicas e de Laboratório
- Aparecem imagens de microscópio, DNA, células, laboratório? [ ] Sim [ ] Não
- Para cada uma:
  - Timestamp:
  - O que mostra especificamente:
  - Parece imagem real de estudo ou banco de imagens?
  - Há identificação da fonte visível na tela?

---

## BLOCO 4 — TEXTO NA TELA E ELEMENTOS GRÁFICOS

### 4.1 Headlines e Texto de Alto Destaque
Liste CADA headline ou texto grande que aparece na tela (em ordem cronológica):

| Timestamp | Texto exato | Cor/estilo de destaque |
|---|---|---|
| | | |

### 4.2 Números e Dados em Destaque Visual
Liste CADA número que aparece com destaque visual:
- Número:
- Contexto (% de desconto, número de estudos, anos de pesquisa, quantidade de usuários, etc.):
- Timestamp:
- Fonte aparece na tela? [ ] Sim — qual: ___ [ ] Não

### 4.3 Logos e Marcas Institucionais
Liste CADA logo ou nome de instituição que aparece na tela:
- Instituição/marca:
- Timestamp:
- Contexto de aparição (validação, parceiro, mídia, etc.):
- Parece endosso real ou uso oportunista?

### 4.4 Lower Thirds e Identificações
- Aparecem identificações sobrepostas (nome + título abaixo da pessoa falando)?
- Para cada uma: texto exato / timestamp / parece profissional ou improvisado?

---

## BLOCO 5 — PROVAS VISUAIS (ANÁLISE DETALHADA)

### 5.1 Estudos Científicos Mostrados Visualmente
- [ ] Sim [ ] Não
- Para cada estudo mostrado:
  - Timestamp:
  - Nome do journal visível?
  - Título do paper visível?
  - Número específico/dado em destaque?
  - É imagem real do paper ou infográfico produzido?
  - Duração que fica na tela (segundos):

### 5.2 Animações de Mecanismo (Como Funciona no Corpo)
- [ ] Sim [ ] Não
- Para cada animação:
  - Timestamp de início e fim:
  - O que está sendo animado (órgão, célula, processo, etc.):
  - Estilo da animação (2D simples, 3D realista, infográfico flat, outro):
  - Parece produção cara ou template genérico?

### 5.3 Fotos de Antes/Depois
- [ ] Sim [ ] Não
- Quantidade de pares antes/depois:
- Para cada par:
  - Timestamp:
  - Área do corpo/resultado mostrado:
  - Parece foto real ou stock/editado?
  - Nome/identificação da pessoa aparece?

### 5.4 Demonstrações ao Vivo ou Testes
- [ ] Sim [ ] Não
- Para cada demonstração:
  - Timestamp:
  - O que está sendo demonstrado:
  - Parece real ou encenado?

### 5.5 Screenshots de Redes Sociais / Reviews
- [ ] Sim [ ] Não
- Para cada screenshot:
  - Timestamp:
  - Plataforma (Facebook, Amazon, Trustpilot, etc.):
  - Estrelas / avaliação visível?
  - Nome do usuário aparece?

---

## BLOCO 6 — PRODUTO E OFERTA (VISUAL)

### 6.1 Aparição do Produto
- Produto aparece visualmente? [ ] Sim [ ] Não
- Primeiro timestamp de aparição:
- Formatos de aparição:
  [ ] Foto do frasco/embalagem
  [ ] Mockup 3D
  [ ] Pessoa segurando o produto
  [ ] Produto no contexto de uso
  [ ] Ingredientes individuais mostrados
  [ ] Diagrama/infográfico do produto
- Nome do produto aparece em destaque? [ ] Sim [ ] Não — quando:

### 6.2 Tela de Oferta / Checkout
- Preço mostrado na tela? [ ] Sim [ ] Não — valor exato:
- Preço "riscado"? [ ] Sim [ ] Não
- Pacotes mostrados? [ ] Sim [ ] Não
  - Layout dos pacotes:
  - Qual pacote está destacado como "mais popular"?
- Bônus mostrados visualmente? [ ] Sim [ ] Não — quais:
- Garantia mostrada? [ ] Sim [ ] Não
  - Formato visual (selo, badge, texto, ícone, dias):
- Urgência visual? [ ] Sim [ ] Não
  - Tipo (timer, "somente X unidades", "oferta expira"):
  - Parece urgência real ou artificial?
- Botão de compra visível? [ ] Sim [ ] Não — texto exato no botão:

---

## BLOCO 7 — MAPA VISUAL DA VSL (TIMELINE COMPLETA)

| Bloco | Timestamp | O que aparece na tela | Elemento visual mais marcante |
|---|---|---|---|
| Pre-hook / Gancho Visual | 0:00 – | | |
| Hook Principal | – | | |
| Lead / Promessa | – | | |
| Background Story | – | | |
| Emotional Story | – | | |
| Discovery Story | – | | |
| Mecanismo do Problema | – | | |
| Inviabilização (caminho antigo) | – | | |
| Teste "descubra você mesmo" | – | | |
| Mecanismo da Solução | – | | |
| Demonstração Visual | – | | |
| Product Buildup | – | | |
| Depoimentos | – | | |
| Oferta / Close | – | | |
| FAQ | – | | |
| CTA Final | – | | |

---

## BLOCO 8 — ANÁLISE DE PRODUÇÃO E RARIDADES

### 8.1 Qualidade de Produção
- Orçamento estimado: [ ] Baixo (<$5k) [ ] Médio ($5k-$50k) [ ] Alto ($50k+)
- Justificativa:

### 8.2 Elementos Visuais Incomuns ou Raros para o Nicho
- Elemento:
- Por que é incomum:
- Timestamp:

### 8.3 Técnicas de Retenção Visual
- [ ] Texto na tela acompanhando a narração
- [ ] Mudança frequente de cenário/ângulo
- [ ] Cliffhangers visuais
- [ ] Contador visual ou barra de progresso
- [ ] Mudança de ritmo proposital
- [ ] Outro:

### 8.4 Consistência Visual da VSL
- O estilo visual é consistente do início ao fim? [ ] Sim [ ] Não
- Se não, onde e como muda:
- Há elementos que parecem adicionados depois?

### 8.5 Avaliação Final
- **Qual bloco visual tem o maior impacto emocional?**
- **Qual prova visual é a mais forte da VSL?**
- **Qual elemento visual você usaria como referência para replicar?**
- **O que está visivelmente fraco ou incoerente?**
"""


# ─────────────────────────────────────────────
# MCP ENDPOINT
# ─────────────────────────────────────────────
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    body = await request.json()
    method = body.get("method")
    params = body.get("params", {})
    req_id = body.get("id")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "gemini-vsl-analyzer", "version": "2.0.0"}
            }
        })

    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "analyze_vsl_video",
                        "description": (
                            "Sends a VSL video to Gemini 2.5 Pro for full forensic analysis. "
                            "Extracts visual elements, characters, authority figures, editing, "
                            "b-roll, on-screen text, proof elements, offer structure, and production quality. "
                            "Pass the Google Drive direct download URL of the video."
                        ),
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "video_url": {
                                    "type": "string",
                                    "description": "Google Drive shareable link or direct video URL (mp4/mov)"
                                }
                            },
                            "required": ["video_url"]
                        }
                    }
                ]
            }
        })

    if method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "analyze_vsl_video":
            try:
                result = await analyze_vsl(video_url=args["video_url"])
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result}]
                    }
                })
            except Exception as e:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Erro: {str(e)}"}],
                        "isError": True
                    }
                })

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"}
    })


# ─────────────────────────────────────────────
# CORE: download → Gemini File API → analyze
# ─────────────────────────────────────────────
async def analyze_vsl(video_url: str) -> str:
    if not GEMINI_API_KEY:
        raise Exception("GEMINI_API_KEY não configurada no servidor")

    genai.configure(api_key=GEMINI_API_KEY)

    if "drive.google.com" in video_url:
        video_url = drive_to_direct(video_url)

    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "video/mp4")

    suffix = ".mov" if "mov" in content_type else ".mp4"
    mime = "video/quicktime" if suffix == ".mov" else "video/mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(response.content)
        tmp_path = f.name

    video_file = genai.upload_file(tmp_path, mime_type=mime)

    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise Exception("Gemini File API falhou ao processar o vídeo")

    model = genai.GenerativeModel("gemini-2.5-pro")
    result = model.generate_content([video_file, VSL_ANALYSIS_PROMPT])

    os.unlink(tmp_path)

    return result.text


def drive_to_direct(url: str) -> str:
    if "/file/d/" in url:
        file_id = url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        file_id = url.split("id=")[1].split("&")[0]
    else:
        raise ValueError("Não foi possível extrair o ID do Google Drive da URL")
    return f"https://drive.google.com/uc?export=download&id={file_id}"
