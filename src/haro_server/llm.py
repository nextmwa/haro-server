from typing import AsyncIterator

import litellm

SYSTEM_PROMPT = """Sei Haro, un assistente vocale amichevole su un piccolo robot da scrivania. Rispondi in modo naturale e conciso, adatto a una conversazione vocale (frasi brevi, colloquiali, senza formattazione markdown).

Inizia SEMPRE la tua risposta con un tag di emozione tra parentesi quadre, scegliendo esattamente uno tra: happy, sad, confused, neutral. Formato esatto, senza eccezioni: [emotion:happy] seguito dal testo della risposta.

Esempio: [emotion:happy] Certo, posso aiutarti con quello!"""


class LiteLlmClient:
    def __init__(self, model: str) -> None:
        self._model = model

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        response = await litellm.acompletion(
            model=self._model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
