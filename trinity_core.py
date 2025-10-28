---

## 🧩 trinity_core.py
```python
from typing import Dict, Optional
import os

class LLMProvider:
    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai
            self._client = openai.OpenAI()
            self.model = model
            self._ok = True
        except Exception:
            self._ok = False

    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 800) -> str:
        if not self._ok:
            return f"[FAKE OUTPUT]\n{prompt[:200]}"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

GEN = "Topic:{topic}\\nGoal:{goal}\\nConstraints:{constraints}\\nGenerate 5 approaches."
OPP = "Oppose the following:\\n{generated}\\nList tensions, risks, and top 2 approaches."
SYN = "Fuse these:\\n{opposed}\\nReturn a final plan, rationale, metrics, and risks."

def run_trinity_loop(topic: str, goal: str="clarity", constraints: str="realistic",
                     provider: Optional[LLMProvider]=None, temperature: float=0.7) -> Dict[str,str]:
    provider = provider or OpenAIProvider()
    gen = provider.complete(GEN.format(topic=topic, goal=goal, constraints=constraints))
    opp = provider.complete(OPP.format(generated=gen))
    syn = provider.complete(SYN.format(opposed=opp), temperature=0.5)
    return {"generate": gen, "oppose": opp, "synthesize": syn}  
