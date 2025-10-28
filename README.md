# trinity-5.81
trinity mind a modular loop for idea generation , oppositional testing and fusion synthesis. 
# Trinity Mind // Moonlander Mode

A **recursive, comparative, synthesis-based reasoning engine** for modern AI workflows.  
Core loop: **Generate → Oppose → Synthesize**.

## ✨ What it does
- **Thought Experiment Generator:** creates diverse approaches, surfaces assumptions.
- **Oppositional Method Mapper:** stress-tests ideas with objections & alternatives.
- **Result Fusion Synthesizer:** fuses the strongest elements into an actionable plan.

## 🚀 Quickstart (FastAPI)
```bash
# install
pip install fastapi uvicorn pydantic openai

# run (from repo root)
uvicorn api.main:app --reload --port 8000

# open the docs
# http://localhost:8000/docs

## 🛰️ Moonlander Mode (CLI)
```bash
# run a single Trinity loop from the terminal
python trinity_core.py --topic "AI Safety" --goal "roadmap" --constraints "deploy in 6 months"

# emit structured JSON instead of formatted text
python trinity_core.py --topic "AI Safety" --format json
```
