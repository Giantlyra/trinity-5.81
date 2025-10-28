"""Core reasoning loop for the Trinity Mind engine.

This module implements the three phase reasoning pipeline used across the
project.  It intentionally keeps the interface small so it can be imported by a
FastAPI service, a CLI script, or unit tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional


GEN_TEMPLATE = (
    "Topic: {topic}\n"
    "Goal: {goal}\n"
    "Constraints: {constraints}\n"
    "Generate 5 approaches."
)
OPP_TEMPLATE = (
    "Oppose the following:\n"
    "{generated}\n"
    "List tensions, risks, and the top 2 approaches."
)
SYN_TEMPLATE = (
    "Fuse these perspectives:\n"
    "{opposed}\n"
    "Return a final plan, rationale, metrics, and risks."
)


class LLMProvider:
    """Abstract provider used to fetch model completions."""

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 800) -> str:
        """Return a model completion for ``prompt``.

        Concrete subclasses are expected to implement this and raise a
        ``NotImplementedError`` when the model cannot be accessed.
        """

        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """Provider backed by the OpenAI Chat Completions API.

    The constructor attempts to import :mod:`openai`.  When this fails (for
    example in CI or an air-gapped environment) we fall back to an extremely
    simple echo implementation.  This allows the rest of the codebase to run in
    tests without requiring network access or credentials.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = None
        try:
            import openai  # type: ignore

            self._client = openai.OpenAI()
            self._ok = True
        except Exception:
            # The SDK is not available – switch to a degraded but deterministic
            # fallback so the engine is still usable during tests.
            self._ok = False

    def complete(
        self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 800
    ) -> str:
        if not self._ok or self._client is None:
            header = "[OFFLINE COMPLETION]"
            return f"{header}\nPrompt: {prompt[:200]}"

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


@dataclass
class TrinityConfig:
    """Input configuration for a Trinity reasoning pass."""

    topic: str
    goal: str = "clarity"
    constraints: str = "realistic"


def run_trinity_loop(
    topic: str,
    goal: str = "clarity",
    constraints: str = "realistic",
    *,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """Execute the Generate → Oppose → Synthesize reasoning pipeline."""

    provider = provider or OpenAIProvider()

    gen_prompt = GEN_TEMPLATE.format(topic=topic, goal=goal, constraints=constraints)
    generated = provider.complete(gen_prompt, temperature=temperature)

    opp_prompt = OPP_TEMPLATE.format(generated=generated)
    opposed = provider.complete(opp_prompt, temperature=temperature)

    syn_prompt = SYN_TEMPLATE.format(opposed=opposed)
    synthesized = provider.complete(syn_prompt, temperature=max(0.0, temperature - 0.2))

    return {"generate": generated, "oppose": opposed, "synthesize": synthesized}


def boot_moonlander_mode(
    config: TrinityConfig,
    *,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """Convenience wrapper mirroring the README terminology.

    ``Moonlander Mode`` is simply a branded name for a single Trinity loop run.
    This helper makes it easy for CLIs or other tooling to communicate the
    concept without duplicating prompt wiring while still allowing
    temperature control and provider injection.
    """

    return run_trinity_loop(
        topic=config.topic,
        goal=config.goal,
        constraints=config.constraints,
        provider=provider,
        temperature=temperature,
    )


def _format_results(results: Dict[str, str]) -> str:
    sections = [
        "🚀 Trinity Mind // Moonlander Mode",
        "",
        "[Generate]",
        results["generate"],
        "",
        "[Oppose]",
        results["oppose"],
        "",
        "[Synthesize]",
        results["synthesize"],
    ]
    return "\n".join(sections)


def run_moonlander_cli(
    config: TrinityConfig,
    *,
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.7,
) -> Dict[str, str]:
    """Run a single Trinity loop suitable for CLI execution."""

    return boot_moonlander_mode(
        config,
        provider=provider,
        temperature=temperature,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Lightweight CLI for quick experiments.

    Example::

        python -m trinity_core --topic "AI Safety" --goal "roadmap"
    """

    import argparse

    parser = argparse.ArgumentParser(description="Boot Trinity Mind // Moonlander Mode")
    parser.add_argument("--topic", help="Topic the engine should explore")
    parser.add_argument("--goal", default="clarity", help="Desired outcome")
    parser.add_argument(
        "--constraints",
        default="realistic",
        help="Any constraints that should steer the reasoning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM backend",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Choose between rich text or JSON output",
    )
    args = parser.parse_args(argv)

    topic = args.topic
    if not topic:
        print("Boot Trinity Mind // Moonlander Mode")
        try:
            topic = input("Topic> ").strip()
        except EOFError as exc:
            raise SystemExit("No topic supplied") from exc
        if not topic:
            raise SystemExit("Topic is required to boot Moonlander Mode")

    config = TrinityConfig(topic=topic, goal=args.goal, constraints=args.constraints)
    results = run_moonlander_cli(
        config,
        temperature=args.temperature,
    )
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(_format_results(results))


if __name__ == "__main__":
    main()
