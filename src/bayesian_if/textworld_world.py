"""TextWorld adapter — wraps textworld environments as a World."""

from __future__ import annotations

from bayesian_if.world import Observation, StateSnapshot


class TextWorldWorld:
    """World adapter for Microsoft TextWorld environments."""

    def __init__(self, game_file: str) -> None:
        try:
            import textworld
        except ImportError as e:
            raise ImportError("Install textworld: pip install bayesian-if[textworld]") from e

        self._tw = textworld
        request_infos = textworld.EnvInfos(
            admissible_commands=True,
            score=True,
            inventory=True,
            description=True,
            location=True,
            won=True,
            lost=True,
        )
        self.env = textworld.start(game_file, request_infos)
        self._last_infos: dict = {}
        self._game_file = game_file

    def reset(self) -> Observation:
        obs, infos = self.env.reset()
        self._last_infos = infos
        return self._make_observation(obs, infos)

    def step(self, action: str) -> tuple[Observation, float, bool]:
        obs, score, done, infos = self.env.step(action)
        prev_score = self._last_infos.get("score", 0)
        reward = float(score - prev_score)
        self._last_infos = infos
        return self._make_observation(obs, infos), reward, done

    def valid_actions(self) -> list[str]:
        return list(self._last_infos.get("admissible_commands", []))

    def save(self) -> StateSnapshot:
        return self.env.copy()

    def restore(self, snapshot: StateSnapshot) -> None:
        self.env = snapshot

    def _make_observation(self, text: str, infos: dict) -> Observation:
        location = infos.get("location", None)
        inv_text = infos.get("inventory", "")
        inventory = self._parse_inventory(inv_text) if inv_text else ()
        score = infos.get("score", 0)
        return Observation(text=text, score=score, location=location, inventory=inventory)

    @staticmethod
    def _parse_inventory(inv_text: str) -> tuple[str, ...]:
        items: list[str] = []
        for line in inv_text.strip().splitlines():
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                items.append(line.lstrip("-* ").strip())
        return tuple(items)
