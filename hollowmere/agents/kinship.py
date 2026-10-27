"""Who is family to whom, read from the family ties on each Persona (personas.json, and children born in the run)."""

from __future__ import annotations

from hollowmere.agents.persona import Persona

# what someone is to a relative, by their sex: (female, male, unknown)
WORDS = {
    "parent": ("mother", "father", "parent"),
    "child": ("daughter", "son", "child"),
    "sibling": ("sister", "brother", "sibling"),
    "spouse": ("wife", "husband", "spouse"),
    "grandparent": ("grandmother", "grandfather", "grandparent"),
    "grandchild": ("granddaughter", "grandson", "grandchild"),
    "uncle": ("aunt", "uncle", "aunt or uncle"),
    "niece": ("niece", "nephew", "niece or nephew"),
}


class Kinship:
    def __init__(self, personas: dict[str, Persona]):
        self.sex = {pid: p.sex for pid, p in personas.items()}
        self.parents: dict[str, set[str]] = {pid: set(p.parents) for pid, p in personas.items()}
        self.siblings = {frozenset((pid, other)) for pid, p in personas.items() for other in p.siblings}
        self.uncles = {(other, pid) for pid, p in personas.items() for other in p.aunt_or_uncle}  # (uncle, niece)
        self.spouses = {frozenset((pid, p.spouse)) for pid, p in personas.items() if p.spouse}

    def _word(self, role: str, who: str) -> str:
        female, male, neutral = WORDS[role]
        return {"female": female, "male": male}.get(self.sex.get(who, ""), neutral)

    def _siblings(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.siblings or bool(self.parents.get(a, set()) & self.parents.get(b, set()))

    def _uncle_of(self, who: str, of: str) -> bool:
        return (who, of) in self.uncles or any(self._siblings(who, p) for p in self.parents.get(of, set()))

    def related(self, a: str, b: str) -> bool:
        """Family by blood or marriage-to-blood, spouses excluded: the people who may not become lovers."""
        return frozenset((a, b)) not in self.spouses and self.what_is(a, b) is not None

    def what_is(self, who: str, of: str) -> str | None:
        """What `who` is to `of` ("mother", "niece", ...), or None when they are not family."""
        if who == of:
            return None
        if who in self.parents.get(of, set()):
            return self._word("parent", who)
        if of in self.parents.get(who, set()):
            return self._word("child", who)
        if frozenset((who, of)) in self.spouses:
            return self._word("spouse", who)
        if self._siblings(who, of):
            return self._word("sibling", who)
        if any(who in self.parents.get(p, set()) for p in self.parents.get(of, set())):
            return self._word("grandparent", who)
        if any(of in self.parents.get(p, set()) for p in self.parents.get(who, set())):
            return self._word("grandchild", who)
        if self._uncle_of(who, of):
            return self._word("uncle", who)
        if self._uncle_of(of, who):
            return self._word("niece", who)
        return None
