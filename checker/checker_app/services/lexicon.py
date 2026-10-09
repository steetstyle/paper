"""Lexicons of machine-flavoured phrasing, in Turkish and English.

These are *markers*, not proof. The signals exist because current instruction
tuned models fall into a small number of register habits:

* discourse scaffolding ("Ayrıca,", "Furthermore,", "In conclusion,")
* stacked hedges and intensifiers ("önemli ölçüde", "plays a crucial role")
* the presentative passive of generated academic Turkish ("sunulmaktadır")
* typography and formatting habits (em dashes, curly quotes, bold markers)
* a small set of favourite nouns ("landscape", "delve", "realm", "kapsamlı")

Every entry carries a weight and a severity. Weights are deliberately small and
additive: many weak markers are strong evidence, one strong marker is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from checker_app.domain.enums import Severity

__all__ = ["Rule", "build_rules"]


@dataclass(frozen=True, slots=True)
class Rule:
    """A phrase family and how much weight each hit carries."""

    code: str
    severity: Severity
    weight: float
    patterns: tuple[str, ...]
    languages: tuple[str, ...] = ("tr", "en")
    group: str = ""
    """Hits inside one group are counted once, so a sentence that says
    "crucial" three times does not triple its own evidence."""

    max_hits: int = 1
    sample_limit: int = 2
    compiled: tuple[re.Pattern[str], ...] = ()

    def matches(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in self.compiled:
            m = pattern.search(text)
            if m:
                found.append(m.group(0).strip())
        return found[: self.max_hits]


def _rule(
    code: str,
    weight: float,
    severity: Severity,
    patterns: tuple[str, ...],
    languages: tuple[str, ...] = ("tr", "en"),
    group: str = "",
    max_hits: int = 1,
) -> Rule:
    compiled = tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    return Rule(
        code=code,
        severity=severity,
        weight=weight,
        patterns=patterns,
        languages=languages,
        group=group or code,
        max_hits=max_hits,
        compiled=compiled,
    )


# --------------------------------------------------------------------- markers
_DISCOURSE_TR = (
    r"\bayrıca\b",
    r"\böte yandan\b",
    r"\bbununla birlikte\b",
    r"\bbuna ek olarak\b",
    r"\bsonuç olarak\b",
    r"\bgenel olarak\b",
    r"\böncelikle\b",
    r"\bbu bağlamda\b",
    r"\bbu doğrultuda\b",
    r"\bbu nedenle\b",
    r"\bbu yüzden\b",
    r"\bdiğer bir ifadeyle\b",
    r"\bönemli bir noktaya değinmektir\b",
    r"\bhatırlatmak gerekirse\b",
)
_DISCOURSE_EN = (
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\badditionally\b",
    r"\bin addition\b",
    r"\bin conclusion\b",
    r"\bin summary\b",
    r"\boverall\b",
    r"\bin this regard\b",
    r"\bin this context\b",
    r"\bit is (?:important|worth) (?:to note|noting|highlighting)\b",
    r"\bit should be noted that\b",
    r"\bas (?:a|an) .{0,30}(?:result|consequence)\b",
    r"\bby (?:analysing|analyzing|examining|leveraging)\b",
)
_HEDGES = (
    r"\bgenellikle\b",
    r"\bçoğu zaman\b",
    r"\btipik olarak\b",
    r"\bgeniş ölçüde\b",
    r"\bbüyük ölçüde\b",
    r"\bönemli ölçüde\b",
    r"\bgörece\b",
    r"\bneredeyse\b",
    r"\barguably\b",
    r"\bsomewhat\b",
    r"\bgenel olarak\b",
    r"\bnumerous\b",
    r"\bgenerally\b",
    r"\btypically\b",
    r"\boften\b",
    r"\busually\b",
    r"\blikely\b",
    r"\bpotentially\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuggests? that\b",
    r"\bappears? to\b",
    r"\bindicates? that\b",
)
_INTENSIFIERS = (
    r"\bson derece\b",
    r"\bçok önemli\b",
    r"\bhayati\b",
    r"\bkritik\b",
    r"\bvurgulanmalıdır\b",
    r"\bönemli bir rol\b",
    r"\bdevrim niteliğinde\b",
    r"\bdönüştürücü\b",
    r"\btemel\b",
    r"\bcrucial\b",
    r"\bvital\b",
    r"\bessential\b",
    r"\bpivotal\b",
    r"\bparamount\b",
    r"\btransformative\b",
    r"\bvital role\b",
    r"\bkey role\b",
)
_FAVOURITE_NOUNS_EN = (
    r"\bdelve[sd]?\b",
    r"\btapestry\b",
    r"\brealm of\b",
    r"\blandscape of\b",
    r"\bnavigate the\b",
    r"\bstands as a testament\b",
    r"\ba testament to\b",
    r"\bunderscores? the\b",
    r"\bshowcases?\b",
    r"\bseamless(ly)?\b",
    r"\bholistic\b",
    r"\bmeticulous\b",
    r"\bintricate\b",
    r"\bmyriad\b",
    r"\bcomprehensive\b",
    r"\bnovel approach\b",
    r"\brobust\b",
    r"\bleverage[sd]?\b",
    r"\bunlock(?:s|ed|ing)?\b",
    r"\bharness(?:es|ed|ing)?\b",
    r"\bempower(?:s|ed|ing)?\b",
    r"\bfoster(?:s|ed|ing)?\b",
    r"\bnuanced\b",
    r"\bever-evolving\b",
    r"\bcutting-edge\b",
)
_FAVOURITE_NOUNS_TR = (
    r"\bkapsamlı bir\b",
    r"\bdetaylı bir (?:analiz|inceleme|değerlendirme)\b",
    r"\banlamlı bir katkı\b",
    r"\bderinlemesine\b",
    r"\bgeniş kapsamlı\b",
    r"\bişbirliği içinde\b",
    r"\btüm boyutlarıyla\b",
    r"\bönemli bir (?:rol|katkı)\b",
    r"\bbaşarılı bir şekilde\b",
    r"\bdoğru bir şekilde\b",
)
_PRESENTATIVE_TR = (
    r"\b(?:sunulmaktadır|verilmektedir|gösterilmektedir|ele alınmaktadır)\b",
    r"\bbu makalede\b",
    r"\bbu çalışmada\b",
    r"\bbu tezde\b",
    r"\b(?:amaç|amacımız)\b[^.]{0,40}\b(?:incelemektir|incelemektedir)\b",
)
_HUMAN_MARKERS = (
    r"\b(?:^|\s)I\b",
    r"\bmy\b",
    r"\bme\b",
    r"\bin my (?:view|opinion|experience)\b",
    r"\bbiz\b",
    r"\bbenim\b",
    r"\bben\b",
    r"\bbence\b",
    r"\bsana\b",
    r"\byani\b",
    r"\bmesela\b",
    r"\bvalla\b",
    r"\bya\b",
    r"\bya\b",
    r"\btabii\b",
    r"\bhani\b",
    r"\bvalla\b",
)
_STRONG_POSITIVITY = (
    r"\bharika\b",
    r"\bmükemmel\b",
    r"\bşahane\b",
    r"\bcoşku verici\b",
    r"\binanılmaz\b",
    r"\bmükemmel bir (?:sonuç|performans)\b",
)
_META_HEDGE = (
    r"\bbir nevi\b",
    r"\bneredeyse\b",
    r"\bhemen hemen\b",
    r"\bkısaca\b",
    r"\bsöyleyebiliriz ki\b",
    r"\bdenilebilir ki\b",
)

_CODEBOOK: tuple[Rule, ...] = (
    _rule(
        "discourse_marker",
        0.16,
        Severity.MEDIUM,
        _DISCOURSE_TR + _DISCOURSE_EN,
        group="discourse",
        max_hits=2,
    ),
    _rule("hedge_stack", 0.14, Severity.MEDIUM, _HEDGES, group="hedge", max_hits=2),
    _rule("intensifier_stack", 0.12, Severity.LOW, _INTENSIFIERS, group="intensifier", max_hits=2),
    _rule(
        "favourite_noun",
        0.13,
        Severity.MEDIUM,
        _FAVOURITE_NOUNS_EN + _FAVOURITE_NOUNS_TR,
        group="noun",
        max_hits=2,
    ),
    _rule(
        "presentative_passive",
        0.15,
        Severity.MEDIUM,
        _PRESENTATIVE_TR,
        languages=("tr",),
        group="passive",
    ),
    _rule("meta_hedge", 0.08, Severity.LOW, _META_HEDGE, group="meta", max_hits=2),
    _rule("strong_positivity", 0.12, Severity.MEDIUM, _STRONG_POSITIVITY, group="emotion"),
    _rule("human_marker", -0.20, Severity.INFO, _HUMAN_MARKERS, group="human", max_hits=2),
)


@lru_cache(maxsize=8)
def build_rules(languages: tuple[str, ...] = ("tr", "en")) -> tuple[Rule, ...]:
    """Rules filtered to the requested languages."""
    wanted = set(languages)
    return tuple(rule for rule in _CODEBOOK if set(rule.languages) & wanted)
