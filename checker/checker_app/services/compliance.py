"""Thesis compliance report: what the Turkish rules actually ask for.

**YÖK sets no percentage.** The binding national obligation is procedural - the
institute must obtain an originality report and send it to the advisor and the
jury before the defense (Lisansüstü Eğitim ve Öğretim Yönetmeliği, madde 9/2 for
master's and 21/2 for doctoral theses). The percentages live in each institute
board's own rule, and they cluster on **15% excluding quotations / 20% including
them / 2% single source** (YTÜ) with 30% at Eskişehir, Çukurova and Akdeniz.

**On AI there is a guideline, not a regulation.** YÖK's May 2024 *Yükseköğretimde
Üretken Yapay Zekâ Kullanımına Dair Etik Rehber* requires disclosure of GenAI
use in the affected sections and forbids using it for the stages that need
genuine expertise (hypothesis, discussion, interpretation). It sets no
percentage. TÜBİTAK's January 2026 guide is the same: a qualitative
"significant use" declaration. As of this writing the promised YÖK regulation on
AI use in theses has been announced (November 2025) but not issued.

**"AI detector results alone cannot be the basis for a disciplinary process"**
(İstanbul Aydın University LEE, July 2026) - which is also what the benchmark
authors say: RAID's ethics statement opposes detector use in any punitive
context, and IEEE S&P 2026 reports FPR 0.05-68.6% across commercial detectors on
security papers.

So this report is a checklist with evidence, not a verdict. Every item points at
a section, a sentence, a source and a rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from checker_app.domain.enums import CitationStatus, RiskLevel
from checker_app.domain.models import DocumentReport, SentenceReport
from checker_app.domain.sections import SectionRole, role_label

__all__ = [
    "ComplianceItem",
    "ComplianceReport",
    "build_compliance_report",
    "find_disclosure",
    "YOK_GUIDE",
]

#: YÖK's *Yükseköğretim Kurumları Bilimsel Araştırma ve Yayın Faaliyetlerinde
#: Üretken Yapay Zekâ Kullanımına Dair Etik Rehber* (Mayıs 2024, 20 sayfa),
#: read in full. Three things in it matter for a thesis and are not commonly
#: known:
#:
#: 1. It sets **no numeric threshold at all**. The words *benzerlik* and *oran*
#:    appear **zero** times; *yüzde* appears once, in a non-threshold context.
#:    So there is no national AI-authorship percentage in force.
#: 2. It **does not mention tez/thesis** at all (four incidental hits, all
#:    generic). It binds research and publication, not thesis submission.
#: 3. It draws an explicit **permitted / forbidden** line, and the forbidden side
#:    is the part that a thesis actually needs: hypothesis generation,
#:    discussion, interpretation and application - "üst düzey beceri, deneyim ve
#:    uzmanlık gerektiren aşamalar". The permitted side explicitly includes
#:    literature review, source organisation, grammar/spelling check and
#:    translation, each conditional on the researcher reviewing and correcting
#:    the output and assuming full responsibility.
YOK_GUIDE = {
    "title": (
        "Yükseköğretim Kurumları Bilimsel Araştırma ve Yayın Faaliyetlerinde "
        "Üretken Yapay Zekâ Kullanımına Dair Etik Rehber"
    ),
    "date": "Mayıs 2024",
    "pages": 20,
    "url": (
        "https://proje.yok.gov.tr/documentFiles/17539645334."
        "Yükseköğretimde%20üretken%20yapay%20zeka%20kullanımı-tr.pdf"
    ),
    "numeric_threshold": None,
    "mentions_thesis": False,
    "permitted": (
        "hipotez, yöntem, örneklem büyüklüğü belirleme/güç analizi, veri analizi, "
        "veri toplama, veri saklama ve paylaşma, kaynak araştırması, "
        "kaynak düzenleme, dil bilgisi/yazım denetimi ve çeviri"
    ),
    "forbidden": (
        "hipotez üretimi, tartışma, yorum/interpretation ve uygulama — "
        "'üst düzey beceri, deneyim ve uzmanlık gerektiren aşamalar'"
    ),
    "conditions": (
        "araştırmacı çıktıyı gözden geçirip materyal/yanlılık/faktüel hataları "
        "düzeltmeli ve tüm hukuki ve etik sorumluluğu üstlenmelidir"
    ),
    "named_risks": (
        "kullanımın açıklanmaması; izinsiz başkası içeriği; kaynaksız ya da "
        "hatalı alıntılama; üretken zekânın ürettiği yanıltıcı veri"
    ),
    "reading": (
        "Rehber tezden söz etmiyor ve hiçbir sayısal eşik koymuyor. Bu yüzden "
        "uyum listesi beyanı ve atıfı denetler; benzerlik yüzdesini ise kurumun "
        "kendi kararına bırakır."
    ),
}

#: Turkish de-facto similarity thresholds. Altıntop (2026, §2, citing Toprak 2017
#: and Güçlüer vd. 2024) records them as practice, not as a national rule.
TURKISH_SIMILARITY_NORM = {
    "general_ceiling_percent": 15.0,
    "mostly_accepted_percent": 20.0,
    "single_source_problem_percent": 5.0,
    "source": (
        "Altıntop (2026, DOI 10.56493/nkusbmyo.1866431), Toprak (2017) ve "
        "Güçlüer vd. (2024)'ten aktardığı uygulama değerleri; bağlayıcı ulusal "
        "eşik değildir."
    ),
}

# YÖK Etik Rehber (May 2024) requires the use to be stated in the affected
# section; these are the phrases an author would write.
_DISCLOSURE_RE = re.compile(
    r"(?:yapay zek[âa]|üretken yapay zek[âa]|generative ai|üretken yapay zekâ|dil modeli|"
    r"chatgpt|gemini|claude|copilot|llm|büyük dil modeli)",
    re.IGNORECASE,
)
_DISCLOSURE_VERB_RE = re.compile(
    r"(?:kullan(?:ıiıl)m(?:ıiış)?t|kullanm(?:ıiı)ş|yararlan(?:ıiıl)m(?:ıiı)ş|"
    r"yararlanılm|yarar g(?:öö)rmüş|deste[ğg](?:i|yle| inde|ilmi)|"
    r"yard[ıi]m(?:ıi| ile|iyeti)|aracıl[ıi][ğg][ıi]yla|"
    r"[üu]retilmiş|[üu]retilmiştir|oluşturulmuş|oluşturulmuştür|"
    r"d[üu]zenlenmiş|şekillendirilmiş|çerçevelenmiş|yeniden yaz[ıi]lmış|"
    r"d[üu]zenleme yap[ıi]lmış|metin üretilmiş|içerik üretilmiş)",
    re.IGNORECASE,
)
# Sections where generated prose is specifically not accepted by journals and
# where YÖK expects the author's own interpretation.
_GENERATED_TEXT_SENSITIVE = frozenset(
    {
        SectionRole.RESULTS,
        SectionRole.DISCUSSION,
        SectionRole.CONCLUSION,
        SectionRole.METHOD,
    }
)


@dataclass(frozen=True, slots=True)
class ComplianceItem:
    """One checklist row: a rule, the evidence, and what to do about it."""

    code: str
    rule: str
    status: str
    """One of ``ok`` (fine), ``note`` (information), ``review`` (a human should
    look) or ``required`` (action needed before submission)."""

    severity: str
    evidence: str = ""
    where: str = ""
    action: str = ""
    source: str = ""
    detail: str = ""
    """The rule text itself, where the summary line would lose the operative
    detail. A checklist row that says "disclose AI use" without saying which
    uses are forbidden is not actionable."""


@dataclass(slots=True)
class ComplianceReport:
    items: tuple[ComplianceItem, ...] = ()
    ai_heavy_sections: tuple[dict[str, object], ...] = ()
    uncited_matches: tuple[dict[str, object], ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def blocking(self) -> int:
        return sum(1 for i in self.items if i.status == "required")

    def to_dict(self) -> dict[str, object]:
        return {
            "blocking_items": self.blocking,
            "items": [
                {
                    "code": i.code,
                    "rule": i.rule,
                    "status": i.status,
                    "severity": i.severity,
                    "evidence": i.evidence,
                    "where": i.where,
                    "action": i.action,
                    "source": i.source,
                    "detail": i.detail,
                }
                for i in self.items
            ],
            "yok_guide": YOK_GUIDE,
            "turkish_similarity_norm": TURKISH_SIMILARITY_NORM,
            "ai_heavy_sections": list(self.ai_heavy_sections),
            "uncited_matches": list(self.uncited_matches),
            "notes": list(self.notes),
        }


def find_disclosure(text: str) -> tuple[bool, str]:
    """Does the thesis disclose AI assistance anywhere?"""
    for match in _DISCLOSURE_RE.finditer(text):
        window = text[match.start() : match.start() + 220]
        if _DISCLOSURE_VERB_RE.search(window):
            return True, match.group(0)
        following = text[match.end() : match.end() + 200]
        if _DISCLOSURE_VERB_RE.search(following):
            return True, match.group(0)
    return False, ""


def build_compliance_report(
    report: DocumentReport,
    *,
    raw_text: str = "",
    heavy_threshold: float = 0.5,
    min_match_words: int = 5,
) -> ComplianceReport:
    """Turn a scan into the checklist a thesis advisor or jury member needs."""
    items: list[ComplianceItem] = []
    notes: list[str] = []

    # 1) AI disclosure
    if report.sentences:
        disclosed, evidence = find_disclosure(
            raw_text or " ".join(s.text for s in report.sentences)
        )
        items.append(
            ComplianceItem(
                code="ai_disclosure",
                rule=(
                    "Üretken yapay zekâ kullanımı kullanılan bölümde açıklanmalı "
                    "(YÖK Etik Rehber, Mayıs 2024)"
                ),
                status="ok" if disclosed else "required",
                severity="high" if not disclosed else "none",
                evidence=evidence or "tez içinde yapay zekâ kullanımına dair ifade bulunamadı",
                where="tüm metin",
                action=(
                    ""
                    if disclosed
                    else "Kullanılan aracın adı/sürümü, hangi bölümde ve nasıl "
                    "kullanıldığı yöntem bölümünde yazılmalı."
                ),
                source=(
                    "YÖK, Yükseköğretimde Üretken Yapay Zekâ Kullanımına Dair Etik "
                    "Rehber (Mayıs 2024); TÜBİTAK UYZ Rehberi (Ocak 2026)"
                ),
                detail=(
                    f"Rehber (Mayıs 2024, 20 s.) izinli: {YOK_GUIDE['permitted']}. "
                    f"YASAK: {YOK_GUIDE['forbidden']}. Koşul: "
                    f"{YOK_GUIDE['conditions']}. Sayısal eşik YOK, tezden hiç söz "
                    "etmiyor."
                ),
            )
        )

    # 2) sections where generated prose is not accepted
    heavy: list[dict[str, object]] = []
    for section in report.sections:
        role = SectionRole.OTHER
        for sentence in report.sentences:
            if " › ".join(sentence.location.section_path) == section.title:
                role = sentence.location.section_role
                break
        if role in _GENERATED_TEXT_SENSITIVE and section.ai_score >= heavy_threshold:
            offenders = [
                {"index": s.index, "ai_score": s.ai_score, "snippet": s.snippet(140)}
                for s in report.sentences
                if " › ".join(s.location.section_path) == section.title
                and s.ai_score >= heavy_threshold
            ][:5]
            heavy.append(
                {
                    "section": section.title,
                    "role": role_label(role),
                    "ai_score": section.ai_score,
                    "sentences": offenders,
                }
            )
    if heavy:
        items.append(
            ComplianceItem(
                code="generated_prose_in_critical_sections",
                rule=(
                    "Bulgular/Tartışma/Sonuç/Yöntem bölümlerinin kendi yorumuyla "
                    "yazılması gerekir; dergi politikaları bu bölümlerde üretilen metni "
                    "kabul etmez"
                ),
                status="review",
                severity="medium",
                evidence="; ".join(
                    f"{h['section']} ({h['role']}) AI skoru {h['ai_score']:.2f}" for h in heavy
                ),
                where=", ".join(str(h["section"]) for h in heavy),
                action=(
                    "Bu bölümleri yeniden yazın: yorum, sınırlılık ve tartışma kısmı "
                    "yazarın kendi olmalı. AI tespit sonucu tek başına delil değildir, "
                    "ama bu bölümler insan eline geçmemişse savunulması zordur."
                ),
                source="IMCS/DergiPark yayın politikaları (2026); YÖK Etik Rehber (2024)",
            )
        )

    # 3) unattributed matches
    uncited: list[dict[str, object]] = []
    ranked: list[tuple[int, dict[str, object]]] = []
    for sentence in report.sentences:
        if not sentence.plagiarism or not sentence.plagiarism.matches:
            continue
        for match in sentence.plagiarism.matches:
            # Same "5 kelimeden az eşleşme hariç" filter the similarity report
            # uses, otherwise the two panels disagree on the same document.
            if match.matched_words < min_match_words:
                continue
            if match.citation_status is CitationStatus.NOT_CITED_OR_QUOTED:
                ranked.append(
                    (
                        match.matched_words,
                        {
                            "sentence": sentence.index,
                            "location": sentence.location.label(),
                            "source": match.source_name,
                            "kind": match.kind.value,
                            "matched_words": match.matched_words,
                            "snippet": match.snippet[:180],
                        },
                    )
                )
    ranked.sort(key=lambda pair: -pair[0])
    uncited = [row for _, row in ranked]
    if uncited:
        items.append(
            ComplianceItem(
                code="uncited_matches",
                rule=(
                    "Kaynak gösterilmemiş ve tırnak içine alınmamış eşleşmeler "
                    "(kurumsal filtreler: kaynakça hariç, 5 kelimeden az eşleşme hariç)"
                ),
                status="required",
                severity="high",
                evidence=f"{len(uncited)} eşleşme, en uzunu {uncited[0]['matched_words']} kelime",
                where=f"{len(uncited)} eşleşme",
                action=(
                    "Her eşleşme için ya kaynak gösterin ya da metni kendi cümlelerinizle "
                    "yeniden yazın. Kaynakça bölümü zaten sayıma dâhil edilmiyor."
                ),
                source="Lisansüstü Eğitim ve Öğretim Yönetmeliği md. 9/2; YTÜ/İÜ/Başkent kuralları",
            )
        )
    elif report.similarity is not None:
        items.append(
            ComplianceItem(
                code="attribution_complete",
                rule="Bütün eşleşmelerde atıf var",
                status="ok",
                severity="none",
                evidence="atıfsız eşleşme bulunamadı",
                action="",
            )
        )

    # 4) similarity against institutional bands
    if report.similarity is not None:
        flagged = [band for band in report.similarity.bands if band["issues"]]
        items.append(
            ComplianceItem(
                code="similarity_bands",
                rule=(
                    "Kurumsal benzerlik eşikleri (YÖK ulusal oran belirlemiyor; "
                    "enstitü/kurul kararı)"
                ),
                status="review" if flagged else "ok",
                severity="medium" if flagged else "none",
                evidence=(
                    "; ".join(
                        f"{band['institution']}: {', '.join(band['issues'])}" for band in flagged
                    )
                    or f"hiçbir kurum eşiği aşılmadı (en yüksek eşik: {report.similarity.similarity_incl_quotes:.1f}% alıntılar dahil)"
                ),
                where="belge",
                action=(
                    "Kurumunuzun kendi filtresini raporda uygulayın; "
                    "--excl-quotes / --keep-references ile yeniden hesaplayın."
                ),
                source="YTÜ, İstanbul Üniv., Başkent, ESÜ, Çukurova/Akdeniz kurul kararları",
            )
        )
        baseline = report.similarity.baseline
        items.append(
            ComplianceItem(
                code="similarity_baseline",
                rule=("Ölçülmüş Türkçe tez dağılımıyla karşılaştırma (uydurma eşik yok)"),
                status="note",
                severity="none",
                evidence=(
                    f"bu tez {report.similarity.similarity_incl_quotes:.1f}% · "
                    f"ölçülen ortalama {baseline.mean_percent}% ± {baseline.sd_percent} "
                    f"(n={baseline.sample_size}); {baseline.percentile_note(report.similarity.similarity_incl_quotes)}"
                ),
                where="belge",
                action="",
                source=baseline.source,
            )
        )

    # 5) text integrity
    if report.integrity is not None and report.integrity.tampered:
        items.append(
            ComplianceItem(
                code="text_integrity",
                rule="Metin bütünlüğü (homoglyph / görünmez karakter)",
                status="required",
                severity="high",
                evidence="; ".join(f"{f.code}×{f.count}" for f in report.integrity.findings),
                where="belge",
                action="Karakterlerin bir kısmı değiştirilmiş; metni kaynağından yeniden alın.",
                source="RAID (arXiv:2405.07940): homoglyph altında ortalama doğruluk düşüşü %40.6",
            )
        )

    # 6) how much of the document was actually judged
    ai_signals_ran = any(
        sentence.ratio is not None
        or sentence.classifier is not None
        or sentence.perplexity is not None
        for sentence in report.sentences
    )
    if not ai_signals_ran:
        items.append(
            ComplianceItem(
                code="ai_signals_off",
                rule="AI yazım sinyalleri çalıştırılmadı",
                status="note",
                severity="none",
                evidence=(
                    "--no-perplexity / --no-classifier / --no-ratio verilmiş; "
                    "yalnızca intihal ve stilometri çalıştı"
                ),
                where="belge",
                action="AI bölümü için sinyalleri açıp yeniden çalıştırın.",
                source="",
            )
        )
    elif report.sentence_count:
        share = report.unscored_sentences / report.sentence_count
        items.append(
            ComplianceItem(
                code="coverage",
                rule="AI ölçümünün kapsadığı cümle oranı",
                status="note" if share < 0.5 else "review",
                severity="none" if share < 0.5 else "low",
                evidence=(
                    f"{report.sentence_count - report.unscored_sentences}/"
                    f"{report.sentence_count} cümle ölçüldü (%{round((1 - share) * 100)})"
                ),
                where="belge",
                action=(
                    "Kısa cümleler (12 kelimenin altı) ölçülmez: kısa metinde tespit "
                    "güvenilmez (AUROC 0.62/10 token)."
                ),
                source="arXiv:2509.18880; arXiv:2603.23146",
            )
        )

    notes.append(
        "Bu rapor tek başına disiplin işlemi gerekçesi olamaz. RAID ekibi "
        "(arXiv:2405.07940) dedektörlerin cezai bağlamda kullanılmasına itiraz eder; "
        "IEEE S&P 2026 ticari dedektörlerin akademik bağlamda 'yüksek riskli kararlar "
        "için uygun olmadığını' ve ölçülen FPR aralığının %0.05-68.6 olduğunu "
        "bildiriyor (N=6.295)."
    )
    notes.append(
        "Dürüst yapay zekâ kullanımı, kullanımı gizlemekten daha fazla işaretlenme "
        "riski taşır: hafif düzeltilmiş özetlerde Pangram %80.1, GPTZero %48.5 "
        "işaretliyor; 'humanizer' ile kaçırılan metinlerde kaçırma oranı %96'nın "
        "üstünde (arXiv:2608.11256)."
    )
    return ComplianceReport(
        items=tuple(items),
        ai_heavy_sections=tuple(heavy),
        uncited_matches=tuple(uncited[:25]),
        notes=tuple(notes),
    )


def sentences_to_review(
    report: DocumentReport, level: RiskLevel = RiskLevel.MEDIUM, limit: int = 20
) -> Sequence[SentenceReport]:
    """The sentences a human should actually read, worst first."""
    ordered = sorted(report.sentences, key=lambda s: (-s.risk_score, s.index))
    return [s for s in ordered if s.risk.at_least(level)][:limit]
