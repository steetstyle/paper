"""Turnitin-style similarity metrics, with the filters Turkish theses use.

Universities do not read "you have 3 matches". They read a percentage that was
produced with a specific filter set, and they compare it to a number written
into their own regulation. The filters that appear in the Turkish institutional
rules (YÖK sets **no** national percentage; each institute board decides) are:

* ``Kaynakça hariç`` - reference list excluded
* ``Alıntılar dahil`` - quotations included in the headline number, and a second
  number with quotations excluded
* ``5 kelimeden az eşleşmeler hariç`` - matches shorter than ~5 words excluded
* a per-source ceiling, because a single source is the usual actual problem

Observed institutional thresholds (see :data:`INSTITUTION_BANDS` for sources):
15% excluding quotations / 20% including them / 2% single source at Yıldız
Technical University; 20% at İstanbul and Başkent; 30% at Eskişehir, Çukurova and
Akdeniz.

Because these numbers describe *coverage against a source set we were given*, this
module also prints the measured Turkish baseline it should be read against:
28.7% mean similarity (SD 11.58, n=600 education-sciences theses; 29.31% for
Turkish-language and 24.37% for English-language theses, Toprak 2014). A 30%
overlap is unremarkable for a Turkish thesis and alarming for an English one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from checker_app.domain.enums import CitationStatus
from checker_app.domain.models import MatchSpan
from checker_app.domain.text import Token
from checker_app.services.attribution import Span

__all__ = [
    "InstitutionBand",
    "INSTITUTION_BANDS",
    "SourceShare",
    "SimilarityStats",
    "SimilarityReport",
    "TurkishBaseline",
    "EnglishBaseline",
    "TurkishAiPresence",
    "LanguageCoverage",
    "build_language_coverage",
    "build_similarity_report",
]


@dataclass(frozen=True, slots=True)
class InstitutionBand:
    """A published institutional rule."""

    institution: str
    total_excl_quotes: float | None
    total_incl_quotes: float | None
    single_source: float | None
    url: str
    note: str = ""
    language: str = "tr"
    """Which document language this rule applies to.

    Turkish and English thresholds are not interchangeable: the measured clean
    corpus differs by roughly 3x (28.7% vs 9%), so a Turkish rule applied to an
    English thesis would flag three quarters of a normal one, and vice versa.
    """

    no_threshold_reason: str = ""
    """Some institutions publish a rule that is *about* not having a number.

    Rhodes: staff "may not set an acceptable percentage on the similarity index
    ... the similarity index is not an indication of levels of plagiarism."
    Glasgow Caledonian: "there can be no cutoff point where plagiarism begins and
    ends". Those are real, citable positions and belong in a report as such -
    not as an empty row.
    """


# Percentages, exactly as published by each institution (2023-2026 access dates).
INSTITUTION_BANDS: tuple[InstitutionBand, ...] = (
    InstitutionBand(
        "YTÜ (Temiz Enerji / Sosyal Bilimler / Fen Bilimleri)",
        15.0,
        20.0,
        2.0,
        "https://tet.yildiz.edu.tr/mezuniyet/intihal-kontrol",
        "iki rapor (savunma öncesi/sonrası), uyumlu olmayan tez savunmaya giremez",
    ),
    InstitutionBand(
        "İstanbul Üniversitesi Sağlık Bilimleri",
        None,
        20.0,
        None,
        "http://cdn.istanbul.edu.tr/statics/saglikbilimleri.istanbul.edu.tr/wp-content/uploads/2018/02/Alper-Hoca-2.pdf",
        "bibliyograf hariç, alıntılar dahil, 5 kelimeden küçük eşleşmeler hariç",
    ),
    InstitutionBand(
        "Başkent Üniversitesi Enstitüler",
        20.0,
        20.0,
        2.0,
        "https://www.baskent.edu.tr/belgeler/mevzuat/yonerge/enstitu_yong_16.pdf",
        "tek kaynak sınırı %2",
    ),
    InstitutionBand(
        "Eskişehir Teknik Üniversitesi LEE",
        30.0,
        30.0,
        15.0,
        "https://lee.eskisehir.edu.tr/tr/Duyuru/Detay/tezlerin-intihal-kontrolunden-gecirilmesi-hakkinda-bilgilendirme",
        "izin verilen filtreler: kaynakça hariç, %1 altı örtüşme hariç",
    ),
    InstitutionBand(
        "Çukurova Üniversitesi BADI / Akdeniz Üniversitesi SBE",
        30.0,
        30.0,
        None,
        "https://babe.cu.edu.tr/cu/ogrenci/turnitin-intihal-benzesim-programi-kullanim-ilkeleri-ve-kullanim-kilavuzu-turnitin-plagiarism-program/turnitin-intihal-benzesim-programi-kullanim-ilkeleri",
        "%30 üzeri yazılı açıklama ister; kurum bunun 'hukuken intihal yok' demek olmadığını vurgular",
    ),
    # ---------------------------------------------------------------- English
    # Every URL below was fetched, not inferred. Each row also carries the filter
    # configuration the institution specifies, because the same document scores
    # differently under different filters.
    InstitutionBand(
        "Virginia Tech, Graduate School",
        25.0,
        25.0,
        5.0,
        "https://graduateschool.vt.edu/faculty-and-staff-resources/ithenticate.html",
        "eşiği aşan ETD ek incelemeye gider; NOT: VT'nin Honor System sayfası "
        "%15 de yayımlıyor, iki canlı sayfa birbiriyle çelişiyor — tek sayı "
        "olarak alıntılanmamalı",
        language="en",
    ),
    InstitutionBand(
        "University of East Anglia (UK)",
        20.0,
        20.0,
        None,
        "https://assets.uea.ac.uk/f/185167/x/153d8f068d/plagiarism-and-collusion-2018-2019.pdf",
        "hacme göre bantlar: <%5 düşük, %5-20 orta, >%20 yüksek",
        language="en",
    ),
    InstitutionBand(
        "Muhammad Ali Jinnah University (Pakistan), HEC kuralı",
        20.0,
        20.0,
        5.0,
        "https://jinnah.edu/plagiarism-policy/",
        "kendi yayınlarından gelen benzerlik hariç tutulur",
        language="en",
    ),
    InstitutionBand(
        "Charles Sturt University (Australia)",
        25.0,
        25.0,
        None,
        "https://cdn.csu.edu.au/__data/assets/pdf_file/0006/3912117/Interpreting-Similarity-Reports.pdf",
        "kurumun kendi örneği: doğru alıntılanıp kaynak gösterilmiş bir deneme "
        "%30 çıktı ve intihal YOKTU. Eşik karar değil, soru başlatıcıdır",
        language="en",
    ),
    InstitutionBand(
        "Panjab University (India) — bölüm bazlı",
        20.0,
        20.0,
        None,
        "https://newmodel.pau.edu/anti-plagiarism-policy",
        "tez için: giriş %30, literatür %50, yöntem %25, sonuç %10, tartışma %10, "
        "özet %10, tam tez %20. Literatür taraması %50'ye kadar meşru sayılıyor",
        language="en",
    ),
    InstitutionBand(
        "Rhodes University (South Africa)",
        None,
        None,
        None,
        "https://www.ru.ac.za/media/rhodesuniversity/content/deanofstudents/documents/Common_Faculty_Policy_and_Procedures_on_Plagiarism.pdf",
        "kurum öğretim üyesinin benzerlik endeksi için kabul edilebilir yüzde "
        "belirlemesine izin vermiyor: 'benzerlik endeksi intihal düzeyinin göstergesi değildir'",
        language="en",
        no_threshold_reason=(
            "Kurum sayısal eşik koymayı reddediyor ve endeksin bir suçluluk ölçütü "
            "olmadığını açıkça söylüyor."
        ),
    ),
    InstitutionBand(
        "Glasgow Caledonian University (UK)",
        None,
        None,
        None,
        "https://www.gcu.ac.uk/__data/assets/pdf_file/0015/36222/gcu20similarity20checking20policy.pdf",
        "'intihalin başladığı ve bittiği bir kesim noktası yoktur'; ayrıca en küçük "
        "eşleşme eşiğini 3 veya altına çekmenin puanı şişirdiğini uyarıyor",
        language="en",
        no_threshold_reason=(
            "Kurum eşik kavramını reddediyor ve ayrıca filtre ayarının puanı "
            "nasıl bozabileceğini ölçüyor."
        ),
    ),
    InstitutionBand(
        "Australian National University",
        None,
        None,
        None,
        "https://policies.anu.edu.au/ppl/document/ANUP_012815",
        "tezle birlikte iThenticate raporu isteniyor, sayısal eşik yayımlanmıyor",
        language="en",
        no_threshold_reason="Rapor zorunlu, eşik yok.",
    ),
)


@dataclass(frozen=True, slots=True)
class EnglishBaseline:
    """The measured English reference - and it is *three times lower* than Turkish.

    This is the correction that matters most for English mode. The Turkish
    baseline says a thesis averages 28.7% ± 11.58 (n=600, education sciences,
    Toprak 2014). Applying that to an English thesis would flag roughly **three
    quarters of a normal, well-cited thesis**.

    ================================  =========  ====
    English corpus                            mean        SD
    ================================  =========  ====
    **360 US dissertations** (adjusted)      **9%**    **6%**
    360 published articles, same field      11%     10%
    ================================  =========  ====

    Source: Mayes, *A Content Originality Analysis of HRD Focused Dissertations
    and Published Academic Articles using TurnItIn* (2017), 360 dissertations in
    HRD / training / organisational / career development. Level distribution:
    88.1% low, 9.7% high (.15-.24), **2.2%** excessive (.25-1.00). So even in the
    worst band only 2.2% of dissertations cross 25%.

    Two further results from that study matter for how the percentage is read:

    * **The only significant predictors of a high score were the reference count
      and the word count** (multinomial logistic regression, Nagelkerke
      pseudo-R² = 0.169). A long bibliography inflates the score; nothing about
      the writing does.
    * **Removing false positives - reference lists the tool failed to exclude -
      changed scores significantly.** Which is why this tool excludes the
      bibliography by default and says so.

    And the line that is *measured* rather than chosen:

    ==============================  ========  ============
    empirically optimal cutoff        15%     sens. **84.8%**
                                                  spec. **80.5%**
    ==============================  ========  ============

    Higgins, Lin & Evans, *Research Integrity and Peer Review* 1:13 (2016),
    DOI 10.1186/s41073-016-0021-8, on 400 manuscripts manually verified:
    plagiarised mean **25.8** (SD 9.9, n=66), non-plagiarised mean **11.5**
    (SD 6.2, n=333), p < 0.001, AUC **0.902** (95% CI 0.863-0.940). Restricting
    to Abstract/Introduction/Results/Discussion widened the gap: plagiarised
    **25.7**, non-plagiarised **5.6**.

    So for English there are two different numbers, and conflating them is the
    common mistake: **15% is where detection actually works; 25-30% is a policy
    ceiling that a normal thesis never approaches.**
    """

    mean_percent: float = 9.0
    sd_percent: float = 6.0
    article_mean_percent: float = 11.0
    article_sd_percent: float = 10.0
    excessive_share: float = 0.022
    """Dissertations above 25%, after adjustment."""

    optimal_cutoff_percent: float = 15.0
    """Where detection was empirically validated, not where a policy sits."""

    cutoff_sensitivity: float = 0.848
    cutoff_specificity: float = 0.805
    cutoff_auc: float = 0.902

    plagiarised_mean: float = 25.8
    plagiarised_sd: float = 9.9
    clean_mean: float = 11.5
    clean_sd: float = 6.2
    section_only_plagiarised: float = 25.7
    section_only_clean: float = 5.6
    non_native_share_of_cases: float = 0.82
    """82% of confirmed plagiarised manuscripts came from countries where English
    is not an official language (55 of 66). Stated as a caution about L2 writing,
    not as evidence about authorship."""

    sample_size: int = 360
    field: str = "HRD / eğitim geliştirme (ABD)"
    source: str = (
        "Mayes (2017), 360 US dissertations; cutoff Higgins vd. (2016), "
        "DOI 10.1186/s41073-016-0021-8"
    )
    #: Overwritten by ``SimilarityReport.to_dict``; a dataclass field cannot
    #: depend on a sibling's value, and this only exists to give ``to_dict`` a
    #: z-score to print.
    similarity_for_note: float = 0.0
    caveat: str = (
        "ABD, tek alan (HRD), 2017 öncesi. Türkçe tezlerin %28.7 ortalamasıyla "
        "aynı ölçek değildir: dil, kaynak kültürü ve alan birlikte değişir. "
        "Tezler için bölüm bazlı bir İngilizce dağılım yayımlanmamıştır."
    )

    def percentile_note(self, percent: float) -> str:
        if self.mean_percent <= 0:
            return ""
        sigma = (percent - self.mean_percent) / self.sd_percent
        return (
            f"ölçülen doktora tezi ortalaması {self.mean_percent}% ± {self.sd_percent} "
            f"→ z={sigma:+.2f}"
        )

    def reading(self, percent: float) -> str:
        if percent >= 25.0:
            verdict = (
                f"%{percent:.1f} eşiğin üstünde; ama Charles Sturt'un örnek vakasında "
                "doğru alıntılanıp kaynak gösterilmiş bir deneme %30 çıkmıştı ve "
                "intihal yoktu. Eşiği aşmak tez zorunluluğu değildir."
            )
        elif percent >= self.optimal_cutoff_percent:
            verdict = (
                f"%{percent:.1f} ölçülmüş optimal eşiğin (%{self.optimal_cutoff_percent:.0f}) "
                f"üstünde; bu eşikte duyarlılık %{self.cutoff_sensitivity * 100:.1f}, "
                f"özgüllük %{self.cutoff_specificity * 100:.1f} olarak doğrulandı."
            )
        else:
            verdict = (
                f"%{percent:.1f} ölçülmüş eşiğin altında. Temiz doktora tezi ortalaması "
                f"%{self.mean_percent:.0f} ± {self.sd_percent:.0f}; yani bu aralık "
                "normaldir."
            )
        return f"{verdict} Kaynak: {self.source}. {self.caveat}"

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_percent": self.mean_percent,
            "sd_percent": self.sd_percent,
            "published_article_mean_percent": self.article_mean_percent,
            "published_article_sd_percent": self.article_sd_percent,
            "dissertations_above_25_percent": self.excessive_share,
            "optimal_cutoff_percent": self.optimal_cutoff_percent,
            "cutoff_sensitivity": self.cutoff_sensitivity,
            "cutoff_specificity": self.cutoff_specificity,
            "cutoff_auc": self.cutoff_auc,
            "plagiarised_mean": self.plagiarised_mean,
            "plagiarised_sd": self.plagiarised_sd,
            "clean_mean": self.clean_mean,
            "clean_sd": self.clean_sd,
            "section_only_plagiarised": self.section_only_plagiarised,
            "section_only_clean": self.section_only_clean,
            "non_native_share_of_confirmed_cases": self.non_native_share_of_cases,
            "sample_size": self.sample_size,
            "field": self.field,
            "source": self.source,
            "caveat": self.caveat,
            "note": self.percentile_note(self.similarity_for_note),
        }


@dataclass(frozen=True, slots=True)
class TurkishBaseline:
    """The measured distribution this percentage should be read against."""

    mean_percent: float = 28.7
    sd_percent: float = 11.58
    turkish_percent: float = 29.31
    english_percent: float = 24.37
    high_plagiarism_percent: float = 34.5
    sample_size: int = 600
    field: str = "eğitim bilimleri"
    source: str = (
        "Ziya Toprak, Türkiye'de Akademik Yazı: İntihal ve Özgünlük, "
        "Boğaziçi Üniversitesi Eğitim Dergisi 34(2), 2014"
    )
    caveat: str = (
        "Zaman ve alan dar; disiplinsel kırılım için 2025-26 döneminde doğrulanmış "
        "Türkçe tez oranı yayımlanmamıştır."
    )

    def percentile_note(self, percent: float) -> str:
        if self.mean_percent <= 0:
            return ""
        sigma = (percent - self.mean_percent) / self.sd_percent
        return f"ölçülen ortalama {self.mean_percent}% ± {self.sd_percent} → z={sigma:+.2f}"


@dataclass(frozen=True, slots=True)
class TurkishAiPresence:
    """What Turkish academic writing actually looks like, measured.

    This is the only published distribution for Turkish academic prose that the
    research turned up, and it is about AI presence rather than similarity -
    so it is the right companion to :class:`TurkishBaseline`, which measures
    overlap. Akkaya & Beygirci (2026, DOI 10.46452/baksoder.1899625) hand-verified
    **204 Turkish articles** from 68 university journals on DergiPark, published
    2025:

    ==================  =========  ==========
    AI-rate band         articles   band mean
    ==================  =========  ==========
    0-20%                    122  (59.8%)   6%
    21-40%                    45  (22.1%)  28%
    41-60%                    25  (12.3%)  50%
    61-80%                    11   (5.4%)  69%
    81-100%                    1   (0.5%)  94%
    **total**                204          mean **20%**
    ==================  =========  ==========

    Control groups from the same study: 20 AI-generated documents were **100%**
    flagged at "generally 40-60%", and 2010-2020 Turkish articles came out
    **0-10%**. Indexed in TR Dizin (n=102) the mean was **12%**, against **28%**
    for non-TR Dizin (n=102); articles above 40% were 9/102 (8.8%) against
    28/102 (27.5%).

    Two readings matter for a thesis:

    * **59.8% of Turkish academic articles sit below a 20% AI rate**, with a band
      mean of 6% - which is the same 20% this tool uses as its reporting floor,
      arrived at from the opposite direction.
    * **AI concentrates in the opening, not the method.** By section: introduction
      and literature review 100/204 (**49.0%**), interpretation of findings 94
      (46.1%), abstract 78 (38.2%), conclusion 77 (37.7%), **method 6 (2.9%)**.
      A thesis whose method section is fine and whose literature review is not
      is the ordinary case, not the suspicious one.

    Caveat carried with it: articles, not theses, and a hand-verified sample of
    one journal platform.
    """

    mean_percent: float = 20.0
    below_20_percent: int = 122
    below_20_share: float = 0.598
    below_20_band_mean_percent: float = 6.0
    tr_dizin_mean_percent: float = 12.0
    non_tr_dizin_mean_percent: float = 28.0
    sample_size: int = 204
    section_rates: Mapping[str, float] = field(
        default_factory=lambda: {
            "introduction_literature": 0.490,
            "interpretation_of_findings": 0.461,
            "abstract": 0.382,
            "conclusion": 0.377,
            "method": 0.029,
        }
    )
    source: str = (
        "Akkaya & Beygirci (2026), MAKALELERDE YAPAY ZEKÂ VARLIĞININ "
        "DEĞERLENDİRİLMESİ, DOI 10.46452/baksoder.1899625"
    )
    caveat: str = (
        "Makale, tez değil; DergiPark'taki 68 üniversite dergisinden elle "
        "doğrulanmış 204 örnek. Tezler için ölçülmüş bir AI oranı dağılımı yok."
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_percent": self.mean_percent,
            "below_20_percent_count": self.below_20_percent,
            "below_20_percent_share": self.below_20_share,
            "below_20_percent_band_mean": self.below_20_band_mean_percent,
            "tr_dizin_mean_percent": self.tr_dizin_mean_percent,
            "non_tr_dizin_mean_percent": self.non_tr_dizin_mean_percent,
            "sample_size": self.sample_size,
            "section_rates": dict(self.section_rates),
            "source": self.source,
            "caveat": self.caveat,
            "reading": (
                f"Türkçe akademik yazıda ortalama AI oranı %{self.mean_percent}; "
                f"{self.below_20_percent}/{self.sample_size} (%{self.below_20_share * 100:.1f}) "
                f"makale %20'nin altında ve o bandın ortalaması yalnız "
                f"%{self.below_20_band_mean_percent}. Bu araç %20'yi raporlama "
                "tabanı olarak kullanıyor; Türkçe veriden bağımsız olarak aynı "
                "eşik çıkıyor. AI giriş ve literatür taramasında yoğunlaşıyor "
                f"(%{self.section_rates['introduction_literature'] * 100:.1f}), "
                f"yöntemde neredeyse hiç yok (%{self.section_rates['method'] * 100:.1f})."
            ),
        }


@dataclass(frozen=True, slots=True)
class LanguageCoverage:
    """What share of the matched words came from a same-language source.

    The ordinary case for a Turkish thesis is that its references are English or
    German, and reuse detection is measurably weaker across that boundary:
    precision **80%** on untranslated text, **26.7%** on translated text and
    **16.7%** on translated-then-paraphrased text (DOI 10.33806/ijaes1026). So a
    clean percentage in that setting is partly a statement about what was
    supplied, not about the thesis - and this object says which case you are in
    instead of leaving it to be guessed.
    """

    document_language: str = "tr"
    same_language_sources: int = 0
    cross_language_sources: int = 0
    same_language_words: int = 0
    cross_language_words: int = 0
    unknown_language_sources: int = 0
    sources_by_language: Mapping[str, int] = field(default_factory=dict)
    words_by_language: Mapping[str, int] = field(default_factory=dict)
    evidence: str = (
        "Çeviri altında tespit kesinliği düşer: çeviri yok %80, çeviri %26.7, "
        "çeviri+paraphrase %16.7 (DOI 10.33806/ijaes1026). Türkçe bir tezin "
        "kaynakları çoğunlukla başka dilde olduğundan, düşük benzerlik yüzdesi "
        "kısmen verilen kaynak kümesinin bir özelliğidir."
    )

    @property
    def cross_language_ratio(self) -> float:
        total = self.same_language_words + self.cross_language_words
        return self.cross_language_words / total if total else 0.0

    def reading(self) -> str:
        total = self.same_language_sources + self.cross_language_sources + self.unknown_language_sources
        if not total:
            return "Kaynak verilmedi."
        if self.cross_language_sources == 0:
            return (
                f"Kaynakların tümü belge diliyle aynı ({self.document_language}): "
                f"{self.same_language_sources} kaynak, {self.same_language_words} kelime eşleşti."
            )
        if self.same_language_sources == 0:
            return (
                f"Kaynakların tümü belge dilinden FARKLI ({self.cross_language_sources} kaynak); "
                f"bunlardan {self.cross_language_words} kelime eşleşti. Çapraz dil olduğu "
                "için tespit duyarlılığı bilinçli olarak düşüktür — bulunan eşleşmeler "
                "güçlü kanıttır, bulunamayanlar kanıt DEĞİLDİR. Kaynak kümesine İngilizce "
                "ya da aynı dilde bir kaynak eklemek sonucu doğrulamak için en ucuz adımdır."
            )
        return (
            f"{self.same_language_sources} kaynak belge diliyle aynı "
            f"({self.same_language_words} kelime), {self.cross_language_sources} kaynak "
            f"farklı ({self.cross_language_words} kelime, %{self.cross_language_ratio * 100:.0f}). "
            "Çapraz dil kısımda tespit duyarlılığı düşüktür."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "document_language": self.document_language,
            "same_language_sources": self.same_language_sources,
            "cross_language_sources": self.cross_language_sources,
            "unknown_language_sources": self.unknown_language_sources,
            "same_language_words": self.same_language_words,
            "cross_language_words": self.cross_language_words,
            "cross_language_ratio": round(self.cross_language_ratio, 4),
            "sources_by_language": dict(self.sources_by_language),
            "words_by_language": dict(self.words_by_language),
            "reading": self.reading(),
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class SourceShare:
    """How much of the document one reference accounts for."""

    source_id: str
    name: str
    matched_words: int
    share_percent: float
    match_count: int
    quoted_words: int = 0
    unattributed_words: int = 0
    uncited_matches: int = 0

    @property
    def attributed(self) -> bool:
        return self.uncited_matches == 0


@dataclass(frozen=True, slots=True)
class SimilarityStats:
    """The numbers behind the percentages, all recomputable."""

    total_words: int
    matched_words: int
    matched_words_quoted: int
    matched_words_in_references: int
    reference_words: int
    quoted_words: int
    match_count: int
    cases: int
    granularity: float
    """Matches per connected case. The PAN metric punishes fragmentation, and
    so should this report: one copied paragraph reported five times looks worse
    than the same paragraph reported once."""


@dataclass(frozen=True, slots=True)
class SimilarityReport:
    """Everything an academic-integrity report needs, plus its provenance."""

    similarity_incl_quotes: float
    similarity_excl_quotes: float
    similarity_excl_references: float
    similarity_excl_references_quotes: float
    stats: SimilarityStats
    per_source: tuple[SourceShare, ...] = ()
    largest_single_source: SourceShare | None = None
    attribution: dict[str, int] = field(default_factory=dict)
    bands: tuple[dict[str, object], ...] = ()
    baseline: TurkishBaseline = field(default_factory=TurkishBaseline)
    english_baseline: EnglishBaseline | None = None
    """Used when the document is English.

    Not optional decoration: the two baselines differ by roughly 3x (28.7% vs
    9%), so applying the Turkish one to an English thesis would flag about three
    quarters of a normal, well-cited thesis. Set by ``document_language``."""
    ai_presence: TurkishAiPresence = field(default_factory=TurkishAiPresence)
    """What AI presence in Turkish academic writing actually looks like, so the
    AI share has a Turkish reference the way the similarity percentage does."""

    language_coverage: LanguageCoverage | None = None
    """Which matched words came from a same-language source. Reported because a
    Turkish thesis citing English literature is the ordinary case, and reuse
    detection across that boundary is measurably weaker."""
    filters: str = "kaynakça hariç · alıntılar dahil · 5 kelimeden az eşleşmeler hariç"
    filters_applied: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "similarity": {
                "incl_quotes_percent": round(self.similarity_incl_quotes, 2),
                "excl_quotes_percent": round(self.similarity_excl_quotes, 2),
                "excl_references_percent": round(self.similarity_excl_references, 2),
                "excl_references_and_quotes_percent": round(
                    self.similarity_excl_references_quotes, 2
                ),
                "largest_single_source_percent": (
                    round(self.largest_single_source.share_percent, 2)
                    if self.largest_single_source
                    else 0.0
                ),
                "largest_single_source_name": (
                    self.largest_single_source.name if self.largest_single_source else None
                ),
            },
            "filters": {
                "description": self.filters,
                "applied": self.filters_applied,
            },
            "stats": {
                "total_words": self.stats.total_words,
                "matched_words": self.stats.matched_words,
                "matched_words_quoted": self.stats.matched_words_quoted,
                "matched_words_in_references": self.stats.matched_words_in_references,
                "reference_words": self.stats.reference_words,
                "quoted_words": self.stats.quoted_words,
                "match_count": self.stats.match_count,
                "cases": self.stats.cases,
                "granularity": round(self.stats.granularity, 3),
            },
            "attribution": self.attribution,
            "per_source": [
                {
                    "source_id": s.source_id,
                    "name": s.name,
                    "matched_words": s.matched_words,
                    "share_percent": round(s.share_percent, 2),
                    "match_count": s.match_count,
                    "quoted_words": s.quoted_words,
                    "unattributed_words": s.unattributed_words,
                    "uncited_matches": s.uncited_matches,
                }
                for s in self.per_source
            ],
            "institutional_bands": list(self.bands),
            "baseline": {
                "mean_percent": self.baseline.mean_percent,
                "sd_percent": self.baseline.sd_percent,
                "turkish_percent": self.baseline.turkish_percent,
                "english_percent": self.baseline.english_percent,
                "high_plagiarism_percent": self.baseline.high_plagiarism_percent,
                "sample_size": self.baseline.sample_size,
                "field": self.baseline.field,
                "source": self.baseline.source,
                "caveat": self.baseline.caveat,
                "note": self.baseline.percentile_note(self.similarity_incl_quotes),
            },
            "turkish_ai_presence": self.ai_presence.to_dict(),
            "english_baseline": (
                replace(self.english_baseline, similarity_for_note=self.similarity_incl_quotes).to_dict()
                if self.english_baseline
                else None
            ),
            "applied_baseline": (
                "english"
                if self.english_baseline is not None
                else "turkish"
            ),
            "language_coverage": (
                self.language_coverage.to_dict() if self.language_coverage else None
            ),
        }


def build_similarity_report(
    *,
    tokens: Sequence[Token],
    matches: Sequence[MatchSpan],
    total_words: int,
    reference_token_indices: frozenset[int] | set[int],
    quote_spans: Sequence[Span] = (),
    statuses: dict[int, CitationStatus] | None = None,
    min_match_words: int = 5,
    document_language: str = "tr",
    exclude_references: bool = True,
    include_quotes: bool = True,
    source_languages: Mapping[str, str] | None = None,
) -> SimilarityReport:
    """Compute the institutional-style percentages.

    ``statuses`` maps ``id(match)`` -> :class:`CitationStatus`; the attribution
    split is what keeps a correctly quoted passage from counting as misconduct.
    """
    statuses = statuses or {}
    starts = [token.char_start for token in tokens]
    covered: set[int] = set()
    per_source_covered: dict[str, set[int]] = {}
    match_count_by_source: dict[str, int] = {}
    uncited_by_source: dict[str, int] = {}
    quoted_by_source: dict[str, set[int]] = {}
    unattributed_by_source: dict[str, set[int]] = {}
    reference_indices = set(reference_token_indices) if exclude_references else set()

    # Computed either way: with include_quotes=False the quoted words still have
    # to leave the numbers, otherwise the flag does nothing.
    quoted_indices = _indices_in_spans(starts, quote_spans)
    counted_quotes = quoted_indices if include_quotes else set()
    kept: list[MatchSpan] = []
    for match in matches:
        if match.matched_words < min_match_words:
            continue
        indices = _indices_for_span(starts, match.doc_char_start, match.doc_char_end)
        if not indices:
            continue
        if reference_indices and indices <= reference_indices:
            # A match that lives entirely inside the reference list is an
            # artefact of comparing against sources, not a finding.
            continue
        kept.append(match)
        covered |= indices
        per_source_covered.setdefault(match.source_id, set()).update(indices)
        match_count_by_source[match.source_id] = match_count_by_source.get(match.source_id, 0) + 1
        status = statuses.get(id(match))
        if status is CitationStatus.NOT_CITED_OR_QUOTED:
            uncited_by_source[match.source_id] = uncited_by_source.get(match.source_id, 0) + 1
        quote_hits = indices & counted_quotes
        if quote_hits:
            quoted_by_source.setdefault(match.source_id, set()).update(quote_hits)
        if status is not CitationStatus.CITED_AND_QUOTED:
            unattributed_by_source.setdefault(match.source_id, set()).update(
                indices - quote_hits if include_quotes else indices
            )

    denominator = max(1, total_words)
    excl_ref = covered - reference_indices
    excl_ref_quotes = excl_ref - quoted_indices
    matched_quotes = covered & quoted_indices
    # With --excl-quotes the quoted words leave the report entirely, so both
    # headline numbers are the same by construction.
    headline = covered if include_quotes else excl_ref_quotes

    cases = _count_cases(kept)
    stats = SimilarityStats(
        total_words=total_words,
        matched_words=len(covered),
        matched_words_quoted=len(matched_quotes),
        matched_words_in_references=len(covered & reference_indices),
        reference_words=len(reference_indices),
        quoted_words=len(quoted_indices),
        match_count=len(kept),
        cases=cases,
        granularity=round(len(kept) / cases, 3) if cases else 0.0,
    )

    per_source = tuple(
        SourceShare(
            source_id=source_id,
            name=next(m.source_name for m in kept if m.source_id == source_id),
            matched_words=len(indices),
            share_percent=100.0 * len(indices) / denominator,
            match_count=match_count_by_source.get(source_id, 0),
            quoted_words=len(quoted_by_source.get(source_id, set())),
            unattributed_words=len(unattributed_by_source.get(source_id, set())),
            uncited_matches=uncited_by_source.get(source_id, 0),
        )
        for source_id, indices in sorted(per_source_covered.items(), key=lambda kv: -len(kv[1]))
    )
    largest = per_source[0] if per_source else None

    report = SimilarityReport(
        similarity_incl_quotes=100.0 * len(headline) / denominator,
        similarity_excl_quotes=100.0 * len(excl_ref_quotes) / denominator,
        similarity_excl_references=100.0 * len(excl_ref) / denominator,
        similarity_excl_references_quotes=100.0 * len(excl_ref_quotes) / denominator,
        stats=stats,
        per_source=per_source,
        largest_single_source=largest,
        attribution={
            status.value: sum(1 for m in kept if statuses.get(id(m)) is status)
            for status in CitationStatus
        },
        filters_applied={
            "exclude_references": exclude_references,
            "include_quotes": include_quotes,
            "min_match_words": min_match_words,
        },
    )
    if source_languages:
        report = replace(
            report,
            language_coverage=build_language_coverage(
                document_language=document_language,
                source_languages=source_languages,
                per_source_covered=per_source_covered,
            ),
        )
    if document_language.startswith("en"):
        # The English reference is three times lower than the Turkish one, so the
        # choice is a correctness issue, not a nicety.
        report = replace(report, english_baseline=EnglishBaseline())
    return replace(report, bands=_evaluate_bands(report, document_language=document_language))


def build_language_coverage(
    *,
    document_language: str,
    source_languages: Mapping[str, str],
    per_source_covered: Mapping[str, set[int]],
) -> LanguageCoverage:
    """Split the matched words by whether the source was the same language.

    Built from the per-source coverage sets rather than the match list, so the
    word counts are the ones that actually went into the percentage - a match
    excluded by the institutional filters does not inflate this.

    The *source* counts, on the other hand, cover every supplied source. A
    report that said "language not determined" because nothing matched would be
    hiding the most useful thing the reader can learn from a zero result: that
    all four of their references were in English, which is precisely why nothing
    was found.
    """
    same_words = 0
    cross_words = 0
    same_sources = 0
    cross_sources = 0
    unknown_sources = 0
    sources_by_language: dict[str, int] = {}
    words_by_language: dict[str, int] = {}

    for language in source_languages.values():
        sources_by_language[language] = sources_by_language.get(language, 0) + 1
        if language == document_language:
            same_sources += 1
        elif language == "unknown":
            unknown_sources += 1
        else:
            cross_sources += 1

    for source_id, indices in per_source_covered.items():
        language = source_languages.get(source_id, "unknown")
        words_by_language[language] = words_by_language.get(language, 0) + len(indices)
        if language == document_language:
            same_words += len(indices)
        elif language != "unknown":
            cross_words += len(indices)

    return LanguageCoverage(
        document_language=document_language,
        same_language_sources=same_sources,
        cross_language_sources=cross_sources,
        unknown_language_sources=unknown_sources,
        same_language_words=same_words,
        cross_language_words=cross_words,
        sources_by_language=sources_by_language,
        words_by_language=words_by_language,
    )


def _evaluate_bands(
    report: SimilarityReport, *, document_language: str = "tr"
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    # Only the bands written for this document's language. Applying Turkish
    # thresholds to an English thesis would be worse than useless: the measured
    # clean corpus is 9% for English and 28.7% for Turkish.
    language = "en" if document_language.startswith("en") else "tr"
    for band in INSTITUTION_BANDS:
        if band.language != language:
            continue
        issues: list[str] = []
        if band.total_excl_quotes is not None and (
            report.similarity_excl_quotes > band.total_excl_quotes
        ):
            issues.append(
                f"alıntılar hariç {report.similarity_excl_quotes:.1f}% > {band.total_excl_quotes:.0f}%"
            )
        if band.total_incl_quotes is not None and (
            report.similarity_incl_quotes > band.total_incl_quotes
        ):
            issues.append(
                f"alıntılar dahil {report.similarity_incl_quotes:.1f}% > {band.total_incl_quotes:.0f}%"
            )
        largest = report.largest_single_source
        if (
            band.single_source is not None
            and largest
            and (largest.share_percent > band.single_source)
        ):
            issues.append(
                f"tek kaynak {largest.share_percent:.1f}% > {band.single_source:.0f}%"
                f" ({largest.name})"
            )
        rows.append(
            {
                "institution": band.institution,
                "url": band.url,
                "note": band.note,
                "thresholds": {
                    "total_excl_quotes_percent": band.total_excl_quotes,
                    "total_incl_quotes_percent": band.total_incl_quotes,
                    "single_source_percent": band.single_source,
                },
                "issues": issues,
                    "no_threshold_reason": band.no_threshold_reason,
                    "language": band.language,
                "status": "uygun" if not issues else "gözden geçirilmeli",
            }
        )
    return tuple(rows)


def _indices_for_span(starts: Sequence[int], char_start: int, char_end: int) -> set[int]:
    from bisect import bisect_left, bisect_right

    first = bisect_left(starts, char_start)
    last = bisect_right(starts, char_end - 1)
    return set(range(first, max(first, last)))


def _indices_in_spans(starts: Sequence[int], spans: Sequence[Span]) -> set[int]:
    out: set[int] = set()
    for span in spans:
        out |= _indices_for_span(starts, span.char_start, span.char_end)
    return out


def _count_cases(matches: Sequence[MatchSpan]) -> int:
    """Connected clusters of matches: one copied passage = one case.

    Clusters are cut on the *document* side only. The PAN granularity penalty
    exists to punish fragmenting one true case into many detections, and a
    passage found in three sources is still one case - counting it three times
    would overstate the fragmentation it is meant to measure.
    """
    if not matches:
        return 0
    ordered = sorted(matches, key=lambda m: m.doc_char_start)
    cases = 0
    current_end = None
    for span in ordered:
        if current_end is None or span.doc_char_start > current_end:
            cases += 1
        current_end = max(current_end or 0, span.doc_char_end)
    return max(1, cases)
