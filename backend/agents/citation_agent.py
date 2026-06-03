"""
Citation Agent — generates APA, MLA, IEEE, and BibTeX citations for all papers.
"""

from __future__ import annotations

import re
from typing import List

from loguru import logger

from backend.agents.base_agent import BaseAgent
from backend.models import PaperCitation, PaperMetadata, ResearchState
from backend.services.database import DatabaseService


class CitationAgent(BaseAgent):
    """
    Generates properly formatted citations for each paper.
    Uses deterministic formatting (no LLM call needed for citations).
    """

    name = "CitationAgent"

    def __init__(self) -> None:
        super().__init__()
        self.db = DatabaseService()

    def run(self, state: ResearchState) -> ResearchState:
        start = self._log_start(state, f"Generating citations for {len(state.papers)} papers")

        citations: List[PaperCitation] = []

        for paper in state.papers:
            try:
                citation = self._generate_citation(paper)
                citations.append(citation)
            except Exception as e:
                logger.warning(f"Citation failed for {paper.paper_id}: {e}")
                citations.append(self._minimal_citation(paper))

        state.citations = citations

        if state.metrics:
            state.metrics.citation_count = len(citations)

        # Persist
        self.db.save_citations(
            session_id=state.session_id,
            citations=[c.model_dump() for c in citations],
        )

        self._log_success(state, start, f"Generated {len(citations)} citations")
        return state

    # ------------------------------------------------------------------
    # Deterministic citation formatters
    # ------------------------------------------------------------------

    def _generate_citation(self, paper: PaperMetadata) -> PaperCitation:
        authors = paper.authors
        year = paper.year
        title = paper.title
        url = paper.url
        doi = paper.doi or ""
        journal = paper.journal_ref or "arXiv preprint"
        arxiv_id = paper.paper_id

        apa = self._format_apa(authors, year, title, journal, arxiv_id, doi)
        mla = self._format_mla(authors, title, journal, year, url)
        ieee = self._format_ieee(authors, title, journal, year, arxiv_id)
        bibtex = self._format_bibtex(authors, title, year, arxiv_id, doi)

        return PaperCitation(
            paper_id=paper.paper_id,
            title=title,
            apa=apa,
            mla=mla,
            ieee=ieee,
            bibtex=bibtex,
        )

    # ------ APA ------

    def _format_apa(
        self,
        authors: List[str],
        year: int,
        title: str,
        journal: str,
        arxiv_id: str,
        doi: str,
    ) -> str:
        author_str = self._apa_authors(authors)
        id_part = f"https://doi.org/{doi}" if doi else f"https://arxiv.org/abs/{arxiv_id}"
        return f"{author_str} ({year}). {title}. {journal}. {id_part}"

    def _apa_authors(self, authors: List[str]) -> str:
        if not authors:
            return "Unknown Author"
        formatted: List[str] = []
        for a in authors[:6]:
            parts = a.strip().split()
            if len(parts) >= 2:
                last = parts[-1]
                initials = ". ".join(p[0].upper() for p in parts[:-1]) + "."
                formatted.append(f"{last}, {initials}")
            else:
                formatted.append(a)
        result = ", ".join(formatted)
        if len(authors) > 6:
            result += ", ... " + self._apa_authors([authors[-1]])
        return result

    # ------ MLA ------

    def _format_mla(
        self,
        authors: List[str],
        title: str,
        journal: str,
        year: int,
        url: str,
    ) -> str:
        if not authors:
            author_str = "Unknown Author"
        elif len(authors) == 1:
            author_str = self._mla_name(authors[0])
        elif len(authors) == 2:
            author_str = f"{self._mla_name(authors[0])}, and {authors[1]}"
        else:
            author_str = f"{self._mla_name(authors[0])}, et al."
        return f'{author_str}. "{title}." {journal}, {year}, {url}.'

    def _mla_name(self, name: str) -> str:
        parts = name.strip().split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])}"
        return name

    # ------ IEEE ------

    def _format_ieee(
        self,
        authors: List[str],
        title: str,
        journal: str,
        year: int,
        arxiv_id: str,
    ) -> str:
        if not authors:
            author_str = "Anon."
        else:
            ieee_authors = []
            for a in authors[:3]:
                parts = a.strip().split()
                if len(parts) >= 2:
                    initials = ". ".join(p[0].upper() for p in parts[:-1])
                    ieee_authors.append(f"{initials}. {parts[-1]}")
                else:
                    ieee_authors.append(a)
            author_str = ", ".join(ieee_authors)
            if len(authors) > 3:
                author_str += " et al."
        return f'{author_str}, "{title}," {journal}, {year}. arXiv:{arxiv_id}.'

    # ------ BibTeX ------

    def _format_bibtex(
        self,
        authors: List[str],
        title: str,
        year: int,
        arxiv_id: str,
        doi: str,
    ) -> str:
        key = self._bibtex_key(authors, year, title)
        author_str = " and ".join(authors[:5])
        doi_line = f"  doi = {{{doi}}},\n" if doi else ""
        return (
            f"@article{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{author_str}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{arXiv preprint}},\n"
            f"  eprint = {{{arxiv_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n"
            f"{doi_line}"
            f"  url = {{https://arxiv.org/abs/{arxiv_id}}}\n"
            f"}}"
        )

    def _bibtex_key(self, authors: List[str], year: int, title: str) -> str:
        first_author_last = "anon"
        if authors:
            parts = authors[0].strip().split()
            if parts:
                first_author_last = re.sub(r"[^a-zA-Z]", "", parts[-1]).lower()

        title_word = re.sub(r"[^a-zA-Z]", "", title.split()[0]).lower() if title else "untitled"
        return f"{first_author_last}{year}{title_word}"

    def _minimal_citation(self, paper: PaperMetadata) -> PaperCitation:
        return PaperCitation(
            paper_id=paper.paper_id,
            title=paper.title,
            apa=f"Authors ({paper.year}). {paper.title}. arXiv:{paper.paper_id}.",
            mla=f'Authors. "{paper.title}." arXiv, {paper.year}.',
            ieee=f'Authors, "{paper.title}," arXiv:{paper.paper_id}, {paper.year}.',
            bibtex=f"@article{{paper{paper.year},\n  title={{{paper.title}}}\n}}",
        )
