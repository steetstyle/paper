# ArXiv AI Assistant — Backend

Production-ready Python backend for an ArXiv research assistant.

```
[User query]
     │
     ▼
1. ArXiv Atom API ──► Parsed Metadata (title, abstract, authors, PDF/HTML URL)
     │
     ▼
2. Fetch Content ─────► [HTML rendering?] ──YES──► Download HTML ──► Markdown
     │                                                        │
     NO                                                       │
     ▼                                                        │
  Download PDF ──► MinerU (magic-pdf) ──► Markdown            │
     │                                                     │
     └──────────────────────┬──────────────────────────────┘
                            ▼
                   3. Chunking (markdown-aware, token budgeted)
                            ▼
                   4. Embeddings (OpenAI | sentence-transformers | hashing)
                            ▼
                   5. Persist ──► relational DB (papers, chunks, runs)
                            └──► vector store (pgvector | Qdrant | memory)
```

## Layering

Imports flow strictly downward; nothing below knows about anything above it.

| Layer | Responsibility | Modules |
|---|---|---|
| `api`, `cli` | Transport | `app/api/**`, `app/cli.py` |
| `services` | Use cases | `app/services/**` |
| `pipeline` | Step orchestration | `app/pipeline/**` |
| `clients`, `embeddings`, `db` | Integration | `app/clients/**`, `app/embeddings/**`, `app/db/**` |
| `domain` | Pure data + rules | `app/domain/**` |
| `infra` | Transport & storage plumbing | `app/infra/**` |

`app/container.py` is the composition root — the single place where concrete
implementations are chosen. Everything else takes its collaborators through
constructor injection, which is what makes the whole stack testable with fakes.

## Papers, authors and categories

ArXiv metadata that is *relational* is stored relationally. `authors` and
`categories` were JSON columns and became four tables:

```
papers ──< paper_authors >── authors
       └─< paper_categories >── categories
```

| Table | Holds | Why not a JSON column |
|---|---|---|
| `authors` | One row per person, deduplicated on `normalized_name` | An author's papers across every paper they wrote become reachable by one id, instead of a substring search over serialized text |
| `paper_authors` | `ordinal`, `affiliation` | The author list is a citation-order **sequence**; `affiliation` is a fact about *this* paper, not the person |
| `categories` | `code`, `parent`, `depth` | The taxonomy (`cs.CL.MS` → `cs.CL` → `cs`) is stored, so a subtree can be selected in SQL |
| `paper_categories` | `ordinal`, `is_primary` | `is_primary` is ArXiv's `arxiv:primary_category`; `ordinal` keeps the declared order |

ArXiv gives no author identifier, so `normalized_name` (case-folded,
whitespace-collapsed) is what keeps one person's papers together. `find_author`
resolves a loose string in decreasing order of confidence — full name, then
`family, given`, then a surname **only when it identifies exactly one person**,
because guessing between two `Smith`s would silently return the wrong papers.

The read side still looks like the old columns:

```python
paper.categories        # ['cs.CL', 'cs.LG']
paper.author_names      # ['Ashish Vaswani', 'Noam Shazeer']
paper.authors()         # [{'name': ..., 'affiliation': ...}, ...]
```

Filtering became an `EXISTS` rather than a JSON operator. This also fixed a
correctness bug: `Column.contains()` on plain `JSON` renders
`LIKE '%' || :value`, a substring match over the serialized text, so a category
named `cs.LG` also matched `cs.LG.MS`.

```
GET /api/v1/papers/categories              → every code, with paper counts
GET /api/v1/papers/authors                 → prolific authors
GET /api/v1/papers/authors/{name}          → every paper by one author
```

`author_names` is a property, not a method, for the same reason `categories` is:
it reads an eagerly-loaded relationship (`lazy="selectin"`), because a deferred
load from async code raises `MissingGreenlet`.

## Projects and references

Two additions on top of the corpus, both built on tables rather than columns.

### Papers are global, projects are joins

```
papers ──< project_papers >── projects
```

A project is a named collection. Importing a paper into a second project adds one
`project_papers` row — the `papers` row, its chunks and its vectors are shared, so
there is exactly one copy and no two project views can drift apart:

```console
$ paper projects new "Graph Neural Networks" -d "GNN architectures"
$ paper projects add graph-neural-networks 1706.03762 1607.06450 --ingest
$ paper projects add transformers 1706.03762 --note "ana metin"
```

The `--note` and read state live on the junction row, so they are per project:
the same paper can be "temel okumalar" in one project and unread in another.

```console
$ paper projects show graph-neural-networks
$ paper projects list
$ paper projects read transformers 1706.03762
$ paper projects rm graph-neural-networks --paper 1706.03762
```

`--ingest` is opt-in because it is the only write: anything not in the corpus is
fetched and run through the pipeline first (minutes per PDF). Without it the
command reports exactly which ids it could not resolve instead of importing
half a list. Re-importing is a no-op.

### Sessions

`paper ask --project X` answers one question and throws the scope away; the next
one needs the flag again, and there is no way to see what is in scope. Reading a
list is mostly sequencing, so `paper session` opens the project once:

```console
$ paper session sheaf-neural-networks
session ╭──────────────────────────────────────────────────────╮
        │ Sheaf Neural Networks  bge-large (BAAI/bge-large-en-v1.5)│
        ╰──────────────────────────────────────────────────────╯
(sheaf-papers) papers            # the project's papers, unread first
(sheaf-papers) set -c equations  # restrict later questions
(sheaf-papers) ask why the connection laplacian
(sheaf-papers) show 2608.08710   # that paper's chunks
(sheaf-papers) assets            # figures/tables/equations across the project
(sheaf-papers) read 2608.08710   # toggle read state
(sheaf-papers) exit
```

`set` takes `content=`, `top_k=` and `text=`, in either spelling — `set -c
equations` and `set content=equations` are the same command. Nothing in a session
silently widens to the whole corpus; that is one command away with `paper ask`.

A session survives a bad command. An unknown name, a mistyped setting or an
un-ingested paper prints the reason and leaves the prompt in place — losing your
place in a reading list because of one typo is the opposite of the point.

### The citation graph

```
papers ──< paper_references >── papers
```

`paper_references` is a self-join on `papers`, and **`cited_paper_id` is
nullable** — that is the design decision that matters. Most cited works are not
in the corpus, so a reference has to be storable before the cited paper is ever
ingested. `cited_arxiv_id` keeps the identity that makes a later resolution
possible: when that paper *is* ingested, the pending references are linked
automatically.

```console
$ paper projects refs 1706.03762 --limit 5
5 references
#  in corpus   year  reference
1  yes         2016  Jimmy Lei Ba (2016) — Layer normalization.
2  yes         2014  Dzmitry Bahdanu (2014) — Neural machine translation by …
3  1703.03906  2017  Denny Britz (2017) — Massive exploration of neural machine …

$ paper projects refs 1607.06450 --direction cited-by
Layer Normalization
cited by 1 of our papers
  1706.03762 Attention Is All You Need
```

References are extracted during ingestion from the paper's **HTML rendering**,
which arXiv prints as three `ltx_bibblock` spans per entry — authors, title, then
venue and year. Measured on 1706.03762v7: 40 references, 22 carrying an arXiv
id. The markdown extractor deliberately drops `.ltx_bibliography` because a
bibliography is not body text, so the references come from the source separately.

Every field is a lossy guess at `raw_text`, which is the only part that is always
true — so it is stored too. Extraction is **never fatal**: papers without a
machine-readable bibliography are the norm for older PDFs, and losing a citation
graph is not a reason to lose the paper.

Two details the implementation had to get right, both found by running it:

- the venue is wrapped in an *italic* span with the year sitting **after** it, so
  a non-greedy `</span>` loses the year
- some renderings print no reference number at all and instead cite inline
  (`Krizhevsky et al. [2012]`), so a naive "first `[n]` wins" reads a **year** as
  the reference number and collides on `unique(citing, ordinal)`

### HTTP and MCP

| Method | Path | Purpose |
|---|---|---|
| `GET`/`POST` | `/api/v1/projects` | List / create |
| `GET`/`DELETE` | `/api/v1/projects/{slug}` | Detail / delete (papers survive) |
| `GET`/`POST` | `/api/v1/projects/{slug}/papers` | List / import |
| `PATCH`/`DELETE` | `/api/v1/projects/{slug}/papers/{id}` | Note, read state / unlink |
| `GET` | `/api/v1/papers/{id}/references` | `?direction=references\|cited-by` |
| `GET` | `/api/v1/references/most-cited` | Corpus papers by incoming citations |

Five of the twenty-four MCP tools cover projects and references — `list_projects`,
`create_project`, `import_papers_to_project`, `list_project_papers`,
`list_references` — with `set_paper_read` and `delete_project` for working through
and cleaning up a list. `list_assets` covers figures, tables and equations (see
**Tools**).

### Assets

Figures, tables and equations are stored in three tables, so they can be listed
and filtered rather than scraped out of prose:

```console
$ paper assets 1706.03762v7 --kind figures   # caption + image URL or blob
$ paper assets 1706.03762v7 --kind equations --display
```

`--project` answers the corpus-level question — which papers in this reading list
have figures at all — instead of paging through each paper:

```console
$ paper assets --project sheaf-neural-networks
sheaf-neural-networks · 36 papers with assets

arxiv_id       figures   tables   display eq   inline
2603.14831        45        3           44      1048
2409.08036         5       25           90      2466
```

Equation search is a substring match on the stored LaTeX; `GET /api/v1/equations`
takes `?q=` plus `?project=` and `?arxiv_id=` for the same scope rules as
semantic search. For a meaning-based search restricted to formulas, use
`POST /api/v1/search/semantic` with `"content_kinds": ["equation"]`.

## Content kinds

Every chunk carries a `content_kind`, decided once by the chunker: `body`,
`abstract`, `figure`, `table`, `equation`, `reference`, `code`. It is stored
rather than recomputed, so filtering is a column comparison and composes with
every other search scope.

| Surface | Filter |
|---|---|
| CLI | `paper ask "..." --content equations --content tables` (`-C`, repeatable) |
| CLI | `paper show <id> --content figures` |
| HTTP | `POST /api/v1/search/semantic` with `"content_kinds": ["equation"]` |
| HTTP | `GET /api/v1/papers/{id}/chunks?content=equation` |
| MCP | `ask_paper_corpus(query=…, content=["eqs"])` |
| MCP | `read_chunks(arxiv_id=…, content=["figure"])` |

Plurals and short aliases are accepted (`equations`, `tables`, `figs`, `refs`,
`eqs`), and `paper show` prints the kind per chunk. An unknown value is an
**error on every surface** — CLI exit 2, HTTP 422, `{"ok": false, …}` in MCP —
because a typo that returned nothing silently is indistinguishable from a corpus
that holds no equations at all.

In MCP, `read_chunks` returns each chunk's kind as `kind`, and `total_chunks`
counts only the filtered set.

Chunks stored before the column existed are labeled by `paper kinds`:

```console
$ paper kinds               # label the chunks that have no kind yet
$ paper kinds --all         # re-label everything
$ paper kinds --report      # show the distribution, change nothing
```

Note what relabeling can and cannot do: it reads the **stored chunk text**, so it
can fix a mislabel but cannot invent content that was never extracted. Use it
after switching models or moving between spaces; use `paper ingest --force` when
the extraction itself has changed.

### Two extraction bugs that made the kinds wrong

Both were found by measuring real arXiv HTML, and both were corpus-level
correctness bugs rather than cosmetic labels.

- LaTeXML lays every **display equation** out in a three-column
  `<table class="ltx_equation">`. That reached the chunker as a markdown pipe
  table, so every display equation in an arXiv paper looked like a *table*. It is
  now lifted out through the `<math alttext>` LaTeX into real `$$ … $$` display
  math, and the classifier also recognizes the layout (empty pad cells at both
  ends, LaTeX between), so corpora ingested earlier are classified correctly too.
- `figure.ltx_figure` used to be dropped wholesale, which left the corpus with
  **zero** figure chunks. The figure is now reduced to its caption text; the
  structured figure record — image URL, page, bounding box — is still
  extracted separately from the raw HTML.

Measured on 1706.03762v7 (Attention Is All You Need): 5 figure captions and 4
table captions classify correctly, and 9 display-math blocks are recovered.

The classifier can relabel an existing corpus (`paper kinds --all`), but the
figure fix changes the **extracted markdown** — a paper ingested before it has no
figure captions in its stored text to find. Re-ingest with `--force` to pick them
up:

```bash
paper ingest 1706.03762v7 --force   # 25 chunks: 3 figure, 3 table, 3 equation
paper show 1706.03762v7 --content figures
```

## Scoped search

`paper ask` scopes by project, by paper, by content kind and by source, and the
scopes compose rather than override each other:

```console
$ paper ask "how is over-smoothing handled?" --project graph-neural-networks \
    --content equations --min-score 0.3
```

Project scope **narrows**: it never widens the result set, so a project with no
papers returns zero rows instead of the whole corpus, and it intersects with
`--paper`. The same scope exists as `"project"` on
`POST /api/v1/search/semantic`, as `project=` on `ask_paper_corpus`, and on
`GET /api/v1/equations?q=…&project=…&arxiv_id=…`.

`--source` is repeatable and validated against the extraction backends that
exist: `arxiv_html`, `ar5iv`, `pdf_mineru`, `pdf_pypdf`, `abstract_only`.
`--min-score` drops hits below a cosine score.

### Any spelling of a paper id

`--paper` and `arxiv_id` resolve a paper from anything that identifies it: a
bare arXiv id, a versioned id (`2408.05245v1`), an `arXiv:` prefix, an `abs` or
`pdf` URL, or the internal id. An id that is not ingested raises an error that
names it — CLI exit 2, HTTP 404, an MCP error — instead of quietly returning
nothing.

## The pipeline

Ten independent steps, each with its own failure boundary and timing record
(persisted to `pipeline_step_runs`):

| # | Step | Output |
|---|---|---|
| 1 | `fetch_metadata` | `PaperMetadata` from the Atom API |
| 2 | `persist_metadata` | `papers` row (stable id for everything downstream) |
| 3 | `fetch_content` | HTML blob, else PDF blob → `raw_documents` |
| 4 | `extract_text` | markdown (`extract_text`) from HTML or MinerU |
| 5 | `extract_references` | bibliography → `paper_references`, linked where possible |
| 6 | `extract_assets` | figures, tables and equations from HTML or MinerU |
| 7 | `chunk_text` | `chunks` rows with heading breadcrumbs and `content_kind` |
| 8 | `embed_chunks` | vectors → `embeddings` rows |
| 9 | `index_vectors` | vectors → vector store |
| 10 | `finalize` | stamps `ingested_at` |

Every step degrades instead of failing. MinerU unavailable or a PDF unreadable →
step 4 falls back to `abstract_only` and the run is `partial`. No machine-readable
bibliography → step 5 records nothing. Asset extraction (step 6) is never fatal
either, so a paper whose figures cannot be extracted still ingests, just without
them. A paper is never lost.

## Quick start

```bash
cp .env.example .env

# SQLite + in-memory vectors, no external services required:
export DATABASE_URL="sqlite+aiosqlite:///./.data/paper.db"
export VECTOR_BACKEND=memory
export EMBEDDING_PROVIDER=hashing

paper db upgrade
paper doctor
paper search "cat:cs.CL AND ti:transformer" --max 5
paper ingest 1706.03762
paper ask "how does multi-head attention work?" --text
```

Run the API:

```bash
paper serve --reload          # http://localhost:8000/docs
```

## Docker (Postgres + pgvector)

```bash
docker compose up -d db
export DATABASE_URL="postgresql+asyncpg://paper:paper@localhost:5432/paper"
export DATABASE_SYNC_URL="postgresql+psycopg://paper:paper@localhost:5432/paper"
export VECTOR_BACKEND=pgvector

paper db upgrade
paper ingest 1706.03762
```

Alembic runs **synchronously**, so it needs `DATABASE_SYNC_URL` (psycopg) even
though the application itself is async (asyncpg). It is derived from
`DATABASE_URL` when unset, so the two `export` lines above are belt-and-braces
rather than required.

`.env` in this repository is already pointed at the `docker compose` Postgres,
which is what `paper doctor`, `paper ingest` and `paper ask` use as-is.

`docker compose --profile app up --build` builds the API image as well.

## MinerU

MinerU converts the PDF into layout-aware Markdown. It has renamed itself and
moved its API twice, so the integration **probes what is installed and adapts**
(`app/clients/content/mineru_resolver.py`):

| Generation | Python package / CLI | Python entry points | CLI shape |
|---|---|---|---|
| 4.x | `mineru` | `doc_analyze` + `render_markdown` | `mineru parse <pdf> -o out.md --tier …` |
| 2.x | `mineru` | `read_fn` + `do_parse` | `mineru -p <pdf> -o outdir --backend …` |
| 1.x | `magic-pdf` | `magic_pdf.cli.common.do_parse` | `magic-pdf -p <pdf> -o outdir` |

```bash
pip install 'paper-app-backend[mineru]'   # + CPU/GPU torch
paper doctor                              # shows the detected generation
```

### Backends

Tried in order until one yields usable text:

| Backend | Notes |
|---|---|
| `python_api` | In-process. Uses MinerU's native `aio_*` coroutine where available |
| `cli` | Subprocess — the reliable choice inside Docker |
| `pypdf` | Last-resort text layer so ingestion degrades instead of failing |

```bash
export MINERU_BACKEND_ORDER='["python_api","cli","pypdf"]'
```

### Verified end-to-end

Against a real MinerU 2.7.6 on CPU, `paper ingest 1706.03762 --pdf` produced
29,588 characters and **153 content blocks** through the in-process API, and
byte-identical output through the CLI backend — with the full heading hierarchy
surviving into the chunks:

```
# 2  tokens=294  Attention Is All You Need > Abstract
# 5  tokens=187  Attention Is All You Need > 3 Model Architecture
# 8  tokens=371  ... > 3.2.1 Scaled Dot-Product Attention
```

### Model sources and backends

```bash
export MINERU_MODEL_SOURCE=modelscope   # huggingface | modelscope | local
export MINERU_BACKEND=pipeline          # pipeline | vlm-auto-engine | hybrid-auto-engine
export MINERU_DEVICE=cuda:0             # cpu | cuda | cuda:0 | mps
export MINERU_TIMEOUT_SECONDS=1200      # CPU-only runs are slow on first download
```

Pre-download weights to avoid a per-request download:

```bash
mineru-models-download -s modelscope -m pipeline
export MINERU_MODEL_SOURCE=local
```

### Version pinning

MinerU 2.x has two upstream conflicts that are easy to trip over; both are
already pinned in the `[mineru]` extra:

| Pin | Why |
|---|---|
| `torch<2.6` | 2.6 flipped `torch.load(weights_only=True)`, which breaks `doclayout_yolo` checkpoints |
| `transformers<4.50` | Unimernet imports `pytorch_utils` symbols removed in 4.50 |

If you hit a `No module named …` traceback during extraction, the missing
package is a MinerU transitive dep that `pip` did not resolve into your
environment; `LOG_LEVEL=DEBUG` names the exact module.

### Failure behaviour

A MinerU failure never loses a paper — the run degrades to `pypdf`, and if that
also fails, to title + abstract with the run marked `partial`:

```
python_api → cli → pypdf → abstract_only
```

## MCP server

Exposes the same corpus to Claude Desktop, Claude Code or any MCP client. It runs
**in-process** against the same `Container` the CLI and HTTP API use, so there is
no second process and one shared database.

```bash
pip install 'paper-app-backend[mcp]'
paper mcp                          # stdio
paper mcp --http --port 8080        # streamable HTTP
```

```json
{
  "mcpServers": {
    "arxiv-assistant": { "command": "paper", "args": ["mcp"] }
  }
}
```

### Tools

| Tool | Writes? | Purpose |
|---|---|---|
| `search_arxiv` | no | ArXiv metadata search (fast, no download) |
| `get_paper` | no | Canonical metadata for one id/URL |
| `ingest_paper` | **yes** | Full pipeline: fetch, MinerU, chunk, embed, index |
| `ask_paper_corpus` | no | Semantic search, scoped by `project`, `arxiv_id`, `content` |
| `read_chunks` | no | A paper's chunks by ordinal offset; `content=` filters, each chunk carries `kind` |
| `read_markdown` | no | The extracted markdown for a whole paper |
| `list_papers` | no | What is in the local corpus |
| `list_embedding_spaces` | no | Registered models and their vector counts |
| `status` | no | Database, vector store, active model, extraction backends |
| `list_projects` | no | Named collections and their paper counts |
| `create_project` | **yes** | Create a named reading list |
| `import_papers_to_project` | **yes** | Add papers to a project, ingesting them first if asked |
| `list_project_papers` | no | A project's papers, with note and read state |
| `list_references` | no | A paper's citations (`direction=references\|cited-by`) |
| `list_assets` | no | Figures/tables/equations, by `arxiv_id` **or** by `project` |
| `search_equations` | no | Substring search over stored LaTeX; `project`/`arxiv_id` scoped |
| `list_categories` | no | Every subject category in the corpus, with paper counts |
| `list_authors` | no | Most prolific authors; pass `name` to resolve one and list their papers |
| `most_cited_references` | no | Corpus papers ranked by how often the corpus cites them |
| `list_ingest_runs` | no | Ingestion history with status and failed steps |
| `set_paper_read` | **yes** | Toggle read state in a project, one paper or all |
| `delete_project` | **yes** (destructive) | Remove a reading list; the papers are kept |
| `reembed_space` | **yes** | Embed already-stored chunks into another space (no network) |
| `chunk_kinds` | **yes** | Report the `content_kind` distribution, or relabel chunks |

Plus the `paper://{arxiv_id}` and `corpus://spaces` resources.

Every CLI command has a tool equivalent, and so does every read-only HTTP route.
`delete_project` is the only tool annotated `destructive_hint`: the other writes
are idempotent, because re-ingesting or re-embedding *replaces* rather than
accumulates. That distinction is what lets a client auto-approve the cheap calls
and still ask before the one call that loses something.

Only `ingest_paper` and `ask_paper_corpus` accept `space` to pick an embedding
model, defaulting to the active one, so the assistant can A/B two corpora over
the same questions. `reembed_space` takes one because that is its whole purpose.

### Why the annotations matter

`ingest_paper` is the expensive write, so it is the tool a client should prompt
for. It is marked `read_only_hint=False` and `idempotent_hint=True` (re-ingest
replaces rather than duplicates), as are `create_project` and
`import_papers_to_project` (re-importing is a no-op); everything else is
`read_only_hint=True` — clients use that to auto-approve the cheap calls.

### Context-window discipline

Results are truncated and each payload names the tool to call for more
(`read_chunks`, `read_markdown`). Dropping a whole paper into a model's context
is the fastest way to waste its attention, so the server never does it
unprompted.

Ingestion is the one slow operation (minutes for a PDF on CPU), so it reports
progress through the protocol. `ctx.log` is not used: the SDK deprecated that
capability in SEP-2577.

### Errors

MCP has no error channel a model can reason about, so failures come back as
`{"ok": false, "error": "<what to do next>"}` rather than a traceback:

```
{"ok": false, "error": "1706.03762 is not ingested. Use ingest_paper first, or search_arxiv to find it."}
```

## Embedding spaces: one model, one table

pgvector's `vector(n)` column and its HNSW index are **width-locked**, so two
models with different dimensions cannot share a column. Rather than commit to
one model forever, each model gets its own table:

```
chunks --+-- embeddings                 default    . text-embedding-3-small  . 1536d
         +-- embeddings__small_384     experiment  . same model truncated     .  384d
         +-- embeddings__bge_m3        candidate   . BAAI/bge-m3              . 1024d
         +-- embeddings__bge_large     bge         . BAAI/bge-large-en-v1.5   . 1024d
```

Chunk text is model-independent, so it exists **once**; only vectors are
duplicated. Adding a model costs disk and one re-embed - no migration of existing
rows, and no disruption to the space you already serve.

That makes a dimension sweep cheap: build a space at 384, embed into it
alongside the 1536 incumbent, and compare them on the same chunks.

### Using them

```bash
paper spaces list                                     # what is registered
paper spaces add small-384 --model text-embedding-3-small --dims 384
paper ingest 1706.03762 --space small-384             # add vectors for that model
paper reembed --space bge-large --dry-run             # list the targets, embed nothing
paper reembed --space bge-large                       # embed chunks already stored
paper ask "how does attention work?" --space small-384
paper spaces activate small-384                       # make it the default
paper spaces sql small-384                            # DDL, for reviewed migrations
paper spaces rm small-128 --drop-table                # drop it when done
```

`paper reembed` embeds chunks that are **already** ingested into another space:
no downloads, no re-parsing, **no network at all**. Chunks live in the `chunks`
table and every space has its own vector table, so a model switch costs minutes
rather than hours. `--project` and `--paper` scope the work, `--force` re-embeds
papers the space already has, `--limit` caps it and `--dry-run` only lists the
targets — without even creating the table.

Verified by blocking every outbound socket except the local database and
re-embedding one paper into a fresh space: **25 chunks in 0.1s**, no connection
attempted. `--paper` accepts any id spelling — `1706.03762`, `1706.03762v7`,
`https://arxiv.org/abs/1706.03762v1` all resolve — because the paper ids are
resolved rather than string-matched. An id that is not in the corpus is an error
naming it, not a silent "0 papers"; re-embedding only covers papers whose text
is already stored, so a paper that was never ingested has to go through
`paper ingest` (which does reach ArXiv, and reuses the stored HTML when it can).

```bash
curl -X POST localhost:8000/api/v1/ingest \
     -d '{"arxiv_id":"1706.03762","space":"small-384"}'
curl -X POST localhost:8000/api/v1/search/semantic \
     -d '{"query":"...","space":"small-384"}'
curl localhost:8000/api/v1/embedding-spaces
```

### Verified live

Against the `bge-large` space, `paper reembed` embedded **2878 chunks from 54
papers** in ~29 minutes on CPU (~10 chunks/s) — no paper was downloaded or
parsed again. A single paper into an empty space takes well under a second.

### Guards

The registry refuses anything that would silently corrupt an index:

| Attempt | Result |
|---|---|
| Two models mapping to one table | `409`, naming the space that holds it |
| Redefining a space's model or width | rejected: vectors are not comparable |
| Removing a space that `EMBEDDING_*` owns | rejected (it is `is_locked`) |
| Querying with another model's embedding | rejected before the similarity scan |

`POST /{name}/activate` only changes which model *new* work uses. Vectors are
never moved, so switching is instant and reversible.

### Schema ownership

`VECTOR_AUTO_CREATE=true` (default) creates a space's table and HNSW index
idempotently on first use: one catalogue query per ingest, negligible next to
embedding. Set it to `false` where schema changes must be reviewed, and apply
`paper spaces sql <name>` through a migration instead.

## Embeddings

```bash
# Local, no API key, no data egress
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Local BGE (1024 dimensions)
export EMBEDDING_PROVIDER=bge
export EMBEDDING_MODEL=BAAI/bge-large-en-v1.5

# Hosted
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-small
export OPENAI_API_KEY=sk-...

# Deterministic, dependency-free (dev/CI only — not semantic)
export EMBEDDING_PROVIDER=hashing
```

The `bge` provider applies the model card's query instruction
(`Represent this sentence for searching relevant passages: `) to **queries
only**, never to passages — the two sides are embedded differently on purpose.
Dimensions are read from the loaded model rather than trusted from config.

`hashing` is not a retrieval model, whatever the name suggests: it embeds token
hashes, so it matches words rather than meaning and cannot answer "which paper
explains X" for anything but a literal overlap. Measured on the same question
(`shehir laplacian`, a misspelling of "sheaf laplacian"):

| space | top hit | score |
|---|---|---|
| `default` (hashing) | an unrelated paper, all tokens missed | 0.25 |
| `bge-large` | *Sheaf Neural Networks with Connection Laplacians* | 0.83 |

Keep it for tests and CI, where determinism matters more than meaning. The
default in `.env.example` is `bge` for that reason.

Every vector stores a `fingerprint` (`provider:model:dimensions`). When the
fingerprint changes, old vectors are detectable and can be pruned with
`EmbeddingRepository.delete_stale`.

> pgvector columns are fixed-width: `EMBEDDING_DIMENSIONS` must match the
> dimension the column was created with. Changing it requires a new migration.

## Vector backends

| `VECTOR_BACKEND` | Notes |
|---|---|
| `pgvector` | Vectors in the `embeddings` table, HNSW index, `<=>` operator |
| `qdrant` | Vectors external to Postgres; chunk text stays relational |
| `memory` | Tests and local experiments |

Non-Postgres URLs automatically fall back to `memory` with a warning instead of
failing at query time.

> `memory` holds vectors **in the process**. Each CLI invocation and each API
> worker starts empty, so `paper ingest` followed by `paper ask` will find
> nothing. Use `pgvector` (or `qdrant`) whenever you want persistence across
> processes — `memory` exists so the pipeline is runnable with zero
> infrastructure, not as a storage backend.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness + component status |
| `GET` | `/api/v1/arxiv/search` | ArXiv metadata search (see **ArXiv filters**) |
| `GET` | `/api/v1/arxiv/filter-help` | Supported filter fields and operators |
| `GET` | `/api/v1/arxiv/paper/{id}` | Single paper metadata |
| `GET` | `/api/v1/papers` | Stored papers (`?category=`, `?ingested_only=`) |
| `GET` | `/api/v1/papers/categories` | Every category, with paper counts |
| `GET` | `/api/v1/papers/authors` | Prolific authors |
| `GET` | `/api/v1/papers/authors/{name}` | Every paper by one author |
| `GET` | `/api/v1/papers/{id}` | Detail + content provenance |
| `GET` | `/api/v1/papers/{id}/chunks` | Chunk listing (`?content=` to filter by kind) |
| `GET` | `/api/v1/papers/{id}/markdown` | Extracted markdown |
| `POST` | `/api/v1/ingest` | Ingest (`?wait=true` for synchronous) |
| `GET` | `/api/v1/ingest/runs/{id}` | Per-step run status |
| `POST` | `/api/v1/search/semantic` | Vector search (`space`, `project`, `content_kinds`) |
| `GET` | `/api/v1/equations` | Formula search (`?q=`, `?project=`, `?arxiv_id=`) |
| `GET` | `/api/v1/embedding-spaces` | Registered models |
| `POST` | `/api/v1/embedding-spaces` | Register a model as its own table |
| `GET` | `/api/v1/embedding-spaces/{name}` | Detail + DDL |
| `POST` | `/api/v1/embedding-spaces/{name}/activate` | Change the default space |
| `DELETE` | `/api/v1/embedding-spaces/{name}` | Remove (`?drop_table=true` to drop vectors) |

```bash
curl -s localhost:8000/api/v1/arxiv/search?q=transformer&category=cs.CL -X GET | jq
curl -s localhost:8000/api/v1/ingest -H 'content-type: application/json' \
     -d '{"arxiv_id":"1706.03762","prefer_html":true}' | jq
curl -s localhost:8000/api/v1/search/semantic -H 'content-type: application/json' \
     -d '{"query":"self-attention complexity","top_k":5}' | jq
```

## ArXiv filters

The filter language is ArXiv's own, not a wrapper around it. Anything you can
paste into `export.arxiv.org/api/query` works here unchanged, and every surface
(CLI, HTTP, MCP) compiles it identically — they all funnel through one
`search_query_from` call, and `tests/test_filter_parity.py` asserts the three
produce byte-identical `search_query` strings.

### The language

| Prefix | Field |
|---|---|
| `ti` | title |
| `au` | author |
| `abs` | abstract |
| `co` | comment |
| `jr` | journal reference |
| `cat` | subject category |
| `rn` | report number |
| `all` | all of the above, searched simultaneously |

Operators are `AND`, `OR`, `ANDNOT` — **uppercase only**. Parentheses group,
double quotes make a phrase, and `submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]`
filters by submission time (evaluated by ArXiv, not locally).

### Adjacent words are ORed, not ANDed

This is the one thing the manual never says — it only describes a space as
extending a query "to include multiple fields" — and it is the most common way a
search returns noise. Measured against the live API:

| Query | Results |
|---|---|
| `sheaf` | 3 491 |
| `neural` | 198 684 |
| `network` | 335 204 |
| `sheaf neural network` | **394 907** ← the union |
| `sheaf AND neural AND network` | **56** |
| `"sheaf neural network"` | **33** |

A misspelt word makes it worse, not narrower: a term matching nothing is
**dropped**, not treated as zero. So `sheaf neureal network` returns 338 593
papers about 5G networks and bitcoin rather than failing — which is exactly what
`paper search` shows you today.

```console
$ paper search 'sheaf neureal network' --show-query
warning: ArXiv ORs adjacent terms instead of ANDing them, so 3 bare words
(sheaf, neureal, network) match almost anything — results are dominated by the
most common term. Quote the phrase as 'sheaf neureal network' (single quotes
outside, double quotes inside: a shell strips "sheaf neureal network" bare), or
AND the terms explicitly.
sheaf neureal network
```

Write the phrase, or the operators — the query is reported, never rewritten:

```bash
paper search '"sheaf neural network"'          # 33
paper search --phrase "sheaf neural network"   # 33, no quoting to get wrong
paper search 'ti:sheaf AND ti:neural'          # 31
paper search 'ti:sheaf AND abs:neural'         # 43
```

> **Your shell eats the quotes.** `paper search "sheaf neural network"` arrives
> with no quotes at all, and `paper search ""sheaf neural network""` is split by
> the shell into three separate arguments. Only `'…'` (single quotes outside,
> double quotes inside) or `--phrase` actually deliver a phrase. `--phrase` is
> immune to all of it, which is why it exists.

`--phrase` combines with the other filters, and never discards them: an
unquoted `--phrase a b c` reaches us as the phrase `a` plus two positional
words, and the compiled query keeps both — `"a" AND b c`.

An exact phrase that matches nothing is also reported rather than left blank,
since a typo inside quotes is otherwise indistinguishable from a missing paper:

```console
$ paper search --phrase "sheaf neureal network"
search_query: "sheaf neureal network"
hint: No paper matches this exact phrase. ArXiv phrases are literal, so a
misspelling returns nothing: check the spelling of "sheaf neureal network", or
search the words with AND, or drop the quotes.
0 shown
```

`--title a --title b` is not affected: structured fields are always compiled
with an explicit `AND`, because that join is written by us rather than by you.

> The manual writes that range separator as `+TO+`. Inside a URL `+` *is* a
> space, so any encoder emits `%2B` and ArXiv answers **500**. The compiled
> clause therefore uses a literal space, which encodes to `%20` and is what
> ArXiv actually parses — verified against the live API.

```bash
paper search 'cat:cs.CL AND ti:"attention is all you need"'
paper search '(ti:diffusion OR ti:flow) ANDNOT abs:survey --since 2024-01-01'
paper search --title transformer --title "large language models" --op OR
```

```bash
curl -sG localhost:8000/api/v1/arxiv/search \
  --data-urlencode 'filter=cat:cs.CL AND ti:"attention"' --data-urlencode 'has_pdf=true'
```

```
search_arxiv(query='(ti:diffusion OR ti:flow) ANDNOT abs:survey', submitted_from="2024-01-01")
```

### Structured parameters

Every prefix has a named equivalent, which compiles to exactly the same query:
`?title=x` is `ti:x`, `--report-number X` is `rn:X`. Names are accepted either
way (`journal` and `journal_ref` both mean `jr`), so scripts and humans can use
whichever is natural.

Two details worth knowing:

- `--op` binds values *within* one field. `--title a --title b --author ho
  --op OR` means "a or b, written by ho"; fields are always AND-ed.
- Passing both a raw expression and named fields keeps both: they are AND-ed,
  never silently dropped.

### Client-side filters

Some things ArXiv's API cannot express. These run on the parsed response:

| Parameter | Meaning |
|---|---|
| `has_pdf` / `has_html` / `has_doi` / `has_journal_ref` | entry has (or lacks) that link |
| `ingested` | only papers already in the local corpus (or only those not) |
| `also_categories` | keep entries carrying **all** of these categories |
| `exclude_categories` | drop entries carrying any of these |

They apply **after** pagination, so a page can come back shorter than
`max_results` while more matches exist. Every response reports `filtered_out`
and `next_start` so a caller can keep going.

### Validation

Hand-written queries are checked before they leave, because two mistakes cause
nearly all failures:

```
$ paper search 'ti:transformer and cat:cs.CL' --show-query
warning: ArXiv requires UPPERCASE boolean operators; found and. It may be
treating these as plain text.
ti:transformer and cat:cs.CL
```

Lowercase operators are **reported, not rewritten** — the query goes to ArXiv
exactly as typed, so you can see what it will actually do. Unknown prefixes,
unbalanced parentheses and unterminated quotes are hard errors (exit code 2,
HTTP 422, `{"ok": false, ...}` in MCP).

`id:` is also flagged: ArXiv documents `id_list` for lookups because it resolves
article versions correctly, so `paper search --id 1706.03762` is the supported
form.

That is not only a documentation preference — it is the difference between
`get_paper` working and not. Measured against the live API for the old-style id
`cond-mat/0404680v1`:

| request | HTTP | entries |
|---|---|---|
| `search_query=id:cond-mat/0404680v1` | 200 | 0 |
| `id_list=…`, no `search_query` | 200 | 1 |
| both together | 200 | 0 |

All three succeed at the HTTP level. The old-style form is not matched by the
`id:` field at all, and sending it alongside `id_list` makes ArXiv ignore
`id_list` — so the failure looked exactly like "this paper does not exist".
An `id_list` request therefore sends only `id_list`, and `get_paper` falls back
to the versionless id when a versioned one does not resolve: `cond-mat/0305062v1`
answers 500 while `cond-mat/0305062` returns the paper, and `solv-int/9712001v2`
returns nothing because v2 was never published.

Every response echoes the compiled `search_query` back, and
`GET /api/v1/arxiv/filter-help` lists the fields and operators at runtime.

## Politeness

ArXiv asks for at least one API request every three seconds. `HttpFetcher`
enforces that with a shared rate limiter, retries transient failures with
exponential backoff and jitter, caps download size, and streams PDFs to disk in
1 MiB blocks so a 40 MB paper never lands in memory.

## Tests

```bash
make test                   # pytest
make check                  # ruff + mypy + pytest
pytest --cov=app
```

688 tests (+23 skipped), up from 505; `ruff` and `mypy` are clean. ArXiv, the
HTML/PDF CDNs, MinerU, the embedding provider and the vector store are all faked,
so the suite is deterministic and the end-to-end pipeline test drives all ten
steps against a real SQLite database.

Three suites are worth knowing about because each answers a question the fakes
cannot:

| Suite | What only a real dependency can answer |
|---|---|
| `test_filter_parity.py` | Drives the CLI, HTTP API and MCP server with one filter and asserts byte-identical `search_query` output, so the three cannot drift apart |
| `test_mineru_integration.py` | Asserts against the **real** MinerU package that the resolved `do_parse` / `doc_analyze` signatures still match what we call |
| `test_pgvector_store.py`, `test_migrations.py` | Run the real Alembic chain and a real `vector` scan |

The last two need PostgreSQL and skip without it:

```bash
docker compose up -d db
TEST_PG_URL=postgresql+asyncpg://paper:paper@127.0.0.1:5432/paper pytest tests/test_pgvector_store.py tests/test_migrations.py
```

They are not optional extras. Both bugs below were invisible to the faked suite
and only appeared once a real `vector` column was scanned:

- `1.0 - distance` let SQLAlchemy infer the whole subtraction as `VECTOR`, so
  `1.0` was handed to pgvector and raised *"expected list or ndarray"*.
- `min_score` compiled to `HAVING` with no `GROUP BY`, which PostgreSQL rejects
  outright.

`test_migrations.py` also exists because `create_all` builds the schema from the
models: it cannot catch a migration that disagrees with them, which is exactly
where the `json`/GIN failure lived (`data type json has no default operator class
for access method gin`).

## Failure behaviour

The pipeline is designed to never lose a paper:

| Failure | Behaviour |
|---|---|
| No HTML rendering | Falls back to ar5iv, then the PDF |
| MinerU missing / PDF unreadable | Falls through `cli` -> `pypdf` -> title+abstract, run is `partial` |
| Vector store unreachable | Run is `failed`; the row keeps the error |
| Older ArXiv version than stored | Ignored, logged as `paper_version_ignored` |
| Re-ingest | Chunks and embeddings replaced in place, no orphans |
| Query a space with another model's embedding | `422` before the scan, not silent garbage |
| Space/table collision | Rejected at registration, naming the occupant |

## Layout

```
backend/
├── app/
│   ├── api/            # FastAPI routes, schemas, dependencies
│   ├── clients/        # arxiv/ (Atom), content/ (fetcher, MinerU, html, references, assets)
│   ├── config.py       # typed settings
│   ├── container.py    # composition root
│   ├── db/             # models, repositories (paper/project/reference), vector_store/
│   ├── domain/         # pure models, enums, ids, arxiv filters
│   ├── embeddings/     # base + openai / sentence-transformers / hashing
│   ├── infra/          # http, storage, text
│   ├── pipeline/       # context, steps, runner
│   ├── services/       # ingestion, semantic_search, arxiv_search, chunker
│   ├── mcp/            # Model Context Protocol server
│   ├── main.py         # ASGI app
│   └── cli.py          # typer CLI
├── migrations/         # Alembic
├── tests/
├── docker-compose.yml
└── pyproject.toml
```