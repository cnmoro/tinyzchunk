"""Build a diverse structured-document training corpus (Q&A, schedules, bios,
rosters, sections) from sources OTHER than the target held-out files, so the
distilled chunker generalizes contextually instead of memorizing.

Sources:
  - AllTripletsMsMarco-PTBR (cached): real PT-BR queries+passages -> Q&A Q&A scripts
  - synthetic schedules / rosters (realistic, varied formats, PT)
  - wikipedia-ptbr title+paragraph docs for header/section patterns
"""
import glob
import json
import random
import re

import pyarrow.parquet as pq

random.seed(2024)
OUT = "data/struct_corpus.jsonl"
HF = "/home/moro/.cache/huggingface/hub"

QWORD_RE = re.compile(r"^(o que|qual|quais|quem|quando|onde|como|por que|quanto|quantos|quantas|existe|existem|tem|há|pode|posso|poderia)\b", re.I)


def clean(t):
    t = re.sub(r"\s+", " ", str(t)).strip()
    return t


def load_triplet_sample(n=40000):
    files = sorted(glob.glob(f"{HF}/datasets--cnmoro--AllTripletsMsMarco-PTBR/snapshots/*/data/train-0000*.parquet"))
    rows = []
    for f in files[:3]:
        pf = pq.ParquetFile(f)
        tbl = pf.read_row_group(0, columns=["anchor", "positive"])
        rows.extend(zip(tbl["anchor"].to_pylist(), tbl["positive"].to_pylist()))
        if len(rows) >= n:
            break
    random.shuffle(rows)
    return rows[:n]


def add_qa_scripts(docs, rows):
    """Question-line + answer-paragraph Q&A-script-style docs (PT)."""
    pairs = []
    for a, p in rows:
        a = clean(a)
        p = clean(p)
        if not a or not p or len(p) < 120 or len(p) > 500:
            continue
        if not (a.endswith("?") or QWORD_RE.match(a)):
            continue
        if not a.endswith("?"):
            a = a.rstrip(".") + "?"
        pairs.append((a, p))
    random.shuffle(pairs)
    # build docs of 4-7 Q&A pairs
    k = 0
    while k + 4 <= len(pairs) and len(docs) < 220:
        group = pairs[k:k + random.randint(4, 7)]
        k += len(group)
        lines = []
        for q, a in group:
            if random.random() < 0.4:
                lines.append(f"{q} {a[: min(len(a), random.randint(60, 200))]}." if not a.endswith(("?", ".")) else f"{q} {a[: min(len(a), random.randint(60, 200))]}")
            else:
                lines.append(f"{q}\n{a}")
        docs.append({"lang": "pt", "source": "qa", "text": "\n\n".join(lines)})
    print(f"qa docs: {sum(1 for d in docs if d['source']=='qa')}")


def add_bio_docs(docs, rows):
    """short anchor as heading + passage (bios/sections)."""
    bios = []
    for a, p in rows:
        a = clean(a)
        p = clean(p)
        if not a or not p or len(p) < 180 or len(p) > 600:
            continue
        if QWORD_RE.match(a) or a.endswith("?"):
            continue
        if len(a) > 50:
            a = " ".join(a.split()[:6]).title()
        bios.append((a, p))
    random.shuffle(bios)
    k = 0
    while k + 3 <= len(bios) and len(docs) < 300:
        group = bios[k:k + random.randint(3, 5)]
        k += len(group)
        lines = []
        for h, p in group:
            lines.append(f"{h}\n{p}")
        docs.append({"lang": "pt", "source": "bio", "text": "\n\n".join(lines)})
    print(f"bio docs: {sum(1 for d in docs if d['source']=='bio')}")


NAMES = ["Ana Souza", "Carlos Lima", "Mariana Costa", "João Pereira", "Fernanda Alves",
         "Ricardo Nunes", "Beatriz Rocha", "Paulo Mendes", "Camila Duarte", "André Cardoso"]
CITIES = ["Curitiba", "Londrina", "Maringá", "Cascavel", "Foz do Iguaçu", "Ponta Grossa",
          "Guarapuava", "Umuarama", "Paranaguá", "Campo Mourão"]
TOPICS = ["Inteligência Artificial", "Marketing Digital", "Gestão Financeira", "Vendas B2B",
          "Inovação Aberta", "Experiência do Cliente", "Transformação Digital", "Liderança",
          "E-commerce", "Redes Sociais", "Comércio Exterior", "Franquias"]
ROOMS = ["Auditório Principal", "Sala 2", "Sala 3", "Plenária", "Espaço Criativo"]
PEOPLE = [f"{random.choice(NAMES)}" for _ in range(40)]


def add_schedules(docs):
    """Synthetic, varied PT program/agenda docs."""
    for _ in range(30):
        style = random.choice(["markdown", "plain", "table"])
        days = random.randint(2, 4)
        n_entries = random.randint(3, 8)
        if style == "markdown":
            out = [f"# Programação {random.choice(TOPICS)} 2024", ""]
            for d in range(1, days + 1):
                day = random.randint(20, 30)
                out.append(f"## {day:02d}/09/2024 - {random.choice(ROOMS)}")
                out.append("")
                for _ in range(n_entries):
                    h = random.randint(9, 19)
                    m = random.choice(["00", "30"])
                    out.append(f"- **Horário:** {h:02d}:{m}")
                    out.append(f"- **Palestra:** {random.choice(TOPICS)} e {random.choice(TOPICS).lower()}")
                    out.append(f"- **Palestrante:** {random.choice(PEOPLE)}")
                    out.append("")
                out.append("---")
                out.append("")
        elif style == "plain":
            out = [f"PROGRAMAÇÃO - {random.choice(TOPICS)}", ""]
            for d in range(1, days + 1):
                day = random.randint(20, 30)
                out.append(f"Dia {day}/09")
                for _ in range(n_entries):
                    h = random.randint(9, 19)
                    m = random.choice(["00", "30"])
                    out.append(f"{h:02d}:{m} - {random.choice(TOPICS)} - {random.choice(PEOPLE)}")
                out.append("")
        else:
            out = [f"AGENDA - {random.choice(TOPICS)}", ""]
            for d in range(1, days + 1):
                day = random.randint(20, 30)
                out.append(f"Data: {day:02d}/09/2024")
                for _ in range(n_entries):
                    h = random.randint(9, 19)
                    m = random.choice(["00", "30"])
                    out.append(f"Horário: {h:02d}:{m}  Tema: {random.choice(TOPICS)}  Local: {random.choice(ROOMS)}")
                out.append("")
        docs.append({"lang": "pt", "source": "schedule", "text": "\n".join(out).strip()})
    print(f"schedule docs: {sum(1 for d in docs if d['source']=='schedule')}")


def add_rosters(docs):
    """Synthetic PT contact-roster docs."""
    for _ in range(20):
        n = random.randint(4, 9)
        blocks = []
        for i in range(n):
            blocks.append(
                "Nome: " + random.choice(PEOPLE) + "\n"
                "Município: " + random.choice(CITIES) + "\n"
                "Telefone: " + random.choice(["41 99999-0000", "(42) 3222-0000", "44 99988-0000"]) + "\n"
                "E-mail: " + "contato" + str(i) + "@empresa.com.br"
            )
        docs.append({"lang": "pt", "source": "roster", "text": "\n\n".join(blocks)})
    print(f"roster docs: {sum(1 for d in docs if d['source']=='roster')}")


def main():
    rows = load_triplet_sample()
    docs = []
    add_qa_scripts(docs, rows)
    add_bio_docs(docs, rows)
    add_schedules(docs)
    add_rosters(docs)
    random.shuffle(docs)
    with open(OUT, "w") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"doc_id": i, **d}, ensure_ascii=False) + "\n")
    srcs = {}
    for d in docs:
        srcs[d["source"]] = srcs.get(d["source"], 0) + 1
    lens = [len(d["text"]) for d in docs]
    print(f"TOTAL {len(docs)} docs, {srcs}, mean len {sum(lens)//len(lens)}")


if __name__ == "__main__":
    main()
