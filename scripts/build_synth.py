"""Comprehensive synthetic training corpus (English + Brazilian Portuguese).

Generates a large, balanced set of documents covering the pattern CONFLICTS a
chunker has to resolve:

  line starts that ARE units      -> Q&A scripts, schedules, headers, list
                                     entries, chat turns, legal articles, ...
  line starts that are NOT units  -> wrapped prose, dense field lists, table
                                     rows, code lines, roster blocks, ...

All labels are constructed deterministically from the known structure (the
"teacher" here is the document's own generator), so this runs on CPU without an
LLM.  All names, companies and topics are invented placeholders.
"""
import argparse
import collections
import json
import os
import random
import re

OUT = "data/synth_labels/labels.jsonl"

# ------------------------------------------------------------------ name pools
# entirely fictional placeholders
NAMES_PT = ["Ana Souza", "Carlos Lima", "Mariana Costa", "João Pereira",
            "Fernanda Alves", "Ricardo Nunes", "Beatriz Rocha", "Paulo Mendes",
            "Camila Duarte", "André Cardoso", "Luiza Barros", "Tiago Ramos"]
NAMES_EN = ["Alice Turner", "Brian Walsh", "Chloe Bennett", "Daniel Ortiz",
            "Emma Lindqvist", "Frank Osei", "Grace Kowalski", "Henry Nakamura",
            "Isabel Ferreira", "Jonas Bergman", "Karen Whitfield", "Liam Doyle"]
ORGS = ["Northwind Labs", "Acme Analytics", "Blue Harbor", "Cedar Systems",
        "Delta Foundry", "Everline", "Fairmount Group", "Granite Works",
        "Harbor & Co", "Ironwood Studio", "Juniper Retail", "Keystone Digital"]
ROOMS_PT = ["Auditório Principal", "Sala 2", "Sala 3", "Plenária", "Espaço Criativo"]
ROOMS_EN = ["Main Auditorium", "Room 2", "Room 3", "Plenary Hall", "Studio A"]
TOPICS_PT = ["Inteligência Artificial", "Marketing Digital", "Gestão Financeira",
             "Vendas B2B", "Inovação Aberta", "Experiência do Cliente",
             "Transformação Digital", "Liderança", "Comércio Eletrônico",
             "Redes Sociais", "Comércio Exterior", "Franquias"]
TOPICS_EN = ["Artificial Intelligence", "Digital Marketing", "Financial Planning",
             "B2B Sales", "Open Innovation", "Customer Experience",
             "Digital Transformation", "Leadership", "E-commerce",
             "Social Media", "International Trade", "Franchising"]
FIELDS_PT = ["Empatia", "Respeito", "Diálogo", "Integridade", "Responsabilidade",
             "Inovação", "Imagem", "Consciência", "Coerência", "Transparência"]
FIELDS_EN = ["Empathy", "Respect", "Dialogue", "Integrity", "Accountability",
             "Innovation", "Reputation", "Awareness", "Consistency", "Transparency"]

Q_PT = [
    "Qual o nome do espaço ou processo?", "Quem é o responsável?",
    "Estará disponível durante todos os horários?", "Onde estará localizado?",
    "O que é?", "Qual o objetivo?", "A quem é destinado?",
    "Conte um pouco como funcionará seu espaço.",
    "Existe limitação de capacidade ou regras de acesso?",
    "Existe inscrição ou fila de espera?", "Existe programação específica?",
    "Quais são os requisitos para participação?",
    "É acessível para pessoas com deficiência?",
    "Algo mais que seja importante compartilhar?",
    "Há algum arquivo que o cliente pode acessar?",
]
Q_EN = [
    "What is the name of the space or process?", "Who is responsible for it?",
    "Will it be available at all times?", "Where will it be located?",
    "What is it?", "What is the goal?", "Who is it intended for?",
    "Tell us a little about how your space will work.",
    "Are there capacity limits or access rules?",
    "Is registration or a waiting list required?",
    "Is there a specific schedule?", "What are the participation requirements?",
    "Is it accessible for people with disabilities?",
    "Anything else that is important to share?",
    "Is there a file the customer can access?",
]
A_PT = [
    "Sim, estará disponível durante todo o período do evento.",
    "Não, o espaço é aberto para visitação livre.",
    "O atendimento será realizado por ordem de chegada.",
    "Será necessário realizar inscrição prévia com o responsável.",
    "A capacidade é limitada e será controlada pela equipe local.",
    "O espaço contará com monitores para orientar os visitantes.",
    "Não há restrição de idade para participação.",
    "O espaço é acessível para pessoas com deficiência.",
    "As informações adicionais serão divulgadas nos canais oficiais.",
    "O público poderá consultar a programação completa no totem de informações.",
    "A equipe estará preparada para receber e orientar todos os visitantes.",
    "Não identificamos limitações de capacidade ou logística.",
    "Será disponibilizado material informativo impresso e digital.",
]
A_EN = [
    "Yes, it will be available for the entire duration of the event.",
    "No, the space is open for free visitation.",
    "Visitors are served on a first-come, first-served basis.",
    "Prior registration with the coordinator is required.",
    "Capacity is limited and will be managed by the local team.",
    "Staff will be on hand to guide visitors through the space.",
    "There is no age restriction for participation.",
    "The space is fully accessible for people with disabilities.",
    "Additional information will be published on the official channels.",
    "The full schedule is available at the information desk.",
    "The team is prepared to welcome and assist every visitor.",
    "We have not identified any capacity or logistics constraints.",
    "Printed and digital information material will be provided.",
]


def pools(lang):
    if lang == "en":
        return dict(names=NAMES_EN, rooms=ROOMS_EN, topics=TOPICS_EN,
                    fields=FIELDS_EN, qs=Q_EN, ans=A_EN)
    return dict(names=NAMES_PT, rooms=ROOMS_PT, topics=TOPICS_PT,
                fields=FIELDS_PT, qs=Q_PT, ans=A_PT)


# ------------------------------------------------------------------- utilities

def wrap(text, width):
    words = re.findall(r"\S+", text)
    lines, cur, curlen = [], [], 0
    for w in words:
        if cur and curlen + 1 + len(w) > width:
            lines.append(" ".join(cur))
            cur, curlen = [], 0
        cur.append(w)
        curlen += len(w) + 1
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


class Doc:
    """Accumulates text while recording where each unit starts."""

    def __init__(self):
        self.parts = []
        self.n = 0
        self.bds = []

    def add(self, s, boundary=False, sep=""):
        if sep and self.n:
            self.parts.append(sep)
            self.n += len(sep)
        if boundary and self.n:
            self.bds.append(self.n)
        self.parts.append(s)
        self.n += len(s)

    def text(self):
        return "".join(self.parts)


def emit(docs, source, lang, d):
    t = d.text().rstrip()
    bds = [b for b in d.bds if 0 < b < len(t)]
    if len(t) > 60:
        docs.append({"lang": lang, "source": source, "text": t, "boundaries": bds})


# ============================ POSITIVE: line starts ARE unit starts ===========

def qa_script(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for q in random.sample(P["qs"], random.randint(5, 9)):
            a = random.choice(P["ans"])
            if random.random() < 0.4:
                a = a.rstrip(".")
            d.add(q + " " + a, boundary=True, sep="\n")
        emit(docs, "qa_script", lang, d)


def bullet_qa(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        fmt = random.choice(["bullet", "mdheader", "bullet2", "numbered"])
        d = Doc()
        for i, q in enumerate(random.sample(P["qs"], random.randint(4, 7))):
            a = random.choice(P["ans"])
            if fmt == "bullet":
                e = f"- **{q}**  \n  {a}"
            elif fmt == "bullet2":
                e = f"•   {q}\n\n    {a}"
            elif fmt == "numbered":
                e = f"{i+1}. {q}\n{a}"
            else:
                e = f"### {q}\n**{a}**"
            d.add(e, boundary=True, sep="\n")
        emit(docs, "bullet_qa", lang, d)


def schedule(docs, n, lang):
    P = pools(lang)
    LB = dict(pt=("Data", "Sala", "Horário de Início", "Palestrante", "Palestra",
                  "Detalhes completos", "Dia"),
              en=("Date", "Room", "Start Time", "Speaker", "Talk",
                  "Full details", "Day"))[lang]
    for _ in range(n):
        d = Doc()
        for _day in range(random.randint(2, 4)):
            day = random.randint(1, 28)
            room = random.choice(P["rooms"])
            style = random.choice(["md", "plain"])
            if style == "md":
                d.add(f"## {LB[5]} - {room} - {day:02d}/09/2024\n\n",
                      boundary=True, sep="\n\n")
                for _ in range(random.randint(3, 6)):
                    h, m = random.randint(9, 19), random.choice(["00", "30"])
                    d.add(f"- **{LB[0]}:** {day:02d}/09/2024  \n"
                          f"- **{LB[1]}:** {room}  \n"
                          f"- **{LB[2]}:** {h:02d}:{m}  \n"
                          f"- **{LB[3]}:** {random.choice(P['names'])}  \n"
                          f"- **{LB[4]}:** {random.choice(P['topics'])}\n",
                          boundary=True)
                d.add("\n---\n")
            else:
                d.add(f"{LB[6]} {day:02d}/09/2024 - {room}\n", boundary=True, sep="\n\n")
                for _ in range(random.randint(3, 6)):
                    h, m = random.randint(9, 19), random.choice(["00", "30"])
                    d.add(f"{LB[0]}: {day:02d}/09/2024  {LB[1]}: {room}  "
                          f"{LB[2]}: {h:02d}:{m}  {LB[3]}: {random.choice(P['names'])}  "
                          f"{LB[4]}: {random.choice(P['topics'])}\n", boundary=True)
        emit(docs, "schedule", lang, d)


def faq(docs, n, lang, prose):
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(4, 6)):
            body = random.choice(P["ans"]) + " " + random.choice(P["ans"])
            style = random.choice(["hash", "bold", "q"])
            head = {"hash": f"### {t}", "bold": f"**{t}**",
                    "q": f"{t}?"}[style]
            d.add(f"{head}\n- {wrap(body, random.randint(68, 88))}",
                  boundary=True, sep="\n\n")
        emit(docs, "faq", lang, d)


def sectioned(docs, n, lang, prose):
    P = pools(lang)
    used = 0
    for rec in prose:
        if used >= n:
            break
        paras = [p for p in re.split(r"\n\s*\n", rec["text"]) if len(p.strip()) > 120]
        if len(paras) < 3:
            continue
        d = Doc()
        for h, p in zip(random.sample(P["topics"], min(len(paras), len(P["topics"]))),
                        paras):
            style = random.choice(["plain", "hash", "setext", "upper"])
            if style == "hash":
                head = f"## {h}"
            elif style == "setext":
                head = f"{h}\n{'=' * len(h)}"
            elif style == "upper":
                head = h.upper()
            else:
                head = h
            d.add(head + "\n\n" + p, boundary=True, sep="\n\n")
        emit(docs, "sectioned", lang, d)
        used += 1


def bio(docs, n, lang):
    P = pools(lang)
    label = "Mini Bio" if lang == "pt" else "About"
    for _ in range(n):
        d = Doc()
        for nm in random.sample(ORGS, random.randint(3, 6)):
            body = random.choice(P["ans"]) + " " + random.choice(P["ans"])
            d.add(f"{label} {nm}\n\n{body}", boundary=True, sep="\n\n")
        emit(docs, "bio", lang, d)


def entity_list(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for c in random.sample(ORGS, random.randint(6, 12)):
            d.add(f"- **{c}:** {random.choice(P['ans'])}", boundary=True, sep="\n")
        emit(docs, "entity_list", lang, d)


def repeated_header(docs, n, lang):
    P = pools(lang)
    LB = dict(pt=("Apresentação", "Data", "Local", "Horário", "Palestrante", "Tema"),
              en=("Keynote", "Date", "Venue", "Time", "Speaker", "Topic"))[lang]
    for _ in range(n):
        d = Doc()
        for sp in random.sample(P["names"], random.randint(3, 6)):
            d.add(f"### {LB[0]}\n**{LB[1]}:** {random.randint(1,28)}/09/2024\n"
                  f"**{LB[2]}:** {random.choice(P['rooms'])}\n"
                  f"**{LB[3]}:** 20:00 - 21:00\n**{LB[4]}:** {sp}\n"
                  f"**{LB[5]}:** {random.choice(P['topics'])}",
                  boundary=True, sep="\n\n")
        emit(docs, "repeated_header", lang, d)


def colon_header(docs, n, lang):
    P = pools(lang)
    LB = dict(pt=("Capacidade", "Público-Alvo", "Proposta de Valor", "lugares"),
              en=("Capacity", "Audience", "Value Proposition", "seats"))[lang]
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(3, 5)):
            d.add(f"### {t.upper()}:  \n"
                  f"**{LB[0]}:** {random.randint(80,300)} {LB[3]}  \n"
                  f"**{LB[1]}:** {random.choice(P['topics'])}\n"
                  f"**{LB[2]}:** {random.choice(P['ans'])}\n"
                  f"{random.choice(P['ans'])}", boundary=True, sep="\n\n")
        emit(docs, "colon_header", lang, d)


def legal_articles(docs, n, lang, prose):
    """Numbered legal structure: each article is a unit, its paragraphs are not."""
    art = "Art." if lang == "pt" else "Section"
    par = "Parágrafo único." if lang == "pt" else "Sole paragraph."
    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    src = [p for rec in prose for p in re.split(r"\n\s*\n", rec["text"])
           if 150 < len(p.strip()) < 700]
    if not src:
        return
    for _ in range(n):
        d = Doc()
        for a in range(1, random.randint(4, 9)):
            body = wrap(random.choice(src), random.randint(70, 95))
            suffix = "º" if (lang == "pt" and a <= 9) else ""
            d.add(f"{art} {a}{suffix} {body}", boundary=True, sep="\n\n")
            if random.random() < 0.5:
                for r in romans[:random.randint(2, 4)]:
                    d.add(f"\n{r} - " + wrap(random.choice(src)[:180],
                                             random.randint(60, 90)))
            if random.random() < 0.3:
                d.add("\n" + par + " " + wrap(random.choice(src)[:200], 80))
        emit(docs, "legal_articles", lang, d)


def chat_log(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        style = random.choice(["bracket", "plain", "angle"])
        d = Doc()
        for _ in range(random.randint(6, 14)):
            who = random.choice(P["names"]).split()[0]
            msg = random.choice(P["ans"])
            h, m = random.randint(9, 18), random.randint(0, 59)
            if style == "bracket":
                line = f"[{h:02d}:{m:02d}] {who}: {msg}"
            elif style == "angle":
                line = f"<{who}> {msg}"
            else:
                line = f"{who}: {msg}"
            d.add(line, boundary=True, sep="\n")
        emit(docs, "chat_log", lang, d)


def email_thread(docs, n, lang):
    P = pools(lang)
    LB = dict(pt=("De", "Para", "Assunto", "Data"),
              en=("From", "To", "Subject", "Date"))[lang]
    for _ in range(n):
        d = Doc()
        for _ in range(random.randint(2, 5)):
            a, b = random.sample(P["names"], 2)
            body = wrap(" ".join(random.sample(P["ans"], 3)), random.randint(60, 88))
            d.add(f"{LB[0]}: {a} <{a.split()[0].lower()}@example.com>\n"
                  f"{LB[1]}: {b} <{b.split()[0].lower()}@example.com>\n"
                  f"{LB[2]}: {random.choice(P['topics'])}\n"
                  f"{LB[3]}: {random.randint(1,28)}/09/2024 {random.randint(8,18)}:00\n\n"
                  f"{body}", boundary=True, sep="\n\n")
        emit(docs, "email_thread", lang, d)


def changelog(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for v in range(random.randint(3, 7)):
            items = "\n".join(f"- {random.choice(P['ans'])}"
                              for _ in range(random.randint(2, 5)))
            d.add(f"## v1.{random.randint(0,9)}.{v} - 2024-0{random.randint(1,9)}-"
                  f"{random.randint(10,28)}\n\n{items}", boundary=True, sep="\n\n")
        emit(docs, "changelog", lang, d)


def readme_doc(docs, n, lang, prose):
    """Mixed markdown: headings, prose, fenced code, tables, lists."""
    P = pools(lang)
    src = [p for rec in prose for p in re.split(r"\n\s*\n", rec["text"])
           if 120 < len(p.strip()) < 600]
    code = ["def main():\n    total = 0\n    for i in range(10):\n        total += i\n    return total",
            "import numpy as np\n\nx = np.arange(10)\nprint(x.mean())",
            "$ pip install package\n$ package --help",
            "{\n  \"name\": \"demo\",\n  \"version\": \"1.0.0\"\n}",
            "SELECT id, name\nFROM customers\nWHERE active = 1;"]
    if not src:
        return
    for _ in range(n):
        d = Doc()
        for h in random.sample(P["topics"], random.randint(3, 6)):
            body = [random.choice(src)]
            r = random.random()
            if r < 0.35:
                body.append("```\n" + random.choice(code) + "\n```")
            elif r < 0.55:
                cols = ["| " + " | ".join(random.choice(P["topics"]).split()[:2]) + " | Value |",
                        "| --- | --- |"]
                cols += [f"| {random.choice(ORGS)} | {random.randint(1,99)} |"
                         for _ in range(random.randint(2, 5))]
                body.append("\n".join(cols))
            elif r < 0.75:
                body.append("\n".join(f"- {random.choice(P['ans'])}"
                                      for _ in range(random.randint(2, 4))))
            d.add(f"## {h}\n\n" + "\n\n".join(body), boundary=True, sep="\n\n")
        emit(docs, "readme", lang, d)


def steps_doc(docs, n, lang):
    """Numbered procedure: the heading starts a unit, the steps do not."""
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(3, 5)):
            steps = "\n".join(f"{i+1}. {random.choice(P['ans'])}"
                              for i in range(random.randint(3, 6)))
            intro = random.choice(P["ans"])
            d.add(f"### {t}\n\n{intro}\n\n{steps}", boundary=True, sep="\n\n")
        emit(docs, "steps", lang, d)


# ======================= NEGATIVE: line starts are NOT unit starts ============

def wrapped_prose(docs, n, lang, prose, narrow=False):
    used = 0
    src = "narrow_wrapped" if narrow else "wrapped"
    for rec in prose:
        if used >= n:
            break
        paras = [p for p in re.split(r"\n\s*\n", rec["text"]) if p.strip()]
        if len(paras) < 2:
            continue
        w = random.randint(48, 66) if narrow else random.randint(70, 95)
        d = Doc()
        for p in paras:
            d.add(wrap(p, w), boundary=True, sep="\n\n")
            if narrow and random.random() < 0.4:
                d.add("\n\n" + str(random.randint(1, 40)))
        if d.n > 4000:
            continue
        emit(docs, src, lang, d)
        used += 1


def hyphenated_wrap(docs, n, lang, prose):
    """PDF hyphenation: words broken across lines with a trailing hyphen."""
    used = 0
    for rec in prose:
        if used >= n:
            break
        paras = [p for p in re.split(r"\n\s*\n", rec["text"]) if len(p.strip()) > 200]
        if len(paras) < 2:
            continue
        d = Doc()
        for p in paras[:4]:
            lines = wrap(p, random.randint(45, 60)).split("\n")
            out = []
            for ln in lines:
                if random.random() < 0.35 and len(ln.split()[-1:] or [""]) and \
                        len(ln.split()[-1]) > 6:
                    w = ln.split()[-1]
                    cut = random.randint(3, len(w) - 3)
                    out.append(" ".join(ln.split()[:-1] + [w[:cut] + "-"]))
                    out.append(w[cut:])
                else:
                    out.append(ln)
            d.add("\n".join(out), boundary=True, sep="\n\n")
        if d.n > 3500:
            continue
        emit(docs, "hyphenated", lang, d)
        used += 1


def dense_field(docs, n, lang, prose):
    """'Field: long definition' lists; only section titles are units."""
    P = pools(lang)
    used = 0
    for rec in prose:
        if used >= n:
            break
        paras = [p for p in re.split(r"\n\s*\n", rec["text"]) if len(p.strip()) > 150]
        if len(paras) < 2:
            continue
        d = Doc()
        for p in paras:
            d.add(random.choice(P["topics"]) + "\n", boundary=True, sep="\n\n")
            words = p.split()
            k = random.randint(3, 5)
            step = max(len(words) // k, 1)
            for i in range(0, len(words), step):
                seg = " ".join(words[i:i + step])
                if seg:
                    d.add(f"{random.choice(P['fields'])}: {seg}\n")
        emit(docs, "dense_field", lang, d)
        used += 1


def roster_block(docs, n, lang):
    P = pools(lang)
    LB = dict(pt=("Nome", "Município", "Telefone", "E-mail"),
              en=("Name", "City", "Phone", "Email"))[lang]
    cities = ["Riverton", "Ashford", "Belmont", "Clearwater", "Dunmore"]
    for _ in range(n):
        d = Doc()
        for i in range(random.randint(3, 5)):
            d.add(f"{LB[0]}: {random.choice(P['names'])}\n"
                  f"{LB[1]}: {random.choice(cities)}\n"
                  f"{LB[2]}: (41) 9999-{random.randint(1000, 9999)}\n"
                  f"{LB[3]}: contact{i}@example.com", boundary=True, sep="\n\n")
        emit(docs, "roster_block", lang, d)


def contact_block(docs, n, lang):
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for i in range(random.randint(3, 6)):
            block = (f"Contact details for the regional coordinator, "
                     f"{random.choice(ORGS)}: Territory: "
                     f"{random.choice(['North', 'South', 'Coast'])}; "
                     f"Name: {random.choice(P['names'])}; "
                     f"Phone: (41) 9{random.randint(1000,9999)}-{random.randint(1000,9999)}; "
                     f"Email: contact{i}@example.com")
            d.add(wrap(block, random.randint(45, 65)), boundary=True, sep="\n\n")
        emit(docs, "contact_block", lang, d)


def wrapped_qa(docs, n, lang):
    """Q&A where the answer wraps: only question lines are units."""
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for q in random.sample(P["qs"], random.randint(4, 6)):
            a = wrap(random.choice(P["ans"]) + " " + random.choice(P["ans"]),
                     random.randint(50, 70))
            d.add(f"{q}\n{a}", boundary=True, sep="\n")
        emit(docs, "wrapped_qa", lang, d)


def committee(docs, n, lang):
    """Section header + list of 'Name - Org' lines that are NOT units."""
    P = pools(lang)
    roles = ["Coordinator", "Program Lead", "Operations", "Advisor", "Volunteer"]
    heads = dict(pt=["Créditos", "Comitê Organizador", "Equipe"],
                 en=["Credits", "Organising Committee", "Team"])[lang]
    for _ in range(n):
        two_col = random.random() < 0.5
        d = Doc()
        for h in random.sample(heads, random.randint(2, 3)):
            d.add(h + "\n", boundary=True, sep="\n\n")
            for _ in range(random.randint(5, 9)):
                nm, org = random.choice(P["names"]), random.choice(ORGS)
                d.add(f"{nm} - {org:26} {random.choice(roles)}\n" if two_col
                      else f"{nm} - {org}\n")
        emit(docs, "committee", lang, d)


def table_doc(docs, n, lang, prose):
    """A table is ONE unit: its rows are never boundaries."""
    P = pools(lang)
    src = [p for rec in prose for p in re.split(r"\n\s*\n", rec["text"])
           if 100 < len(p.strip()) < 400]
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(2, 4)):
            hdr = ["| Item | " + " | ".join(random.sample(P["fields"], 2)) + " |",
                   "|------|------|------|"]
            rows = [f"| {random.choice(ORGS)} | {random.randint(1,999)} | "
                    f"{random.choice(['yes', 'no', 'n/a'])} |"
                    for _ in range(random.randint(3, 8))]
            intro = wrap(random.choice(src), 80) if src else random.choice(P["ans"])
            d.add(f"## {t}\n\n{intro}\n\n" + "\n".join(hdr + rows),
                  boundary=True, sep="\n\n")
        emit(docs, "table", lang, d)


def code_doc(docs, n, lang, prose):
    """Code inside a fence is ONE unit: its lines are never boundaries."""
    P = pools(lang)
    snippets = [
        "def process(items):\n    results = []\n    for item in items:\n"
        "        if item.valid:\n            results.append(item.value)\n    return results",
        "class Handler:\n    def __init__(self, config):\n        self.config = config\n\n"
        "    def run(self):\n        return self.config.get('mode', 'default')",
        "const app = express()\napp.get('/health', (req, res) => {\n"
        "  res.json({ ok: true })\n})\napp.listen(3000)",
        "#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  wc -l \"$f\"\ndone",
        "SELECT c.id, c.name, COUNT(o.id) AS orders\nFROM customers c\n"
        "LEFT JOIN orders o ON o.customer_id = c.id\nGROUP BY c.id, c.name\n"
        "HAVING COUNT(o.id) > 5;",
    ]
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(2, 5)):
            fence = random.choice(["```", "```python", "~~~"])
            close = "```" if fence.startswith("`") else "~~~"
            d.add(f"### {t}\n\n{random.choice(P['ans'])}\n\n"
                  f"{fence}\n{random.choice(snippets)}\n{close}",
                  boundary=True, sep="\n\n")
        emit(docs, "code", lang, d)


def bibliography(docs, n, lang, prose):
    """A reference list is ONE unit even though every line looks like a header."""
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        head = "Referências" if lang == "pt" else "References"
        body = random.choice(P["ans"])
        d.add(f"{random.choice(P['topics'])}\n\n{wrap(body + ' ' + random.choice(P['ans']), 80)}")
        refs = []
        for _ in range(random.randint(4, 10)):
            nm = random.choice(P["names"]).split()
            refs.append(f"{nm[1].upper()}, {nm[0][0]}. {random.choice(P['topics'])}. "
                        f"{random.choice(ORGS)}, {random.randint(1990, 2024)}.")
        d.add(head + "\n" + "\n".join(refs), boundary=True, sep="\n\n")
        emit(docs, "bibliography", lang, d)


def csv_block(docs, n, lang):
    """Delimited data rows are never boundaries."""
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(2, 3)):
            rows = ["id,name,value,active"]
            rows += [f"{i},{random.choice(ORGS).replace(' ', '_')},"
                     f"{random.randint(1,999)},{random.choice(['true','false'])}"
                     for i in range(random.randint(4, 12))]
            d.add(f"{t}\n" + "\n".join(rows), boundary=True, sep="\n\n")
        emit(docs, "csv", lang, d)


def bullet_group(docs, n, lang):
    """A short bullet list belongs with the paragraph that introduces it."""
    P = pools(lang)
    for _ in range(n):
        d = Doc()
        for t in random.sample(P["topics"], random.randint(3, 5)):
            bullets = "\n".join(f"- {random.choice(P['ans'])}"
                                for _ in range(random.randint(2, 5)))
            d.add(f"{t}\n\n{wrap(random.choice(P['ans']), 80)}\n\n{bullets}",
                  boundary=True, sep="\n\n")
        emit(docs, "bullet_group", lang, d)


def single_line(docs, n, lang, prose):
    """No line structure at all: forces the char-level sentence fallback."""
    for rec in random.sample(prose, min(n, len(prose))):
        flat = re.sub(r"\s+", " ", rec["text"]).strip()
        if len(flat) < 400:
            continue
        d = Doc()
        d.add(flat)
        emit(docs, "single_line", lang, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--prose", default="data/labels/labels.jsonl")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    random.seed(args.seed)
    prose = [json.loads(l) for l in open(args.prose)]
    random.shuffle(prose)
    pt_prose = [r for r in prose if r.get("lang") == "pt"] or prose
    en_prose = [r for r in prose if r.get("lang") == "en"] or prose

    docs = []
    S = lambda k: max(int(k * args.scale), 1)  # noqa: E731
    for lang, pr in (("pt", pt_prose), ("en", en_prose)):
        # positives
        qa_script(docs, S(300), lang)
        bullet_qa(docs, S(200), lang)
        schedule(docs, S(90), lang)
        faq(docs, S(110), lang, pr)
        sectioned(docs, S(110), lang, pr)
        bio(docs, S(80), lang)
        entity_list(docs, S(80), lang)
        repeated_header(docs, S(80), lang)
        colon_header(docs, S(80), lang)
        legal_articles(docs, S(110), lang, pr)
        chat_log(docs, S(110), lang)
        email_thread(docs, S(80), lang)
        changelog(docs, S(60), lang)
        readme_doc(docs, S(140), lang, pr)
        steps_doc(docs, S(80), lang)
        # negatives
        wrapped_prose(docs, S(160), lang, pr)
        wrapped_prose(docs, S(120), lang, pr, narrow=True)
        hyphenated_wrap(docs, S(110), lang, pr)
        dense_field(docs, S(90), lang, pr)
        roster_block(docs, S(70), lang)
        contact_block(docs, S(70), lang)
        wrapped_qa(docs, S(80), lang)
        committee(docs, S(100), lang)
        table_doc(docs, S(110), lang, pr)
        code_doc(docs, S(110), lang, pr)
        bibliography(docs, S(80), lang, pr)
        csv_block(docs, S(70), lang)
        bullet_group(docs, S(80), lang)
        single_line(docs, S(60), lang, pr)

    random.shuffle(docs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for i, d in enumerate(docs):
            d["doc_id"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    srcs = collections.Counter(d["source"] for d in docs)
    langs = collections.Counter(d["lang"] for d in docs)
    print(f"TOTAL {len(docs)} docs -> {args.out}")
    print("  langs:", dict(langs))
    for k, v in sorted(srcs.items()):
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
