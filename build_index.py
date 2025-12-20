import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Senin elindeki dosya bu formatta (;) ayrılmış.
# Dosya yolunu buradan değiştir:
CODES_TXT = Path("icd102019enMeta/icd102019syst_codes.txt")  # <-- dosyanın adını/yolunu buna göre ayarla

OUT_INDEX = "icd10.faiss"
OUT_META  = "icd10_meta.json"

def normalize_code(code: str) -> str:
    code = code.strip()
    # Örn: "A00.-" gibi block/range kodlarını kategoriye indir
    code = code.replace(".-", "").replace("-", "")
    return code

def parse_codes_semicolon(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Bulamadım: {path.resolve()}")

    docs = []
    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            parts = line.split(";")
            # Kısa/garip satırlar var (ör. sadece "097;4-017;...") -> skip
            if len(parts) < 9:
                continue

            # Bu dosyada genelde:
            # parts[5] = code (A00.0 veya A00.- gibi)
            # parts[8] = title (Cholera due to..., Typhoid fever, ...)
            code = normalize_code(parts[5])
            title = parts[8].strip()

            if not code or not title:
                continue

            docs.append({
                "code": code,
                "title": title,
                "text": f"{code} - {title}"
            })

    # dedupe
    seen = set()
    uniq = []
    for d in docs:
        k = (d["code"], d["title"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(d)

    return uniq

def main():
    docs = parse_codes_semicolon(CODES_TXT)
    if not docs:
        raise SystemExit("Hiç ICD kaydı parse edilemedi. Dosya yolunu ve formatı kontrol edelim.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # hafif
    texts = [d["text"] for d in docs]

    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine: normalize + inner product
    index.add(emb)

    faiss.write_index(index, OUT_INDEX)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"OK: {len(docs)} ICD kaydı indexlendi → {OUT_INDEX}, {OUT_META}")

if __name__ == "__main__":
    main()