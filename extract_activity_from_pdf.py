import re
import csv
import sys
import subprocess
from pathlib import Path


def get_pdf_text(pdffile: Path) -> str:
    # Prefer pdftotext for layout preservation
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdffile), "-"],
            check=True,
            capture_output=True,
        )
        return result.stdout.decode("utf-8", errors="ignore")
    except Exception:
        pass
    # Fallback: try PyPDF2 if available
    try:
        import PyPDF2  # type: ignore

        text_parts = []
        with open(pdffile, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                txt = page.extract_text() or ""
                text_parts.append(txt + "\f")
        return "".join(text_parts)
    except Exception as e:
        raise RuntimeError(
            "无法从PDF提取文本：需要系统提供pdftotext或安装PyPDF2。"
        ) from e


SMILES_ALLOWED = set(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@+-=#()/\\[]%.:")
)


def looks_like_smiles(token: str) -> bool:
    # Basic heuristics to reduce false positives
    s = token.strip()
    if not s:
        return False
    # Filter very short or very long tokens
    if len(s) < 3 or len(s) > 300:
        return False
    # Must be composed of allowed chars
    if any(ch not in SMILES_ALLOWED for ch in s):
        return False
    # Avoid tokens that are clearly numbers or units
    if re.fullmatch(r"[\d.×xEe+-]+", s):
        return False
    # Require presence of at least one typical SMILES feature
    if not re.search(r"[=\[\]#@/\\cnopsCNOPSFIBrCl]\b?", s):
        return False
    # Avoid words likely to be names (containing spaces should be already stripped)
    # Avoid obvious headers
    if s.lower() in {"smiles", "structure"}:
        return False
    return True


ACTIVITY_TYPES = [
    "IC50",
    "EC50",
    "Ki",
    "Kd",
    "GI50",
    "ED50",
    "AC50",
    "MIC",
    "MBC",
    "TC50",
    "CC50",
    "pIC50",
    "pEC50",
    "pKi",
    "pKd",
]

ACT_TYPES_RE = r"(?:" + r"|".join(re.escape(t) for t in ACTIVITY_TYPES) + r")"
UNITS_RE = r"(?:pM|nM|µM|μM|uM|mM|M|fM|mg/mL|ug/mL|µg/mL|μg/mL|ng/mL|g/L|%|ppm)"
NUM_RE = r"(?:[<>≤≥~≈]?\s*(?:\d+(?:[.,]\d+)?(?:\s*[×x]\s*10\s*\^\s*-?\d+)?(?:e-?\d+)?))"

# e.g. IC50 = 12.3 nM; pIC50 7.3; Kd<10 nM; EC50 (µM) 0.45
ACTIVITY_PATTERN = re.compile(
    rf"\b({ACT_TYPES_RE})\b\s*[:=]?\s*({NUM_RE})(?:\s*({UNITS_RE}))?",
    re.IGNORECASE,
)


def extract_records(text: str):
    records = []
    # Split by pages using form feed if present
    pages = text.split("\f") if "\f" in text else [text]
    for pageno, page in enumerate(pages, start=1):
        lines = page.splitlines()
        # Collect candidate SMILES with their line index
        smile_hits = []
        for idx, line in enumerate(lines):
            # Split by common separators to catch inline tokens
            tokens = re.split(r"\s{2,}|\t|\s/\s|\s,\s|\s;\s|\s\|\s|\s", line)
            for tok in tokens:
                tok = tok.strip()
                if looks_like_smiles(tok):
                    smile_hits.append((idx, tok))

        # For each SMILES, search within a window for activity mentions
        for idx, smiles in smile_hits:
            win_from = max(0, idx - 3)
            win_to = min(len(lines), idx + 4)
            context = " ".join(lines[win_from:win_to])
            for m in ACTIVITY_PATTERN.finditer(context):
                act_type = m.group(1)
                value = m.group(2).replace(",", ".").replace(" ", "")
                unit = m.group(3) or ""
                records.append(
                    {
                        "SMILES": smiles,
                        "ActivityType": act_type.upper(),
                        "Value": value,
                        "Unit": unit,
                        "Page": pageno,
                        "Context": context[:200],
                    }
                )

    # De-duplicate identical tuples
    dedup = []
    seen = set()
    for r in records:
        key = (r["SMILES"], r["ActivityType"], r["Value"], r["Unit"], r["Page"])  # context ignored
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_activity_from_pdf.py test.pdf [输出CSV]", file=sys.stderr)
        sys.exit(2)
    pdf_path = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("extracted_molecules.csv")
    if not pdf_path.exists():
        print(f"未找到PDF: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    text = get_pdf_text(pdf_path)
    records = extract_records(text)
    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["SMILES", "ActivityType", "Value", "Unit", "Page", "Context"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"写入: {out_csv}，共 {len(records)} 条记录")


if __name__ == "__main__":
    main()

