"""
Synthetic document generator for DocVL-IN.

Generates rasterized invoices / forms / ID cards with randomized layouts and
English + Hindi (Devanagari) text, paired with ground-truth JSON labels.

Why synthetic data at all: public Indic document datasets are scarce and rarely
come with rich structured labels. Synthetic generation gives us (a) volume,
(b) exact ground truth for free, and (c) control over difficulty (noise, blur,
rotation) to stress-test the model. It's meant to be blended with real data in
prepare_dataset.py, not used alone.

Usage:
    python generate_synthetic.py --n 500 --out data/synthetic --font-dir fonts/
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker

from schema import InvoiceFields, FormFields, IDCardFields

fake_en = Faker("en_IN")
fake_hi = Faker("hi_IN")

# Devanagari names for the bilingual fields (faker's hi_IN name provider works well here)
HINDI_TITLES = {
    "invoice": "\u091a\u093e\u0932\u093e\u0928",  # "चालान"
    "form": "\u092b\u093c\u0949\u0930\u094d\u092e",  # "फ़ॉर्म"
    "id_card": "\u092a\u0939\u091a\u093e\u0928 \u092a\u0924\u094d\u0930",  # "पहचान पत्र"
}


def _font(font_dir: Path, size: int, devanagari: bool = False) -> ImageFont.FreeTypeFont:
    """
    Loads a font. Falls back to PIL's default bitmap font if the requested TTF
    isn't found, but note the default font CANNOT render Devanagari glyphs —
    for real Hindi rendering you must supply a Devanagari-capable font
    (e.g. Noto Sans Devanagari, downloadable from Google Fonts) via --font-dir.
    """
    name = "NotoSansDevanagari-Regular.ttf" if devanagari else "DejaVuSans.ttf"
    path = font_dir / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _degrade(img: Image.Image) -> Image.Image:
    """Adds scan-like noise: slight rotation, blur, brightness jitter — mimics real photos of documents."""
    angle = random.uniform(-2.5, 2.5)
    img = img.rotate(angle, expand=True, fillcolor="white")
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    return img


def make_invoice(font_dir: Path) -> tuple[Image.Image, InvoiceFields]:
    w, h = 900, 1200
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(font_dir, 34)
    f_body = _font(font_dir, 22)
    f_hi = _font(font_dir, 24, devanagari=True)

    fields = InvoiceFields(
        invoice_number=f"INV-{random.randint(10000, 99999)}",
        invoice_date=fake_en.date_between(start_date="-2y", end_date="today").isoformat(),
        vendor_name=fake_en.company(),
        buyer_name=fake_en.name(),
        total_amount=str(random.randint(500, 500000)),
        currency="INR",
        gstin=f"{random.randint(10,37)}{fake_en.bothify('?????#####?#Z#').upper()}",
    )

    y = 40
    d.text((40, y), HINDI_TITLES["invoice"], font=f_hi, fill="black"); y += 50
    d.text((40, y), f"Invoice No: {fields.invoice_number}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Date: {fields.invoice_date}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Vendor: {fields.vendor_name}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Bill To: {fields.buyer_name}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"GSTIN: {fields.gstin}", font=f_body, fill="black"); y += 60
    d.line((40, y, w - 40, y), fill="black", width=2); y += 30
    d.text((40, y), f"Total Amount: Rs. {fields.total_amount}", font=f_title, fill="black")

    return _degrade(img), fields


def make_form(font_dir: Path) -> tuple[Image.Image, FormFields]:
    w, h = 900, 1200
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f_title = _font(font_dir, 30)
    f_body = _font(font_dir, 22)
    f_hi = _font(font_dir, 24, devanagari=True)

    name_en = fake_en.name()
    fields = FormFields(
        form_title=random.choice(["Application for Address Proof", "Scholarship Application", "Ration Card Renewal"]),
        applicant_name=name_en,
        applicant_name_hindi=fake_hi.name(),
        date_of_birth=fake_en.date_of_birth(minimum_age=18, maximum_age=60).isoformat(),
        address=fake_en.address().replace("\n", ", "),
        phone_number=fake_en.msisdn()[:10],
        form_id=f"FRM-{random.randint(100000, 999999)}",
    )

    y = 40
    d.text((40, y), fields.form_title, font=f_title, fill="black"); y += 50
    d.text((40, y), HINDI_TITLES["form"], font=f_hi, fill="black"); y += 45
    d.text((40, y), f"Form ID: {fields.form_id}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Name: {fields.applicant_name}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"\u0928\u093e\u092e: {fields.applicant_name_hindi}", font=f_hi, fill="black"); y += 40
    d.text((40, y), f"DOB: {fields.date_of_birth}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Phone: {fields.phone_number}", font=f_body, fill="black"); y += 40
    d.text((40, y), f"Address: {fields.address}", font=f_body, fill="black")

    return _degrade(img), fields


def make_id_card(font_dir: Path) -> tuple[Image.Image, IDCardFields]:
    w, h = 900, 570  # ID-card-ish aspect ratio
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w - 1, h - 1), outline="black", width=4)
    f_title = _font(font_dir, 26)
    f_body = _font(font_dir, 22)
    f_hi = _font(font_dir, 24, devanagari=True)

    fields = IDCardFields(
        id_number=fake_en.bothify("####-####-####"),
        holder_name=fake_en.name(),
        holder_name_hindi=fake_hi.name(),
        date_of_birth=fake_en.date_of_birth(minimum_age=18, maximum_age=70).isoformat(),
        issuing_authority=random.choice(["UIDAI", "State Transport Authority", "Election Commission"]),
        valid_until=fake_en.date_between(start_date="today", end_date="+10y").isoformat(),
    )

    y = 30
    d.text((30, y), HINDI_TITLES["id_card"], font=f_hi, fill="black"); y += 45
    d.text((30, y), f"ID No: {fields.id_number}", font=f_body, fill="black"); y += 38
    d.text((30, y), f"Name: {fields.holder_name}", font=f_body, fill="black"); y += 38
    d.text((30, y), f"\u0928\u093e\u092e: {fields.holder_name_hindi}", font=f_hi, fill="black"); y += 38
    d.text((30, y), f"DOB: {fields.date_of_birth}", font=f_body, fill="black"); y += 38
    d.text((30, y), f"Issued by: {fields.issuing_authority}", font=f_body, fill="black"); y += 38
    d.text((30, y), f"Valid Until: {fields.valid_until}", font=f_title, fill="black")

    return _degrade(img), fields


GENERATORS = {"invoice": make_invoice, "form": make_form, "id_card": make_id_card}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="total number of synthetic docs to generate")
    ap.add_argument("--out", type=str, default="data/synthetic")
    ap.add_argument("--font-dir", type=str, default="fonts/",
                     help="dir containing DejaVuSans.ttf and NotoSansDevanagari-Regular.ttf")
    args = ap.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    font_dir = Path(args.font_dir)

    manifest = []
    doc_types = list(GENERATORS.keys())
    for i in range(args.n):
        doc_type = doc_types[i % len(doc_types)]
        img, fields = GENERATORS[doc_type](font_dir)
        fname = f"{doc_type}_{i:05d}.png"
        img.save(out_dir / "images" / fname)
        manifest.append({
            "image": f"images/{fname}",
            "doc_type": doc_type,
            "label": fields.model_dump(),
        })

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(manifest)} synthetic documents -> {out_dir}")
    print("NOTE: if Hindi text rendered as boxes/tofu, download a Devanagari font "
          "(e.g. Noto Sans Devanagari) into --font-dir.")


if __name__ == "__main__":
    main()
