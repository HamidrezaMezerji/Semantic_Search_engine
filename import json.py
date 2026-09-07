import json
import random
import re
import os
from collections import Counter

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
NUM_RECORDS = 50
OUTPUT_DIR = "synthetic_texts"
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# Entity pools (invented, no real data)
# ------------------------------------------------------------
PEOPLE = ["علی رضایی", "مریم کریمی", "رضا احمدی", "سارا محمدی", "مهدی حسینی",
          "زهرا موسوی", "امیر علی‌پور", "نگار صادقی", "حسین مرادی", "فاطمه باقری"]
SUPPLIERS = ["تأمین‌کننده الکتریک پارس", "صنایع کابل مهر", "بازرگانی سامان",
             "گروه تولیدی آریا", "شرکت فنی و مهندسی نوین", "صنایع بسته‌بندی گلستان"]
UNITS = ["واحد مالی", "واحد تولید", "واحد بازرگانی", "واحد فنی", "واحد فروش"]
PRODUCT_GROUPS = ["الکتریکی", "بسته‌بندی", "مکانیکی", "شیمیایی", "کامپیوتری", "خدماتی"]
BANKS = ["ملت", "صادرات", "تجارت", "ملی", "سپه"]
CURRENCIES = ["ریال", "دلار", "یورو", "درهم"]
UNITS_ITEM = ["عدد", "Units", "بسته", "kg", "m", "L", "جفت"]
UNIT_WEIGHTS = [0.53, 0.23, 0.07, 0.06, 0.06, 0.03, 0.02]

# Category shares (from §2.1)
CATEGORIES = [
    ("ثبت و اصلاح تردد", 0.762),
    ("درخواست پرداخت (کارخانه)", 0.073),
    ("General Approval", 0.064),
    ("Payment Application", 0.040),
    ("درخواست جلسه", 0.014),
    ("پرداخت", 0.013),
    ("Business Trip", 0.011),
    ("تدوین و تأیید محتوای وب‌سایت", 0.007),
    ("درخواست خرید", 0.0014),
    ("ایجاد حساب‌کاربری در سامانه مای‌دنا", 0.001),
]
CAT_NAMES, CAT_PROBS = zip(*CATEGORIES)

# Status mix
STATUSES = ["approved"] * 2267 + ["new"] * 179 + ["pending"] * 177 + ["refused"] * 128 + ["cancel"] * 91
STATUS_LIST = list(set(STATUSES))
STATUS_PROBS = [STATUSES.count(s) / len(STATUSES) for s in STATUS_LIST]

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def zipf_sample(pool, alpha=1.5):
    weights = [1 / (i+1)**alpha for i in range(len(pool))]
    return random.choices(pool, weights=weights, k=1)[0]

def generate_title_and_meta(category):
    """Returns title, person, supplier, requesting_unit."""
    person = zipf_sample(PEOPLE)
    supplier = zipf_sample(SUPPLIERS)
    unit = zipf_sample(UNITS)
    month = random.choice(["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی"])

    if category == "ثبت و اصلاح تردد":
        templates = [
            f"ثبت تردد - {person} {month}",
            f"اصلاح تردد - {person}",
            f"تردد {person}",
            f"درخواست ثبت تردد {person} - {month}",
            f"{person} - تردد {month}",
        ]
    elif category == "درخواست پرداخت (کارخانه)":
        pg = zipf_sample(PRODUCT_GROUPS)
        amount = random.choice(["۵۰", "۱۰۰", "۲۰۰", "۵۰۰", "۱۰۰۰", "۲۰۰۰"])
        curr = random.choice(CURRENCIES)
        templates = [
            f"پرداخت فاکتور - {supplier}",
            f"درخواست پرداخت به {supplier} بابت {pg}",
            f"پیش‌پرداخت {supplier} - {amount} {curr}",
            f"تسویه حساب {supplier} - {pg}",
            f"حواله {supplier}",
        ]
    elif category == "General Approval":
        subj = random.choice(["گزارش", "پیشنهاد", "درخواست", "تأییدیه", "مجوز"])
        templates = [f"{subj} - {person}", f"{subj} عمومی"]
    elif category == "Payment Application":
        templates = [f"پرداخت - {supplier}", f"درخواست پرداخت {supplier}"]
    elif category == "پرداخت":
        templates = [f"پرداخت - {supplier}", f"پرداخت فوری {supplier}"]
    elif category == "درخواست جلسه":
        templates = [f"جلسه - {person}", f"درخواست جلسه با {person}"]
    elif category == "Business Trip":
        templates = [f"سفر کاری - {person}", f"درخواست سفر {person}"]
    elif category == "تدوین و تأیید محتوای وب‌سایت":
        templates = [f"محتوای وب‌سایت - {random.choice(['صفحه اصلی', 'درباره ما', 'تماس', 'محصولات'])}"]
    elif category == "درخواست خرید":
        templates = [f"خرید - {zipf_sample(PRODUCT_GROUPS)}"]
    elif category == "ایجاد حساب‌کاربری در سامانه مای‌دنا":
        templates = [f"ایجاد حساب - {person}"]
    else:
        templates = ["درخواست عمومی"]

    title = random.choice(templates)
    # dash suffix in ~65% of cases
    if random.random() < 0.65:
        suffix = random.choice([f" - {person}", f" - {zipf_sample(PEOPLE)}", f" - {month}"])
        title += suffix

    return title, person, supplier, unit

def generate_reason(category):
    fill_rates = {
        "تدوین و تأیید محتوای وب‌سایت": 1.0,
        "ایجاد حساب‌کاربری در سامانه مای‌دنا": 1.0,
        "Payment Application": 0.77,
        "General Approval": 0.75,
        "درخواست خرید": 0.75,
        "درخواست پرداخت (کارخانه)": 0.57,
        "درخواست جلسه": 0.56,
        "Business Trip": 0.40,
        "ثبت و اصلاح تردد": 0.34,
        "پرداخت": 0.08,
    }
    if random.random() > fill_rates.get(category, 0.4):
        return ""

    bands = [(1, 20, 0.25), (21, 60, 0.34), (61, 200, 0.26), (201, 600, 0.09), (601, 2000, 0.03)]
    lo, hi, _ = random.choices(bands, weights=[b[2] for b in bands])[0]
    target_len = random.randint(lo, hi)

    words = ["تسویه", "پرداخت", "درخواست", "تأیید", "بررسی", "فاکتور", "حواله", "ضمانت‌نامه", "کوتاژ", "ثبت سفارش"]
    text = ""
    while len(text) < target_len:
        text += random.choice(words) + " "
        if random.random() < 0.1:
            text += f"{random.randint(1, 99)} "
        if random.random() < 0.05:
            text += f"{random.choice(['PVC', 'IP65', 'M8'])} "
    text = text[:target_len].strip()

    # ~5% HTML
    if random.random() < 0.05:
        if random.random() < 0.5:
            items = "".join(f"<li>{random.choice(words)}</li>" for _ in range(random.randint(2,5)))
            text = f"<ul>{items}</ul>"
        else:
            rows = "".join(f"<tr><td>{random.choice(words)}</td><td>{random.randint(1,10)}</td></tr>" for _ in range(2))
            text = f"<table>{rows}</table>"
    else:
        if random.random() < 0.7:
            text = f"<p>{text}</p>"
    return text

def generate_request_brief():
    if random.random() > 0.11:
        return ""
    brief_words = ["مختصر", "توضیح", "یادداشت", "خلاصه"]
    return f"{random.choice(brief_words)} {random.randint(1, 100)}"

def generate_item_lines(category):
    if category != "درخواست پرداخت (کارخانه)":
        return []
    # 1‑4 lines per request
    num = 1 if random.random() < 0.5 else random.randint(1, 4)
    lines = []
    for _ in range(num):
        product = f"محصول {random.randint(1000, 9999)}"
        desc_words = ["کابل", "پیچ", "مهره", "ورق", "رنگ", "مواد", "جعبه", "کارتن", "پلاستیک", "الکترود"]
        desc = f"{random.choice(desc_words)} {random.choice(['۲.۵', '۴', '۶', '۱۰', '۱۶', 'M8', 'M10', 'PVC', 'IP65'])}"
        unit = random.choices(UNITS_ITEM, weights=UNIT_WEIGHTS)[0]
        qty = random.choice([0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 2000000])
        lines.append(f"{product} | {desc} | {unit} | {qty}")
    return lines

def roughen(text):
    if not text or random.random() > 0.25:
        return text
    if random.random() < 0.3:
        text = text.replace("ی", "ي").replace("ک", "ك")
    if random.random() < 0.25:
        text = text.replace(" ", "  ")
    if random.random() < 0.15 and len(text) > 2:
        i = random.randint(0, len(text)-2)
        text = text[:i] + text[i+1] + text[i] + text[i+2:]
    return text

# ------------------------------------------------------------
# Generate one record as plain Persian text
# ------------------------------------------------------------
def generate_plain_text_record(record_id):
    category = random.choices(CAT_NAMES, weights=CAT_PROBS)[0]
    status = random.choices(STATUS_LIST, weights=STATUS_PROBS)[0]
    title, person, supplier, unit = generate_title_and_meta(category)
    reason = generate_reason(category)
    brief = generate_request_brief()
    date = f"۱۴۰۵/{random.randint(1,12):02d}/{random.randint(1,28):02d}"
    lines = generate_item_lines(category)

    # apply roughening
    title = roughen(title)
    reason = roughen(reason)
    brief = roughen(brief)

    # status in Persian
    status_map = {
        "approved": "تایید شده",
        "new": "جدید",
        "pending": "در انتظار",
        "refused": "رد شده",
        "cancel": "لغو شده"
    }
    status_fa = status_map.get(status, status)

    # Build plain text block
    text = f"""عنوان درخواست: {title}
دسته‌بندی: {category}
وضعیت: {status_fa}
تاریخ: {date}
شخص مرتبط: {person}
تأمین‌کننده: {supplier}
واحد درخواست‌کننده: {unit}
توضیحات: {reason if reason else '(ندارد)'}
خلاصه درخواست: {brief if brief else '(ندارد)'}"""

    if lines:
        text += "\nاقلام:\n" + "\n".join(f"  - {ln}" for ln in lines)
    else:
        text += "\nاقلام: (ندارد)"

    text += f"\n---\nشناسه: synth_{record_id:03d}"
    return text

# ------------------------------------------------------------
# Generate all 50 files
# ------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, NUM_RECORDS + 1):
    content = generate_plain_text_record(i)
    fname = os.path.join(OUTPUT_DIR, f"record_{i:03d}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(content)

print(f"✅ {NUM_RECORDS} plain Persian text files written to folder '{OUTPUT_DIR}/'")

# ------------------------------------------------------------
# Quick validation summary (from §7)
# ------------------------------------------------------------
def quick_validate():
    titles = []
    reasons = []
    lines_count = 0
    for i in range(1, NUM_RECORDS + 1):
        with open(os.path.join(OUTPUT_DIR, f"record_{i:03d}.txt"), encoding="utf-8") as f:
            txt = f.read()
            # extract title line
            for line in txt.splitlines():
                if line.startswith("عنوان درخواست:"):
                    titles.append(line.replace("عنوان درخواست:", "").strip())
                    break
            # extract reason
            for line in txt.splitlines():
                if line.startswith("توضیحات:"):
                    val = line.replace("توضیحات:", "").strip()
                    if val != "(ندارد)":
                        reasons.append(val)
                    break
            # count lines
            if "اقلام:" in txt:
                items_part = txt.split("اقلام:")[1]
                for line in items_part.splitlines():
                    if line.strip().startswith("-"):
                        lines_count += 1

    if not titles:
        return
    lens = [len(t) for t in titles]
    words = [len(t.split()) for t in titles]
    dash = sum('-' in t for t in titles) / len(titles) * 100
    print("\n=== Quick validation of generated plain text files ===")
    print(f"Records: {len(titles)}")
    print(f"Title median chars: {sorted(lens)[len(lens)//2]} (target 29)")
    print(f"Title median words: {sorted(words)[len(words)//2]} (target 6)")
    print(f"Dash %: {dash:.1f} (target ~65%)")
    print(f"Reason filled %: {len(reasons)/len(titles)*100:.1f} (target ~40%)")
    print(f"Reason median chars: {sorted([len(r) for r in reasons])[len(reasons)//2] if reasons else 0} (target 42)")
    print(f"Item lines count: {lines_count} (should be ~7% of records have 1-4 lines)")

quick_validate()