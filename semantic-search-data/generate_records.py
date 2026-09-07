#!/usr/bin/env python3
"""
Synthetic approval-record generator.

Reproduces the measured distributions from SEMANTIC-SEARCH-SYNTHETIC-DATA.md
for a set of 100 records written to record_XXX.txt files.

No real data. Everything here is invented.
"""

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. Entity pools  (Zipfian-friendly: we keep a full list, sample with Zipf)
# ---------------------------------------------------------------------------

PEOPLE_FIRST = [
    "علی", "محمد", "حسین", "رضا", "مهدی", "احمد", "مصطفی", "امیر", "سعید",
    "مجید", "حمید", "حسن", "عباس", "جواد", "فرهاد", "کاوه", "بهرام",
    "ایمان", "ناصر", "بهنام", "مریم", "فاطمه", "زهرا", "نرگس", "سمیرا",
    "شیما", "الهام", "مینا", "راضیه", "ناهید", "پریسا", "لیلا", "آزاده",
    "غزل", "نگار", "مرضیه", "الهه", "سارا", "مونا", "رویا",
]
PEOPLE_LAST = [
    "احمدی", "محمدی", "حسینی", "رضایی", "کریمی", "موسوی", "حیدری", "جعفری",
    "صادقی", "عباسی", "قاسمی", "رحیمی", "کاظمی", "نوری", "اکبری", "تقی‌پور",
    "صالحی", "امیری", "بهرامی", "فتحی", "میرزایی", "نعمتی", "سرلک",
    "کریمیان", "قربانی", "اسدی", "زند", "شریفی", "حمیدی", "دلیری",
]

SUPPLIERS = [
    "شرکت صنایع الکتریک پارس", "بازرگانی کالای صنعتی تهران", "صنایع کابل البرز",
    "گروه تولیدی آریا", "شرکت بسته‌بندی کرج", "بازرگانی ابزار دقیق",
    "صنایع فلزی کویر", "شرکت مواد اولیه زاگرس", "بازرگانی قطعات الکترونیک",
    "گروه صنعتی صبا", "شرکت پلیمر اصفهان", "بازرگانی ماشین‌آلات صنعتی",
    "صنایع رنگ و رزین", "شرکت کاغذ و مقوا اروند", "بازرگانی شیمیایی امید",
    "گروه فولاد سپاهان", "شرکت چاپ و بسته‌بندی مهر", "بازرگانی یراق‌آلات",
    "صنایع پلاستیک تبریز", "شرکت عایق‌بندی پارس جنوبی", "بازرگانی لوازم برقی",
    "گروه کاشی و سرامیک", "شرکت نساجی مازندران", "بازرگانی غذایی غرب",
    "صنایع لوله و اتصالات", "شرکت رنگ پودری آریا", "بازرگانی چسب صنعتی",
    "گروه آلیاژ شرق", "شرکت الکتروموتور بهبود", "بازرگانی ابزار برق",
    "صنایع شیشه لرستان", "شرکت کیسه و پالت دنا", "بازرگانی لاستیک نوین",
    "گروه مهندسی چکاد", "شرکت قالب‌سازی صنعت نو", "بازرگانی دستکش ایمنی",
    "صنایع سیم و کابل قم", "شرکت مواد بسته‌بندی تیس", "بازرگانی رنگ ابریشم",
    "گروه صنعتی دماوند", "شرکت پالت چوبی شمال", "بازرگانی مفتول آذر",
    "صنایع روغن صنعتی بهار", "شرکت بوش و یاتاقان فردوس", "بازرگانی شیشه آریا",
    "گروه تجهیزات برق تابان", "شرکت کارتن و مقوا زنجان", "بازرگانی ابزار دقیق کوثر",
]

REQUESTING_UNITS = [
    "واحد تولید", "واحد بسته‌بندی", "واحد بازرگانی", "واحد فنی و مهندسی",
    "واحد کنترل کیفیت", "واحد انبار", "کارخانه ظرفیت", "واحد تعمیر و نگهداری",
    "واحد فروش", "واحد مالی", "واحد برنامه‌ریزی", "واحد تحقیق و توسعه",
    "واحد تدارکات", "واحد اداری", "واحد ایمنی",
]

PRODUCT_GROUPS = [
    "مواد اولیه", "بسته‌بندی", "قطعات یدکی", "لوازم مصرفی", "ماشین‌آلات", "ابزار و تجهیزات",
]

BANKS = [
    "ملی", "ملت", "صادرات", "تجارت", "سپه", "پاسارگاد", "پارسیان",
    "اقتصاد نوین", "سامان", "مهر", "رفاه", "گرایش",
]

CURRENCIES = ["ریال", "دلار", "یورو", "درهم"]

MONTHS_J = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]

# Items: we must generate a UNIQUE product per line (no reuse).
ITEM_PRECISION = [
    "کابل", "سیم", "پیچ", "مهره", "پروفیل", "ورق", "لوله", "پلاستیک",
    "گرانول", "رنگ", "چسب", "کیسه", "پالت", "کارتن", "روغن", "فیلتر",
    "تسمه", "بلبرینگ", "الکتروموتور", "کلید", "سنسور", "رله", "کنتاکتور",
    "بوش", "یاتاقان", "چرخ‌دنده", "گیربکس", "شیر", "گسکت", "اتصال",
    "دستکش", "ماسک", "نوار", "فیلم", "اکسترود", "قالب", "مته", "فرز",
    "قطعه", "برد", "ترانسفورماتور", "مکنده", "کمپرسور", "پمپ", "متر",
]
ITEM_DESC = [
    "۲.۵", "۴", "۶", "M8", "M10", "PVC", "IP65", "۳×۲.۵", "۲۲", "۱۶",
    "استیل", "روکش‌دار", "آبکاری", "ضدزنگ", "۱۲و۲۴", "۳۲۰", "۴۵۰", "۸۸",
    "HG", "SS304", "مش رنگ ۸۰", "گرید ۲۰", "سایز ۳", "موتور ۵۰", "نیم‌اینچ",
]

UNITS = ["عدد", "Units", "بسته", "kg", "m", "L", "جفت"]
UNIT_WEIGHTS = [0.53, 0.23, 0.07, 0.06, 0.06, 0.03, 0.02]


# ---------------------------------------------------------------------------
# 2. Zipf helper
# ---------------------------------------------------------------------------

def zipf_choice(seq, s=1.0):
    """Pick from seq with a Zipf-like (power) tail. Rank-1 most likely."""
    n = len(seq)
    ranks = list(range(1, n + 1))
    weights = [1.0 / (r ** s) for r in ranks]
    return random.choices(seq, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# 3. Category config: share + enabled field blocks (§2.7) + reason fill (§2.10)
# ---------------------------------------------------------------------------

CATEGORIES = [
    # (name, share, active, fields, reason_fill)
    ("ثبت و اصلاح تردد", 0.762, True, ["date", "person"], 0.34),
    ("درخواست پرداخت (کارخانه)", 0.073, True,
     ["date", "quantity", "reference", "payment_method", "product", "supplier_location",
      "requesting_unit", "product_group", "currency", "request_brief", "items"], 0.57),
    ("General Approval", 0.064, False, ["request_brief"], 0.75),
    ("Payment Application", 0.040, False, ["amount", "reference", "payment_method",
     "supplier_location", "currency", "request_brief"], 0.77),
    ("درخواست جلسه", 0.014, False, ["date"], 0.56),
    ("پرداخت", 0.013, True, ["amount", "reference", "payment_method",
     "supplier_location", "currency", "request_brief"], 0.08),
    ("Business Trip", 0.011, False, ["date", "person"], 0.40),
    ("تدوین و تأیید محتوای وب‌سایت", 0.007, True, ["reference", "request_brief"], 1.0),
    ("درخواست خرید", 0.0014, False, ["supplier_location", "request_brief"], 0.75),
    ("ایجاد حساب‌کاربری در سامانه مای‌دنا", 0.001, True, ["reference", "request_brief"], 1.0),
    ("دبیرخانه", 0.001, False, ["reference", "request_brief"], 1.0),
    # 36 records (1.3%) have NO category at all
    ("", 0.013, True, [], 0.34),
]

STATUS = [
    ("approved", 2267 / 2842), ("new", 179 / 2842), ("pending", 177 / 2842),
    ("refused", 128 / 2842), ("cancel", 91 / 2842),
]


def pick_status():
    return random.choices([s for s, _ in STATUS],
                          weights=[w for _, w in STATUS], k=1)[0]


def pick_category():
    return random.choices([c[0] for c in CATEGORIES],
                          weights=[c[1] for c in CATEGORIES], k=1)[0]


def fields_for(category):
    for c in CATEGORIES:
        if c[0] == category:
            return set(c[3])
    return set()


def reason_fill_for(category):
    for c in CATEGORIES:
        if c[0] == category:
            return c[4]
    return 0.34


# ---------------------------------------------------------------------------
# 4. Title construction (§2.6 templates)
# ---------------------------------------------------------------------------

def jalali_date():
    month = random.choice(MONTHS_J)
    day = random.randint(1, 30)
    year = random.choice(["۱۴۰۴", "۱۴۰۵", "1404", "1405"])
    return f"{year}/{day:02d}/{MONTHS_J.index(month) + 1:02d}> {month}" if False else f"{day} {month} {year}"


# Title templates per category, as functions returning a title.
# Result should be ~6 words / 29 chars median, 65% dash.
def build_title(category, person, supplier, unit, month):
    if category == "ثبت و اصلاح تردد":
        # WWW - WW dominant
        verbs = ["ثبت تردد", "اصلاح تردد", "ثبت ورود و خروج کارکنان", "اصلاح ورود و خروج",
                 "ثبت تردد پرسنل", "اصلاح تردد ورود", "ثبت رفت و آمد", "تصحیح ورود"]
        r = random.random()
        title = random.choice(verbs)
        if r < 0.66:
            title = f"{title} - {person}"
        elif r < 0.80:
            title = f"{title} - {month}"
        # else: bare title
        return title
    if category == "درخواست پرداخت (کارخانه)":
        verbs = [
            "پرداخت فاکتور {invoice} به {supplier}",
            "درخواست پرداخت به {supplier} بابت {group}",
            "پرداخت پیش‌پرداخت {supplier}",
            "تسویه حساب {supplier}",
            "درخواست پرداخت علی‌الحساب {supplier}",
            "پرداخت {group} {supplier}",
        ]
        t = random.choice(verbs)
        t = t.format(invoice=f"{random.randint(100, 9999)}", supplier=rand_supplier_short(supplier),
                     group=random.choice(PRODUCT_GROUPS))
        if random.random() < 0.65:
            t = f"{t} - {month}"
        return t
    if category == "General Approval":
        words = ["تأیید", "موافقت", "مجوز", "تصویب", "استعلام", "تأییدیه"]
        return f"{random.choice(words)} {random.randint(100, 999)}"
    if category == "Payment Application":
        return f"Payment Application {random.randint(1000, 9999)}"
    if category == "درخواست جلسه":
        return f"درخواست جلسه {month} - {random.choice(['هیت مدیره', 'فنی', 'بازرگانی', 'مالی'])}"
    if category == "پرداخت":
        return f"پرداخت {random.randint(100000, 999000000)} ریال"
    if category == "Business Trip":
        return f"Business Trip {person} - {random.choice(['تهران', 'تبریز', 'اصفهان', 'مشهد'])}"
    if category == "تدوین و تأیید محتوای وب‌سایت":
        return f"تدوین و تأیید محتوای وب‌سایت - صفحه {random.choice(['اصلی', 'محصولات', 'درباره ما', 'تماس'])}"
    if category == "درخواست خرید":
        return f"درخواست خرید {random.choice(PRODUCT_GROUPS)}"
    if category == "ایجاد حساب‌کاربری در سامانه مای‌دنا":
        return f"ایجاد حساب‌کاربری در سامانه مای‌دنا برای {person}"
    if category == "دبیرخانه":
        return f"دبیرخانه - مکاتبه {random.randint(100, 999)}"
    return f"درخواست {random.randint(100, 999)}"


def rand_supplier_short(supplier):
    # 10 suppliers have a colloquial short form; approximate with a split
    for token in ["شرکت", "بازرگانی", "صنایع", "گروه"]:
        if supplier.startswith(token):
            return supplier.replace(token, "").strip()
    return supplier


# ---------------------------------------------------------------------------
# 5. Item lines (اقلام) — only factory payment; unique product each
# ---------------------------------------------------------------------------

def gen_items(n_lines):
    """Return n unique lines. Uses global unique-product counter to avoid reuse."""
    items = []
    used = set()
    while len(items) < n_lines:
        base = random.choice(ITEM_PRECISION)
        spec = random.choice(ITEM_DESC)
        product = f"{base} {spec}".strip()
        if product in used:
            continue
        used.add(product)
        qty = gen_quantity()
        items.append({
            "product": product,
            "description": f"{base} {spec}".strip(),
            "unit": random.choices(UNITS, weights=UNIT_WEIGHTS, k=1)[0],
            "quantity": qty,
        })
    return items


def gen_quantity():
    # 0.5 .. 2,000,000, median ~5, log-uniform-ish
    r = random.random()
    if r < 0.5:
        return round(random.uniform(1, 10), 1)
    if r < 0.8:
        return round(random.uniform(10, 100), 1)
    if r < 0.95:
        return random.randint(100, 5000)
    return random.randint(5000, 2000000)


def n_lines_for_request():
    r = random.random()
    if r < 0.6:
        return 1
    if r < 0.85:
        return random.randint(2, 4)
    return random.randint(5, 24)


# ---------------------------------------------------------------------------
# 6. reason prose (§2.10) — length bands + occasional HTML
# ---------------------------------------------------------------------------

def pad_to(persian_text, target_min, target_max):
    return persian_text  # caller controls length via band helpers


def reason_band():
    # 25% 1-20, 34% 21-60, 26% 61-200, 9% 201-600, 3% 601-2000
    r = random.random()
    if r < 0.25:
        return (1, 20)
    if r < 0.59:
        return (21, 60)
    if r < 0.85:
        return (61, 200)
    if r < 0.94:
        return (201, 600)
    return (601, 2000)


SHORT_NOTES = [
    "تسویه فروردین", "پرداخت کامل", "تأیید شد", "اصلی", "برای انبار",
    "عجله دارد", "فوری", "پیش‌پرداخت", "تسویه حساب", "مرحله دوم",
]

# Category-appropriate phrasing (keeps record semantics coherent).
ATT_POOL = [
    "ثبت تردد", "اصلاح ساعت ورود", "اصلاح ساعت خروج", "ثبت مرخصی", "اصلاح کارت زدن",
]
MEET_POOL = [
    "هماهنگی جلسه", "ثبت صورت جلسه", "تایید زمان جلسه", "دعوت از واحد فنی", "بررسی مصوبات",
]
WEB_POOL = ["اصلاح متن", "تنظیم صفحه", "افزودن مطلب", "ویرایش محتوا", "تایید نسخه نهایی"]
BRIEF_NOTE_POOL = [
    "تایید شد", "اصل انجام شد", "عجله دارد", "فوری", "برای واحد مالی", "ضمایم پیوست است",
]
SENT_ATT = [
    "تردد روز {month} اصلاح گردید.",
    "ساعت ورود کارمند تصحیح شده است.",
    "به دلیل مراجعه به پزشک، تردد اصلاح شد.",
    "خروج روز {month} ثبت گردید.",
]
SENT_MEET = [
    "جلسه {month} با واحد {unit} برگزار شد.",
    "زمان جلسه به {month} موکول گردید.",
    "جلسه هماهنگی {month} درخواست شد.",
]
SENT_PAY = [
    "پرداخت بابت {supplier} انجام شد.",
    "درخواست {month} به تایید رسیده است.",
    "مبلغ مربوط به {group} واریز گردید.",
    "فاکتور {inv} در انتظار تایید است.",
    "محصولات {group} تحویل انبار شد.",
]
PARA = [
    "با توجه به نیاز واحد {unit} و تایید کارشناس فنی، پرداخت به {supplier} بابت {group} درخواست می‌شود. مبلغ فاکتور {inv} ریال بوده و پس از بررسی تطبیق سفارش و تحویل، اقدام به واریز شد.",
    "این درخواست مربوط به خرید {group} از {supplier} است. پس از دریافت کالا توسط واحد {unit} و کنترل کیفیت، فاکتور {inv} جهت پرداخت ارسال گردید. لطفا نسبت به تسویه اقدام فرمایید.",
]
LONG = [
    "با احترام، به استحضار می‌رساند پیرو درخواست واحد {unit} مبنی بر تامین {group}، پس از استعلام قیمت از سه تامین‌کننده معتبر و بررسی تطبیق نمونه ارسالی با مشخصات فنی مصوب، در نهایت {supplier} با قیمت {inv} ریال انتخاب گردید. کالا مطابق برنامه زمان‌بندی تحویل و توسط واحد کنترل کیفیت تایید شد. لذا استدعا دارد نسبت به صدور حواله و واریز مبلغ مربوطه اقدام لازم به عمل آید. با تشکر از همکاری.",
    "مستند به مصوبه جلسه کمیته تدارکات و با توجه به محدودیت موجودی انبار، نیاز به تامین فوری {group} جهت ادامه تولید می‌باشد. با عنایت به ماده ۱۴ آیین‌نامه معاملات داخلی و تایید مدیرعامل، پرداخت پیش‌پرداخت ۳۰ درصد به {supplier} درخواست می‌گردد. مابقی پس از تحویل و کنترل کیفیت پرداخت خواهد شد. فاکتور مربوطه تحت شماره {inv} ثبت گردیده است.",
]


def gen_reason(category, supplier, unit, month):
    lo, hi = reason_band()
    base_tokens = {
        "supplier": supplier,
        "unit": random.choice(REQUESTING_UNITS),
        "group": random.choice(PRODUCT_GROUPS),
        "month": month,
        "inv": str(random.randint(100, 9999)),
    }
    # trim supplier to a usable short line
    sup = rand_supplier_short(supplier)
    base_tokens["supplier"] = sup

    # pick the pool matching the category's semantics
    if category in {"ثبت و اصلاح تردد", "Business Trip"}:
        short_pool, sent_pool = ATT_POOL + BRIEF_NOTE_POOL, SENT_ATT
    elif category == "درخواست جلسه":
        short_pool, sent_pool = MEET_POOL + BRIEF_NOTE_POOL, SENT_MEET
    elif category == "تدوین و تأیید محتوای وب‌سایت":
        short_pool, sent_pool = WEB_POOL + BRIEF_NOTE_POOL, []
    else:
        short_pool, sent_pool = SHORT_NOTES, SENT_PAY

    if hi <= 20:
        body = random.choice(short_pool)
    elif hi <= 60:
        body = (random.choice(sent_pool) if sent_pool
                else random.choice(BRIEF_NOTE_POOL)).format(**base_tokens)
    elif hi <= 200:
        body = random.choice(PARA).format(**base_tokens)
    else:
        body = random.choice(LONG).format(**base_tokens)

    # ~5% HTML (table/list)
    if random.random() < 0.05 and hi >= 40:
        n = random.randint(2, 5)
        rows = "".join(f"<li>{random.choice(SHORT_NOTES)}</li>" for _ in range(n))
        body = f"<ul>{rows}</ul>"
    else:
        body = f"<p>{body}</p>"
    return body


def gen_request_brief(category):
    briefs = [
        "خرید اقلام", "تسویه حساب", "پیش‌پرداخت", "تامین مواد",
        "بسته‌بندی", "قطعات یدکی", "تعمیرات", "هزینه سفر",
    ]
    return random.choice(briefs)


# ---------------------------------------------------------------------------
# 7. Roughening (~25% of records) — §4 Step 4 / Prompt C
# ---------------------------------------------------------------------------

ZWNJ = "\u200c"
ARABIC_Y = "ي"
PERSIAN_Y = "ی"
ARABIC_K = "ك"
PERSIAN_K = "ک"


def roughen(text):
    r = random.random()
    if r < 0.30:
        text = text.replace(PERSIAN_Y, ARABIC_Y).replace(PERSIAN_K, ARABIC_K)
    elif r < 0.55:
        text = text.replace(ZWNJ, "")
    elif r < 0.70:
        # typo: drop a random letter
        idx = random.randint(0, len(text) - 1)
        text = text[:idx] + text[idx + 1:]
    elif r < 0.80:
        text = text.replace("  ", " ")
    elif r < 0.90:
        text = text.replace("۱۲۳", "123").replace("۰۹", "09")
    return text


# ---------------------------------------------------------------------------
# 8. Build the records
# ---------------------------------------------------------------------------

def build_record(i):
    category = pick_category()
    fields = fields_for(category)
    has_cat = bool(category)

    person = f"{random.choice(PEOPLE_FIRST)} {random.choice(PEOPLE_LAST)}"
    supplier = zipf_choice(SUPPLIERS, s=1.3) if random.random() < 0.9 else random.choice(SUPPLIERS)
    unit = zipf_choice(REQUESTING_UNITS)
    month = random.choice(MONTHS_J)

    title = build_title(category, person, supplier, unit, month)

    status = pick_status()
    date = f"{random.randint(1, 30)} {month} ۱۴۰۵"

    rec = {
        "category": category,
        "title": title,
        "status": status,
        "person": person,
        "supplier": supplier,
        "requesting_unit": unit,
        "date": date,
    }

    # structured fields per category
    if "reference" in fields:
        rec["reference"] = f"{random.randint(1000, 99999)}"
    if "amount" in fields or "payment_method" in fields:
        rec["amount"] = f"{random.randint(100000, 500000000):,} ریال"
    if "payment_method" in fields:
        rec["payment_method"] = random.choice(["حواله", "چک", "کارت به کارت", "پایا", "ساتنا", "نقدی"])
    if "currency" in fields:
        rec["currency"] = random.choice(CURRENCIES)
    if "supplier_location" in fields:
        rec["bank"] = zipf_choice(BANKS)
    if "product_group" in fields:
        rec["product_group"] = random.choice(PRODUCT_GROUPS)
    if "items" in fields:
        rec["lines"] = gen_items(n_lines_for_request())
    if "request_brief" in fields:
        rec["request_brief"] = gen_request_brief(category)

    # reason prose at per-category fill rate
    if random.random() < reason_fill_for(category):
        rec["reason"] = gen_reason(category, supplier, unit, month)

    return rec


# ---------------------------------------------------------------------------
# 9. Serialize to text file
# ---------------------------------------------------------------------------

def serialize(rec, rid):
    lines = []
    lines.append(f"شناسه: {rid}")
    lines.append(f"دسته‌بندی: {rec.get('category') or '(بدون دسته)'}")
    lines.append(f"عنوان: {rec.get('title', '')}")
    lines.append(f"وضعیت: {rec.get('status', '')}")
    lines.append(f"تاریخ: {rec.get('date', '')}")
    if rec.get("supplier"):
        lines.append(f"تأمین‌کننده: {rec['supplier']}")
    if rec.get("person"):
        lines.append(f"شخص: {rec['person']}")
    if rec.get("requesting_unit"):
        lines.append(f"واحد درخواست‌دهنده: {rec['requesting_unit']}")
    if rec.get("bank"):
        lines.append(f"بانک: {rec['bank']}")
    if rec.get("amount"):
        lines.append(f"مبلغ: {rec['amount']}")
    if rec.get("payment_method"):
        lines.append(f"روش پرداخت: {rec['payment_method']}")
    if rec.get("currency"):
        lines.append(f"ارز: {rec['currency']}")
    if rec.get("product_group"):
        lines.append(f"گروه کالا: {rec['product_group']}")
    if rec.get("reference"):
        lines.append(f"مرجع: {rec['reference']}")
    if rec.get("request_brief"):
        lines.append(f"خلاصه درخواست: {rec['request_brief']}")

    items = rec.get("lines") or []
    if items:
        lines.append("")
        lines.append("اقلام:")
        for ln in items:
            lines.append(
                f"  - {ln['product']} | {ln['description']} | "
                f"{ln['quantity']} {ln['unit']}"
            )

    if rec.get("reason"):
        prose = rec["reason"]
        plain = re.sub(r"<[^>]+>", " ", prose)
        plain = re.sub(r"\s+", " ", plain).strip()
        lines.append("")
        lines.append("توضیحات:")
        lines.append(f"  {plain}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. Build 100 records and write them out
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, 101):
        rec = build_record(i)
        # roughen ~25% of records
        if random.random() < 0.25:
            rec["title"] = roughen(rec["title"])
            if rec.get("reason"):
                rec["reason"] = roughen(rec["reason"])
        name = f"record_{i:03d}.txt"
        (OUT_DIR / name).write_text(
            serialize(rec, f"REC-{i:03d}") + "\n", encoding="utf-8"
        )
    print("done")


if __name__ == "__main__":
    main()
