"""
Evaluation helpers and JSON export for the LibreChat RAG project.

This module defines the 16 evaluation questions and utilities to collect
short answers suitable for automatic grading. Answers are exported to a
JSON file; no zip archive is created.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from langchain_core.runnables import Runnable

# The 16 evaluation questions used to assess the system, kept in order.
EVALUATION_QUESTIONS = [
    "پرسش ۱: توروالدز برای کار در چه موسسه‌ای دانشگاه هلسینکی را ترک گفت؟",
    "پرسش ۲: آندرو تاننباوم استاد کدام دانشگاه است؟",
    "پرسش ۳: در سال ۲۰۰۶ چند درصد از هسته لینوکس توسط توروالدز نوشته شد (به عدد)؟",
    "پرسش ۴: چه کسی بنیاد نرم‌افزارهای آزاد را بنا نهاد؟",
    "پرسش ۵: ریچارد استالمن در ۲۱ سالگی در کدام شرکت کار می‌کرد؟",
    "پرسش ۶: یکی از مشهورترین پروژه‌هایی که در ابتدا پروژه‌ی آزاد و آکادمیک بود اما بعد وارد محیط بسته‌ی تجاری شد چه بود؟",
    "پرسش ۷: لینکدین در سانسور کردن حساب‌ها به درخواست چه کشوری مشهور است؟",
    "پرسش ۸: ریچارد استالمن پیشنهاد می‌کند به‌جای گوگل مپ از چه سرویسی استفاده کنیم؟",
    "پرسش ۹: آزادی صفرم در نرم‌افزار آزاد چه عنوانی دارد؟",
    "پرسش ۱۰: آیا یک نرم‌افزار آزاد لزوماً رایگان است (بله یا خیر)؟",
    "پرسش ۱۱: استاندارد ناظر بر فایل‌ها و دایرکتوری‌ها به‌اختصار چه نامیده می‌شود؟",
    "پرسش ۱۲: اولین ریپلای به ایمیل درخواست کار چیست؟",
    "پرسش ۱۳: اگر امروز که از شنبه ورزش می‌کنم در واقع دچار چه بایاسی شده‌ایم؟",
    "پرسش ۱۴: دنبال یاد گرفتن کدوم یکی باشیم: برنامه‌نویسی یا دستور زبان یک زبان خاص؟",
    "پرسش ۱۵: اگه هدف‌مون اینه که بریم گوگل کار کنیم اول از همه چه‌چیزی رو سرچ کنیم؟",
    "پرسش ۱۶: در بیانیه‌ی هکرها گفته شده که جرم آن‌ها در یک کلمه چیست؟",
]


def create_QA_result(question_number: int, answer: str) -> dict:
    """
    Wrap a short answer in the expected dictionary format.

    Parameters
    ----------
    question_number:
        1-based index of the question.
    answer:
        Short free-text answer (ideally <= 4 words as enforced by the prompt).
    """

    return {
        "question_number": question_number,
        "answer": answer,
    }


def run_evaluation(chain: Runnable) -> List[dict]:
    """
    Run the full 16-question evaluation against a RAG chain.

    This may be expensive to run, so you should call it explicitly only when
    you actually want the full evaluation.
    """

    results = []
    for idx, question in enumerate(EVALUATION_QUESTIONS, start=1):
        raw_answer = chain.invoke(question)
        answer_str = str(raw_answer)
        qa = create_QA_result(idx, answer_str)
        results.append(qa)
        print(f"[evaluation] Q{idx}: {answer_str}")
    return results


def save_answers(answers: Sequence[Mapping[str, object]], path: str | Path = "answers.json") -> Path:
    """
    Save a sequence of answer dictionaries to a JSON file.

    Parameters
    ----------
    answers:
        Iterable of dictionaries with keys ``\"question_number\"`` and
        ``\"answer\"``.
    path:
        Target JSON path (relative or absolute). Defaults to ``answers.json``
        in the current working directory.
    """

    target = Path(path)
    with target.open("w", encoding="utf-8") as f:
        json.dump(list(answers), f, ensure_ascii=False, indent=4)
    print(f"[evaluation] Saved answers to {target}")
    return target

