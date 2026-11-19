#!/usr/bin/env python3
"""
PONDI Facial Management System (PFMS)
=====================================
All-in-one University Biometric Face Recognition, Student Registration,
Tuition Payment Access Control (FCFA), and Student Research Desktop Application.

Features:
    - Secure Admin Login Screen (PBKDF2-SHA256 salted password hashing + account management)
    - Real-Time Webcam & Image Face Detection and Recognition
      (OpenCV YuNet + SFace 128-D Deep Embeddings, with automatic Haar fallback)
    - Guided Student & Face Registration:
        * Matricule, Full Name, Date of Birth (exact age calculation + accuracy cross-check),
          Field, Option, Academic Year, School, Address, Phone Number,
          Amount Paid (FCFA), Date of Payment, and Official Front Portrait Capture
    - Cumulative Monthly Tuition Access Rules (FCFA):
        * September : 55,000 FCFA -> Access Granted
        * October   : 250,000 + 55,000 (Sep) = 305,000 FCFA -> Access Granted
        * November  : 140,000 + 305,000 (Oct cumulative) = 445,000 FCFA -> Access Granted
        * December  : 40,000 + 445,000 (Nov cumulative) = 485,000 FCFA -> Access Granted
    - Simultaneous Multi-Student Identity Determination displaying Name, Matricule,
      Phone Number, Date of Payment, Amount Paid, Academic Info, and Access Decision
    - Student Research Screen (Search by Matricule / Name / Field):
        * Lists all registered fields and displays every matching student's
          registered front photo alongside their full personal, academic, and payment details
    - Tuition Payment Ledger (record incremental monthly tranches & track arrears)
    - Timestamped Access Audit Logs & CSV Export

Run:
    python3 pondi_facial_management_system.py
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import sys
import threading
import time
import unicodedata
import urllib.request
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Guard against OpenCV/Qt plugin path collisions on cross-platform setups
if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ and "cv2" in os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", ""):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

import cv2
import numpy as np

from PyQt5.QtCore import QDate, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# 1. CONFIGURATION, PATHS & ACADEMIC CONSTANTS
# =============================================================================

APP_NAME = "PONDI Facial Management System"
APP_SHORT = "PFMS"
APP_VERSION = "2.0.0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "pondi_fms_data")
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
MODELS_CACHE_DIR = os.path.join(DATA_DIR, "models")
DB_FILE = os.path.join(DATA_DIR, "pondi_students.db")

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"

YUNET_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"

NUM_ENROLL_SAMPLES = 5
DEFAULT_SFACE_THRESHOLD = 0.38
DEFAULT_FALLBACK_THRESHOLD = 0.72
LOG_COOLDOWN_SECONDS = 8.0

# Exact University Tuition Schedule (FCFA) — Cumulative Monthly Rules
# Sep: 55,000 FCFA
# Oct: 250,000 + Sep (55,000) = 305,000 FCFA
# Nov: 140,000 + Oct cumulative (305,000) = 445,000 FCFA
# Dec: 40,000 + Nov cumulative (445,000) = 485,000 FCFA
TUITION_SCHEDULE: List[Dict[str, Any]] = [
    {
        "month": 9,
        "short": "Sep",
        "name": "September",
        "tranche": 55_000,
        "cumulative": 55_000,
        "rule": "September: 55,000 FCFA paid",
    },
    {
        "month": 10,
        "short": "Oct",
        "name": "October",
        "tranche": 250_000,
        "cumulative": 305_000,
        "rule": "October + September: 250,000 + 55,000 = 305,000 FCFA",
    },
    {
        "month": 11,
        "short": "Nov",
        "name": "November",
        "tranche": 140_000,
        "cumulative": 445_000,
        "rule": "November + October Cumulative: 140,000 + 305,000 = 445,000 FCFA",
    },
    {
        "month": 12,
        "short": "Dec",
        "name": "December",
        "tranche": 40_000,
        "cumulative": 485_000,
        "rule": "December + November Cumulative: 40,000 + 445,000 = 485,000 FCFA",
    },
]
TOTAL_ANNUAL_TUITION = 485_000

COMMON_FIELDS = [
    "Computer Science",
    "Software Engineering",
    "Cybersecurity & Networks",
    "Data Science & AI",
    "Electrical Engineering",
    "Civil Engineering",
    "Mechanical Engineering",
    "Medicine & Health Sciences",
    "Pharmacy",
    "Law & Political Science",
    "Economics & Management",
    "Accounting & Finance",
    "Business Administration",
    "Architecture",
    "Mathematics & Physics",
    "Biology & Biochemistry",
    "Telecommunications",
]

COMMON_OPTIONS = [
    "Artificial Intelligence",
    "Software Development",
    "Network & Security",
    "Database Systems",
    "Cloud & DevOps",
    "Power Systems",
    "Structural Engineering",
    "General Medicine",
    "Corporate Law",
    "Banking & Finance",
    "Marketing & Strategy",
    "Audit & Control",
    "Embedded Systems",
    "General Option",
]

COMMON_YEARS = [
    "Year 1 (L1)",
    "Year 2 (L2)",
    "Year 3 (L3)",
    "Year 4 (M1)",
    "Year 5 (M2)",
    "Year 6 (Doctorate)",
    "PhD",
]

COMMON_SCHOOLS = [
    "Faculty of Science",
    "National Advanced School of Engineering (ENSP)",
    "Faculty of Medicine & Biomedical Sciences",
    "Faculty of Law & Political Science",
    "Faculty of Economics & Management",
    "University Institute of Technology (IUT)",
    "ICT University",
    "School of Engineering & Technology",
    "Higher Institute of Management",
]

EVAL_MONTH_OPTIONS = [
    ("Current System Month", "current"),
    ("September (55,000 FCFA)", 9),
    ("October (305,000 FCFA)", 10),
    ("November (445,000 FCFA)", 11),
    ("December / Full Year (485,000 FCFA)", 12),
    ("Student's Payment Date Month", "payment_date"),
]


# =============================================================================
# 2. DATA MODELS
# =============================================================================

@dataclass
class AgeVerificationResult:
    dob_str: str
    age_years: int
    months: int
    days: int
    is_valid: bool
    status: str  # "VERIFIED", "ACCURATE", "WARNING", "MISMATCH", "INVALID"
    is_birthday_today: bool
    days_until_birthday: int
    declared_age: Optional[int]
    accuracy_note: str

    @property
    def badge_text(self) -> str:
        if self.status == "VERIFIED":
            return f"{self.age_years} yrs (Verified)"
        if self.status == "ACCURATE":
            return f"{self.age_years} yrs (Auto-Checked)"
        if self.status == "WARNING":
            return f"{self.age_years} yrs (Check Range)"
        if self.status == "MISMATCH":
            return f"{self.age_years} yrs (Mismatch!)"
        return "Invalid DOB"


@dataclass
class AccessDecision:
    granted: bool
    evaluation_month: int
    evaluation_month_name: str
    required_amount: int
    amount_paid: int
    shortfall: int
    remaining_annual: int
    highest_cleared: str
    rule_description: str
    status_label: str
    reason: str
    milestones: Dict[str, bool] = dc_field(default_factory=dict)


@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # shape (5, 2)
    raw_row: Optional[np.ndarray] = None
    frontal_score: float = 1.0
    sharpness: float = 0.0


@dataclass
class IdentifiedFaceResult:
    face: DetectedFace
    recognized: bool
    confidence: float
    student: Optional[Dict[str, Any]]
    age_info: Optional[AgeVerificationResult]
    access: Optional[AccessDecision]


# =============================================================================
# 3. CORE BUSINESS LOGIC (AGE ACCURACY, TUITION RULES, VALIDATION, CRYPTO)
# =============================================================================

def ensure_data_directories() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(MODELS_CACHE_DIR, exist_ok=True)


def format_fcfa(amount: int) -> str:
    try:
        return f"{int(amount):,} FCFA"
    except (TypeError, ValueError):
        return "0 FCFA"


def sanitize_ascii(text: str) -> str:
    """Convert accented Unicode characters cleanly to ASCII for OpenCV HUD overlays."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return normalized.encode("ascii", "ignore").decode("ascii")


def hash_password(password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
    """Return (salt_hex, pbkdf2_sha256_hash_hex)."""
    if salt_hex is None:
        salt_hex = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    )
    return salt_hex, dk.hex()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    _, computed = hash_password(password, salt_hex)
    return hmac.compare_digest(computed, expected_hash_hex)


def parse_iso_date(date_str: str) -> Optional[date]:
    if not date_str:
        return None
    s = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _safe_birthday_in_year(dob: date, year: int) -> date:
    """Handle Feb 29 birthdays gracefully in non-leap years."""
    if dob.month == 2 and dob.day == 29 and not calendar.isleap(year):
        return date(year, 2, 28)
    return date(year, dob.month, dob.day)


def calculate_and_verify_age(
    dob_str: str,
    declared_age: Optional[int] = None,
    reference_date: Optional[date] = None,
) -> AgeVerificationResult:
    """
    Determine exact chronological age from Date of Birth (accounting for leap years,
    month, and day) and perform multi-layer accuracy validation:
      1. Date validity & non-future check
      2. Human & university student plausibility check (typical 15–65, hard bounds 10–110)
      3. Optional declared-age cross-verification (detects off-by-1 birthday errors & typos)
    """
    ref = reference_date or date.today()
    dob = parse_iso_date(dob_str)

    if dob is None:
        return AgeVerificationResult(
            dob_str=dob_str,
            age_years=0,
            months=0,
            days=0,
            is_valid=False,
            status="INVALID",
            is_birthday_today=False,
            days_until_birthday=0,
            declared_age=declared_age,
            accuracy_note="Invalid date format. Use YYYY-MM-DD.",
        )

    if dob > ref:
        return AgeVerificationResult(
            dob_str=dob.isoformat(),
            age_years=0,
            months=0,
            days=0,
            is_valid=False,
            status="INVALID",
            is_birthday_today=False,
            days_until_birthday=0,
            declared_age=declared_age,
            accuracy_note="Date of birth cannot be in the future.",
        )

    # Exact years
    bday_this_year = _safe_birthday_in_year(dob, ref.year)
    had_birthday = ref >= bday_this_year
    years = ref.year - dob.year - (0 if had_birthday else 1)

    # Exact remaining months & days since last birthday
    last_bday = bday_this_year if had_birthday else _safe_birthday_in_year(dob, ref.year - 1)
    if ref.day >= last_bday.day:
        months = (ref.year - last_bday.year) * 12 + (ref.month - last_bday.month)
        days = ref.day - last_bday.day
    else:
        months = (ref.year - last_bday.year) * 12 + (ref.month - last_bday.month) - 1
        prev_month = ref.month - 1 if ref.month > 1 else 12
        prev_year = ref.year if ref.month > 1 else ref.year - 1
        days_in_prev = calendar.monthrange(prev_year, prev_month)[1]
        days = (days_in_prev - min(last_bday.day, days_in_prev)) + ref.day

    is_bday_today = (ref == bday_this_year)
    if is_bday_today:
        days_until = 0
    elif ref < bday_this_year:
        days_until = (bday_this_year - ref).days
    else:
        next_bday = _safe_birthday_in_year(dob, ref.year + 1)
        days_until = (next_bday - ref).days

    if years < 10 or years > 110:
        return AgeVerificationResult(
            dob_str=dob.isoformat(),
            age_years=years,
            months=months,
            days=days,
            is_valid=False,
            status="INVALID",
            is_birthday_today=is_bday_today,
            days_until_birthday=days_until,
            declared_age=declared_age,
            accuracy_note=(
                f"Computed age ({years} yrs) is outside acceptable bounds (10–110 yrs). "
                "Please verify the Date of Birth."
            ),
        )

    exact_str = f"{years} yrs, {months} mos, {days} days"
    if is_bday_today:
        exact_str += " (Birthday Today!)"

    # Cross-check against declared age if provided
    if declared_age is not None:
        diff = abs(declared_age - years)
        if diff == 0:
            if 15 <= years <= 65:
                return AgeVerificationResult(
                    dob_str=dob.isoformat(),
                    age_years=years,
                    months=months,
                    days=days,
                    is_valid=True,
                    status="VERIFIED",
                    is_birthday_today=is_bday_today,
                    days_until_birthday=days_until,
                    declared_age=declared_age,
                    accuracy_note=f"Age verified accurately ({exact_str}) — matches declared age {declared_age}.",
                )
            return AgeVerificationResult(
                dob_str=dob.isoformat(),
                age_years=years,
                months=months,
                days=days,
                is_valid=True,
                status="WARNING",
                is_birthday_today=is_bday_today,
                days_until_birthday=days_until,
                declared_age=declared_age,
                accuracy_note=(
                    f"Matches declared age ({exact_str}), but outside typical university range (15–65)."
                ),
            )
        if diff == 1:
            return AgeVerificationResult(
                dob_str=dob.isoformat(),
                age_years=years,
                months=months,
                days=days,
                is_valid=True,
                status="MISMATCH",
                is_birthday_today=is_bday_today,
                days_until_birthday=days_until,
                declared_age=declared_age,
                accuracy_note=(
                    f"Off by 1 year! DOB yields {exact_str}, but declared age is {declared_age}. "
                    "Check birth month/day or year on student ID."
                ),
            )
        return AgeVerificationResult(
            dob_str=dob.isoformat(),
            age_years=years,
            months=months,
            days=days,
            is_valid=True,
            status="MISMATCH",
            is_birthday_today=is_bday_today,
            days_until_birthday=days_until,
            declared_age=declared_age,
            accuracy_note=(
                f"Age discrepancy of {diff} years! DOB yields {years} yrs ({exact_str}), "
                f"declared age is {declared_age}."
            ),
        )

    # No declared age entered — check university plausibility range
    if 15 <= years <= 65:
        return AgeVerificationResult(
            dob_str=dob.isoformat(),
            age_years=years,
            months=months,
            days=days,
            is_valid=True,
            status="ACCURATE",
            is_birthday_today=is_bday_today,
            days_until_birthday=days_until,
            declared_age=None,
            accuracy_note=f"Auto-determined from DOB: {exact_str} (valid university student age).",
        )

    return AgeVerificationResult(
        dob_str=dob.isoformat(),
        age_years=years,
        months=months,
        days=days,
        is_valid=True,
        status="WARNING",
        is_birthday_today=is_bday_today,
        days_until_birthday=days_until,
        declared_age=None,
        accuracy_note=(
            f"Computed age is {exact_str} — outside typical university range (15–65). "
            "Please double-check DOB for accuracy."
        ),
    )


def evaluate_tuition_access(
    amount_paid: int,
    payment_date_str: str = "",
    evaluation_mode: Any = "current",
    reference_date: Optional[date] = None,
) -> AccessDecision:
    """
    Evaluate campus access strictly from the university's cumulative FCFA tuition rules:
      - September (month 9)  : >= 55,000 FCFA   -> ACCESS GRANTED
      - October   (month 10) : >= 305,000 FCFA  (250k Oct + 55k Sep) -> ACCESS GRANTED
      - November  (month 11) : >= 445,000 FCFA  (140k Nov + 305k Oct cum.) -> ACCESS GRANTED
      - December  (month 12) : >= 485,000 FCFA  (40k Dec + 445k Nov cum.) -> ACCESS GRANTED
      - Jan–Aug (post-Dec)   : >= 485,000 FCFA  (full annual tuition cleared) -> ACCESS GRANTED

    `evaluation_mode` may be:
      - "current"      : uses reference_date.month (default today's month)
      - "payment_date" : uses the month of `payment_date_str` (falls back to current month)
      - int (1..12)    : explicitly evaluates against that calendar month
    """
    ref = reference_date or date.today()
    paid = max(0, int(amount_paid or 0))

    if evaluation_mode == "payment_date":
        pdate = parse_iso_date(payment_date_str)
        target_month = pdate.month if pdate is not None else ref.month
    elif isinstance(evaluation_mode, int) and 1 <= evaluation_mode <= 12:
        target_month = evaluation_mode
    else:
        target_month = ref.month

    month_name = calendar.month_name[target_month]

    if target_month == 9:
        required = 55_000
        rule_desc = "September Rule: 55,000 FCFA required"
    elif target_month == 10:
        required = 305_000
        rule_desc = "October Rule: 250,000 FCFA (Oct) + 55,000 FCFA (Sep) = 305,000 FCFA"
    elif target_month == 11:
        required = 445_000
        rule_desc = "November Rule: 140,000 FCFA (Nov) + 305,000 FCFA (Sep+Oct) = 445,000 FCFA"
    elif target_month == 12:
        required = 485_000
        rule_desc = "December Rule: 40,000 FCFA (Dec) + 445,000 FCFA (Sep–Nov) = 485,000 FCFA"
    else:
        required = 485_000
        rule_desc = f"{month_name} (Full Academic Year): Full 485,000 FCFA tuition required"

    milestones = {
        "Sep": paid >= 55_000,
        "Oct": paid >= 305_000,
        "Nov": paid >= 445_000,
        "Dec": paid >= 485_000,
    }

    if paid >= 485_000:
        highest = "December (Full Tuition Cleared)"
    elif paid >= 445_000:
        highest = "November (Sep + Oct + Nov)"
    elif paid >= 305_000:
        highest = "October (Sep + Oct)"
    elif paid >= 55_000:
        highest = "September (Sep Only)"
    else:
        highest = "None (< 55,000 FCFA)"

    granted = paid >= required
    shortfall = max(0, required - paid)
    remaining_annual = max(0, TOTAL_ANNUAL_TUITION - paid)

    if granted:
        status_label = "ACCESS GRANTED"
        if paid >= TOTAL_ANNUAL_TUITION:
            reason = f"Full annual tuition paid ({format_fcfa(paid)}). Cleared for {month_name} & full year."
        else:
            reason = (
                f"Paid {format_fcfa(paid)} >= {format_fcfa(required)} required for {month_name}."
            )
    else:
        status_label = "ACCESS DENIED"
        reason = (
            f"Paid {format_fcfa(paid)} of {format_fcfa(required)} required for {month_name} "
            f"(Shortfall: {format_fcfa(shortfall)}; cleared through: {highest})."
        )

    return AccessDecision(
        granted=granted,
        evaluation_month=target_month,
        evaluation_month_name=month_name,
        required_amount=required,
        amount_paid=paid,
        shortfall=shortfall,
        remaining_annual=remaining_annual,
        highest_cleared=highest,
        rule_description=rule_desc,
        status_label=status_label,
        reason=reason,
        milestones=milestones,
    )


def validate_phone_number(phone: str) -> Tuple[bool, str]:
    cleaned = phone.strip()
    if not cleaned:
        return False, "Phone number is required."
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 8 or len(digits) > 15:
        return False, "Phone number must contain between 8 and 15 digits (e.g. +237 6XX XXX XXX)."
    return True, cleaned


def infer_tranche_label(cumulative_after: int, amount_added: int) -> str:
    if cumulative_after >= 485_000:
        return f"December / Full Tuition (+{format_fcfa(amount_added)})"
    if cumulative_after >= 445_000:
        return f"November Cumulative (+{format_fcfa(amount_added)})"
    if cumulative_after >= 305_000:
        return f"October Cumulative (+{format_fcfa(amount_added)})"
    if cumulative_after >= 55_000:
        return f"September Registration (+{format_fcfa(amount_added)})"
    return f"Partial Payment (+{format_fcfa(amount_added)})"


# =============================================================================
# 4. SQLITE DATABASE MANAGER
# =============================================================================

class DatabaseManager:
    """Thread-safe SQLite persistence layer for admins, students, payments, and logs."""

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        ensure_data_directories()
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()
        self._seed_default_admin()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'Administrator',
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );

                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matricule TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    date_of_birth TEXT NOT NULL,
                    age_at_registration INTEGER NOT NULL,
                    declared_age INTEGER,
                    age_verification_status TEXT NOT NULL DEFAULT 'ACCURATE',
                    field TEXT NOT NULL,
                    option_name TEXT NOT NULL,
                    academic_year TEXT NOT NULL,
                    school TEXT NOT NULL,
                    address TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    amount_paid INTEGER NOT NULL DEFAULT 0,
                    payment_date TEXT NOT NULL,
                    front_photo_path TEXT NOT NULL,
                    face_embeddings_json TEXT NOT NULL,
                    engine_type TEXT NOT NULL DEFAULT 'sface',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS payment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                    amount_added INTEGER NOT NULL,
                    cumulative_after INTEGER NOT NULL,
                    payment_date TEXT NOT NULL,
                    tranche_label TEXT NOT NULL,
                    recorded_by TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS access_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER REFERENCES students(id) ON DELETE SET NULL,
                    matricule TEXT,
                    student_name TEXT NOT NULL,
                    phone_number TEXT,
                    payment_date TEXT,
                    amount_paid INTEGER DEFAULT 0,
                    evaluation_month TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    access_status TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    scanned_by TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS admin_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT DEFAULT '',
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_students_matricule ON students(matricule);
                CREATE INDEX IF NOT EXISTS idx_students_name ON students(full_name);
                CREATE INDEX IF NOT EXISTS idx_students_field ON students(field);
                CREATE INDEX IF NOT EXISTS idx_payments_student ON payment_history(student_id);
                CREATE INDEX IF NOT EXISTS idx_access_logs_ts ON access_logs(timestamp);
                """
            )

    def _seed_default_admin(self) -> None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM admins").fetchone()
            if row["cnt"] == 0:
                salt, pw_hash = hash_password(DEFAULT_ADMIN_PASS)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    """
                    INSERT INTO admins
                        (username, full_name, role, password_salt, password_hash, is_primary, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (DEFAULT_ADMIN_USER, "System Administrator", "Super Admin", salt, pw_hash, now),
                )

    # -- Admin Authentication & Management -----------------------------------

    def authenticate_admin(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM admins WHERE username = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
            if row is None:
                return None
            if not verify_password(password, row["password_salt"], row["password_hash"]):
                return None
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute("UPDATE admins SET last_login = ? WHERE id = ?", (now, row["id"]))
            conn.execute(
                "INSERT INTO admin_audit_logs (username, action, details, timestamp) VALUES (?, ?, ?, ?)",
                (row["username"], "LOGIN", "Admin logged in successfully", now),
            )
            result = dict(row)
            result["last_login"] = now
            return result

    def log_admin_action(self, username: str, action: str, details: str = "") -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO admin_audit_logs (username, action, details, timestamp) VALUES (?, ?, ?, ?)",
                (username, action, details, now),
            )

    def list_admins(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, full_name, role, is_primary, created_at, last_login "
                "FROM admins ORDER BY is_primary DESC, username ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def create_admin(
        self,
        username: str,
        full_name: str,
        password: str,
        role: str = "Administrator",
        created_by: str = "admin",
    ) -> Tuple[bool, str]:
        uname = username.strip()
        fname = full_name.strip()
        if len(uname) < 3:
            return False, "Username must be at least 3 characters."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        if not fname:
            return False, "Full name is required."
        salt, pw_hash = hash_password(password)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO admins
                        (username, full_name, role, password_salt, password_hash, is_primary, created_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    """,
                    (uname, fname, role.strip() or "Administrator", salt, pw_hash, now),
                )
            self.log_admin_action(created_by, "CREATE_ADMIN", f"Created admin '{uname}'")
            return True, f"Admin account '{uname}' created successfully."
        except sqlite3.IntegrityError:
            return False, f"Username '{uname}' already exists."

    def change_admin_password(
        self,
        username: str,
        old_password: str,
        new_password: str,
    ) -> Tuple[bool, str]:
        if len(new_password) < 6:
            return False, "New password must be at least 6 characters."
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM admins WHERE username = ? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
            if row is None:
                return False, "Admin user not found."
            if not verify_password(old_password, row["password_salt"], row["password_hash"]):
                return False, "Current password is incorrect."
            salt, pw_hash = hash_password(new_password)
            conn.execute(
                "UPDATE admins SET password_salt = ?, password_hash = ? WHERE id = ?",
                (salt, pw_hash, row["id"]),
            )
        self.log_admin_action(username, "CHANGE_PASSWORD", "Updated admin password")
        return True, "Password updated successfully."

    def delete_admin(self, admin_id: int, actor: str) -> Tuple[bool, str]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
            if row is None:
                return False, "Admin account not found."
            if row["is_primary"]:
                return False, "Primary Super Admin account cannot be deleted."
            if row["username"].lower() == actor.lower():
                return False, "You cannot delete the currently signed-in account."
            conn.execute("DELETE FROM admins WHERE id = ?", (admin_id,))
        self.log_admin_action(actor, "DELETE_ADMIN", f"Deleted admin '{row['username']}'")
        return True, f"Deleted admin '{row['username']}'."

    # -- Student Operations --------------------------------------------------

    def generate_next_matricule(self) -> str:
        year = date.today().year
        prefix = f"PFMS-{year}-"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT matricule FROM students WHERE matricule LIKE ?",
                (f"{prefix}%",),
            ).fetchall()
        max_seq = 0
        for r in rows:
            suffix = r["matricule"][len(prefix):]
            if suffix.isdigit():
                max_seq = max(max_seq, int(suffix))
        return f"{prefix}{max_seq + 1:04d}"

    def register_student(
        self,
        matricule: str,
        full_name: str,
        date_of_birth: str,
        age_at_registration: int,
        declared_age: Optional[int],
        age_verification_status: str,
        field: str,
        option_name: str,
        academic_year: str,
        school: str,
        address: str,
        phone_number: str,
        amount_paid: int,
        payment_date: str,
        front_photo_path: str,
        embeddings: List[np.ndarray],
        engine_type: str,
        recorded_by: str,
        payment_notes: str = "Initial registration payment",
    ) -> Tuple[bool, str, Optional[int]]:
        mat = matricule.strip().upper()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_serialized = json.dumps([np.asarray(e, dtype=np.float32).tolist() for e in embeddings])
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO students (
                        matricule, full_name, date_of_birth, age_at_registration,
                        declared_age, age_verification_status, field, option_name,
                        academic_year, school, address, phone_number,
                        amount_paid, payment_date, front_photo_path,
                        face_embeddings_json, engine_type, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mat,
                        full_name.strip(),
                        date_of_birth.strip(),
                        int(age_at_registration),
                        declared_age,
                        age_verification_status,
                        field.strip(),
                        option_name.strip(),
                        academic_year.strip(),
                        school.strip(),
                        address.strip(),
                        phone_number.strip(),
                        int(amount_paid),
                        payment_date.strip(),
                        front_photo_path,
                        emb_serialized,
                        engine_type,
                        now,
                        now,
                    ),
                )
                student_id = cur.lastrowid
                if int(amount_paid) > 0:
                    conn.execute(
                        """
                        INSERT INTO payment_history (
                            student_id, amount_added, cumulative_after,
                            payment_date, tranche_label, recorded_by, notes, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            student_id,
                            int(amount_paid),
                            int(amount_paid),
                            payment_date.strip(),
                            infer_tranche_label(int(amount_paid), int(amount_paid)),
                            recorded_by,
                            payment_notes.strip(),
                            now,
                        ),
                    )
            self.log_admin_action(
                recorded_by,
                "REGISTER_STUDENT",
                f"Registered {full_name.strip()} ({mat}) — {field.strip()} — Paid {format_fcfa(amount_paid)}",
            )
            return True, f"Student '{full_name.strip()}' ({mat}) registered successfully.", student_id
        except sqlite3.IntegrityError:
            return False, f"Matricule '{mat}' is already registered in the system.", None

    def update_student_info(
        self,
        student_id: int,
        full_name: str,
        date_of_birth: str,
        age_years: int,
        declared_age: Optional[int],
        age_verification_status: str,
        field: str,
        option_name: str,
        academic_year: str,
        school: str,
        address: str,
        phone_number: str,
        updated_by: str,
        front_photo_path: Optional[str] = None,
        embeddings: Optional[List[np.ndarray]] = None,
        engine_type: Optional[str] = None,
    ) -> Tuple[bool, str]:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            existing = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
            if existing is None:
                return False, "Student not found."

            photo_to_save = front_photo_path if front_photo_path else existing["front_photo_path"]
            emb_json = (
                json.dumps([np.asarray(e, dtype=np.float32).tolist() for e in embeddings])
                if embeddings is not None
                else existing["face_embeddings_json"]
            )
            eng = engine_type if engine_type else existing["engine_type"]

            conn.execute(
                """
                UPDATE students SET
                    full_name = ?, date_of_birth = ?, age_at_registration = ?,
                    declared_age = ?, age_verification_status = ?,
                    field = ?, option_name = ?, academic_year = ?, school = ?,
                    address = ?, phone_number = ?,
                    front_photo_path = ?, face_embeddings_json = ?, engine_type = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    full_name.strip(),
                    date_of_birth.strip(),
                    int(age_years),
                    declared_age,
                    age_verification_status,
                    field.strip(),
                    option_name.strip(),
                    academic_year.strip(),
                    school.strip(),
                    address.strip(),
                    phone_number.strip(),
                    photo_to_save,
                    emb_json,
                    eng,
                    now,
                    student_id,
                ),
            )
        self.log_admin_action(
            updated_by, "UPDATE_STUDENT", f"Updated student #{student_id} ({full_name.strip()})"
        )
        return True, "Student record updated successfully."

    def add_student_payment(
        self,
        student_id: int,
        amount_added: int,
        payment_date: str,
        recorded_by: str,
        notes: str = "",
    ) -> Tuple[bool, str, int]:
        added = int(amount_added)
        if added <= 0:
            return False, "Payment amount must be greater than 0 FCFA.", 0
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
            if row is None:
                return False, "Student not found.", 0
            new_total = int(row["amount_paid"]) + added
            tranche_label = infer_tranche_label(new_total, added)
            conn.execute(
                """
                UPDATE students
                SET amount_paid = ?, payment_date = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_total, payment_date.strip(), now, student_id),
            )
            conn.execute(
                """
                INSERT INTO payment_history (
                    student_id, amount_added, cumulative_after,
                    payment_date, tranche_label, recorded_by, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    added,
                    new_total,
                    payment_date.strip(),
                    tranche_label,
                    recorded_by,
                    notes.strip(),
                    now,
                ),
            )
        self.log_admin_action(
            recorded_by,
            "RECORD_PAYMENT",
            f"Added {format_fcfa(added)} for {row['full_name']} ({row['matricule']}) -> Total {format_fcfa(new_total)}",
        )
        return True, f"Recorded {format_fcfa(added)}. New total: {format_fcfa(new_total)}.", new_total

    def delete_student(self, student_id: int, deleted_by: str) -> Tuple[bool, str]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
            if row is None:
                return False, "Student not found."
            photo_path = row["front_photo_path"]
            conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        if photo_path and os.path.exists(photo_path):
            try:
                # Only delete photos stored inside our managed PHOTOS_DIR
                if os.path.abspath(photo_path).startswith(os.path.abspath(PHOTOS_DIR)):
                    os.remove(photo_path)
            except OSError:
                pass
        self.log_admin_action(
            deleted_by, "DELETE_STUDENT", f"Deleted {row['full_name']} ({row['matricule']})"
        )
        return True, f"Student '{row['full_name']}' ({row['matricule']}) deleted."

    def _hydrate_student(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            raw_embs = json.loads(d.get("face_embeddings_json") or "[]")
            arrs = [np.asarray(v, dtype=np.float32) for v in raw_embs if len(v) > 0]
        except (ValueError, TypeError):
            arrs = []
        d["embeddings"] = arrs
        if arrs:
            centroid = np.mean(np.stack(arrs, axis=0), axis=0)
            norm = float(np.linalg.norm(centroid))
            d["centroid"] = (centroid / norm).astype(np.float32) if norm > 1e-12 else centroid.astype(np.float32)
        else:
            d["centroid"] = None
        return d

    def get_student_by_id(self, student_id: int) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
            return self._hydrate_student(row) if row else None

    def get_student_by_matricule(self, matricule: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM students WHERE matricule = ? COLLATE NOCASE",
                (matricule.strip(),),
            ).fetchone()
            return self._hydrate_student(row) if row else None

    def get_all_students(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM students ORDER BY full_name ASC").fetchall()
            return [self._hydrate_student(r) for r in rows]

    def search_students(
        self,
        query: str = "",
        search_by: str = "all",
        field_filter: str = "",
        year_filter: str = "",
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM students WHERE 1=1"
        params: List[Any] = []

        q = query.strip()
        if q:
            like = f"%{q}%"
            if search_by == "matricule":
                sql += " AND matricule LIKE ? COLLATE NOCASE"
                params.append(like)
            elif search_by == "name":
                sql += " AND full_name LIKE ? COLLATE NOCASE"
                params.append(like)
            elif search_by == "field":
                sql += " AND field LIKE ? COLLATE NOCASE"
                params.append(like)
            else:
                sql += (
                    " AND (matricule LIKE ? COLLATE NOCASE"
                    " OR full_name LIKE ? COLLATE NOCASE"
                    " OR field LIKE ? COLLATE NOCASE"
                    " OR option_name LIKE ? COLLATE NOCASE"
                    " OR school LIKE ? COLLATE NOCASE"
                    " OR phone_number LIKE ? COLLATE NOCASE)"
                )
                params.extend([like] * 6)

        if field_filter and field_filter != "All Fields":
            sql += " AND field = ? COLLATE NOCASE"
            params.append(field_filter)

        if year_filter and year_filter != "All Years":
            sql += " AND academic_year = ? COLLATE NOCASE"
            params.append(year_filter)

        sql += " ORDER BY field ASC, full_name ASC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._hydrate_student(r) for r in rows]

    def get_fields_with_counts(self) -> List[Tuple[str, int]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT field, COUNT(*) AS cnt
                FROM students
                GROUP BY field COLLATE NOCASE
                ORDER BY field ASC
                """
            ).fetchall()
            return [(r["field"], r["cnt"]) for r in rows]

    def get_distinct_years(self) -> List[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT academic_year FROM students ORDER BY academic_year ASC"
            ).fetchall()
            return [r["academic_year"] for r in rows]

    def get_payment_history(self, student_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM payment_history
                WHERE student_id = ?
                ORDER BY payment_date DESC, id DESC
                """,
                (student_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Access Logs ---------------------------------------------------------

    def log_access(
        self,
        student: Optional[Dict[str, Any]],
        evaluation_month_name: str,
        confidence: float,
        access_status: str,
        reason: str,
        scanned_by: str,
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO access_logs (
                    student_id, matricule, student_name, phone_number,
                    payment_date, amount_paid, evaluation_month, confidence,
                    access_status, reason, scanned_by, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student["id"] if student else None,
                    student["matricule"] if student else "UNREGISTERED",
                    student["full_name"] if student else "Unknown Person",
                    student["phone_number"] if student else "N/A",
                    student["payment_date"] if student else "N/A",
                    int(student["amount_paid"]) if student else 0,
                    evaluation_month_name,
                    float(confidence),
                    access_status,
                    reason,
                    scanned_by,
                    now,
                ),
            )

    def get_access_logs(
        self,
        limit: int = 300,
        status_filter: str = "ALL",
        query: str = "",
        student_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM access_logs WHERE 1=1"
        params: List[Any] = []
        if status_filter and status_filter != "ALL":
            sql += " AND access_status = ?"
            params.append(status_filter)
        if student_id is not None:
            sql += " AND student_id = ?"
            params.append(student_id)
        if query.strip():
            like = f"%{query.strip()}%"
            sql += (
                " AND (student_name LIKE ? COLLATE NOCASE"
                " OR matricule LIKE ? COLLATE NOCASE"
                " OR phone_number LIKE ? COLLATE NOCASE"
                " OR reason LIKE ? COLLATE NOCASE)"
            )
            params.extend([like] * 4)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def clear_access_logs(self, actor: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM access_logs")
        self.log_admin_action(actor, "CLEAR_ACCESS_LOGS", "Cleared all access scan logs")

    def get_dashboard_stats(self, evaluation_mode: Any = "current") -> Dict[str, Any]:
        students = self.get_all_students()
        granted = 0
        denied = 0
        total_collected = 0
        total_arrears = 0
        fields_summary: Dict[str, Dict[str, int]] = {}

        for st in students:
            dec = evaluate_tuition_access(
                st["amount_paid"], st["payment_date"], evaluation_mode=evaluation_mode
            )
            total_collected += int(st["amount_paid"])
            f = st["field"]
            if f not in fields_summary:
                fields_summary[f] = {"total": 0, "granted": 0, "denied": 0}
            fields_summary[f]["total"] += 1
            if dec.granted:
                granted += 1
                fields_summary[f]["granted"] += 1
            else:
                denied += 1
                total_arrears += dec.shortfall
                fields_summary[f]["denied"] += 1

        return {
            "total_students": len(students),
            "access_granted": granted,
            "access_denied": denied,
            "total_collected": total_collected,
            "total_arrears": total_arrears,
            "distinct_fields": len(fields_summary),
            "fields_summary": fields_summary,
            "recent_logs": self.get_access_logs(limit=12),
        }


# =============================================================================
# 5. FACE DETECTION & RECOGNITION ENGINE (YuNet + SFace ONNX + Haar Fallback)
# =============================================================================

class FaceRecognitionEngine:
    """
    High-accuracy Face Detection & Recognition Engine:
      - Primary : OpenCV Zoo YuNet (face_detection_yunet_2023mar.onnx)
                  + SFace 128-D Deep Embeddings (face_recognition_sface_2021dec.onnx)
      - Fallback: OpenCV Haar Cascade + Multi-Region Normalized Gradient/Intensity Descriptor
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.engine_type: str = "uninitialized"
        self.detector: Any = None
        self.recognizer: Any = None
        self.haar_cascade: Any = None
        self.yunet_path: Optional[str] = None
        self.sface_path: Optional[str] = None
        self._input_size: Tuple[int, int] = (640, 480)
        self._load_engine()

    @staticmethod
    def _candidate_model_dirs() -> List[str]:
        return [
            os.path.join(BASE_DIR, "Mes models"),
            BASE_DIR,
            MODELS_CACHE_DIR,
            os.path.join(BASE_DIR, "models"),
        ]

    @classmethod
    def locate_onnx_models(cls) -> Tuple[Optional[str], Optional[str]]:
        yunet_found = None
        sface_found = None
        for d in cls._candidate_model_dirs():
            yp = os.path.join(d, YUNET_FILENAME)
            sp = os.path.join(d, SFACE_FILENAME)
            if yunet_found is None and os.path.isfile(yp) and os.path.getsize(yp) > 10_000:
                yunet_found = yp
            if sface_found is None and os.path.isfile(sp) and os.path.getsize(sp) > 100_000:
                sface_found = sp
        return yunet_found, sface_found

    def _load_engine(self) -> None:
        with self._lock:
            yp, sp = self.locate_onnx_models()
            if (
                yp
                and sp
                and hasattr(cv2, "FaceDetectorYN")
                and hasattr(cv2, "FaceRecognizerSF")
            ):
                try:
                    self.detector = cv2.FaceDetectorYN.create(
                        yp,
                        "",
                        self._input_size,
                        score_threshold=0.72,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                    self.recognizer = cv2.FaceRecognizerSF.create(sp, "")
                    self.yunet_path = yp
                    self.sface_path = sp
                    self.engine_type = "sface"
                    return
                except Exception:
                    pass

            # Graceful fallback to Haar Cascade + normalized descriptor
            hc_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
            hc_path = os.path.join(hc_dir, "haarcascade_frontalface_default.xml") if hc_dir else ""
            if os.path.isfile(hc_path):
                self.haar_cascade = cv2.CascadeClassifier(hc_path)
            else:
                self.haar_cascade = None
            self.detector = None
            self.recognizer = None
            self.engine_type = "haar_fallback"

    def download_onnx_models(self) -> Tuple[bool, str]:
        """Download YuNet and SFace ONNX models into pondi_fms_data/models/ and reload."""
        ensure_data_directories()
        yp = os.path.join(MODELS_CACHE_DIR, YUNET_FILENAME)
        sp = os.path.join(MODELS_CACHE_DIR, SFACE_FILENAME)
        try:
            if not os.path.isfile(yp) or os.path.getsize(yp) < 10_000:
                urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, yp)
            if not os.path.isfile(sp) or os.path.getsize(sp) < 100_000:
                urllib.request.urlretrieve(SFACE_DOWNLOAD_URL, sp)
            self._load_engine()
            if self.engine_type == "sface":
                return True, "YuNet & SFace ONNX models downloaded and activated!"
            return False, "Models downloaded, but OpenCV could not initialize FaceDetectorYN/SF."
        except Exception as exc:
            return False, f"Failed to download ONNX models: {exc}"

    @property
    def default_threshold(self) -> float:
        return DEFAULT_SFACE_THRESHOLD if self.engine_type == "sface" else DEFAULT_FALLBACK_THRESHOLD

    @property
    def engine_label(self) -> str:
        if self.engine_type == "sface":
            return "YuNet + SFace Deep ONNX (128-D)"
        return "Haar Cascade + Descriptor (Fallback)"

    @staticmethod
    def _compute_sharpness(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x, y, w, h = bbox
        roi = frame_bgr[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            return 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _compute_frontal_score(landmarks: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> float:
        """Score [0..1] estimating how frontal & level a face is using 5 YuNet landmarks."""
        if landmarks is None or landmarks.shape != (5, 2):
            return 0.85
        re_pt, le_pt, nose_pt = landmarks[0], landmarks[1], landmarks[2]
        eye_dx = abs(float(le_pt[0] - re_pt[0]))
        eye_dy = abs(float(le_pt[1] - re_pt[1]))
        if eye_dx < 5.0:
            return 0.3
        # Roll penalty (eyes should be roughly horizontal)
        roll_ratio = min(1.0, eye_dy / eye_dx)
        # Yaw symmetry (nose x should sit near midpoint of eyes)
        eye_mid_x = 0.5 * (float(re_pt[0]) + float(le_pt[0]))
        yaw_ratio = min(1.0, abs(float(nose_pt[0]) - eye_mid_x) / (0.5 * eye_dx))
        score = 1.0 - 0.5 * roll_ratio - 0.5 * yaw_ratio
        return float(max(0.0, min(1.0, score)))

    def detect_faces(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        results: List[DetectedFace] = []

        with self._lock:
            if self.engine_type == "sface" and self.detector is not None:
                if self._input_size != (w, h):
                    self.detector.setInputSize((w, h))
                    self._input_size = (w, h)
                _, faces = self.detector.detect(frame_bgr)
                if faces is not None:
                    for row in faces:
                        x = max(0, int(round(row[0])))
                        y = max(0, int(round(row[1])))
                        fw = max(1, int(round(row[2])))
                        fh = max(1, int(round(row[3])))
                        fw = min(fw, w - x)
                        fh = min(fh, h - y)
                        if fw < 20 or fh < 20:
                            continue
                        lms = row[4:14].reshape(5, 2).copy()
                        conf = float(row[14]) if len(row) > 14 else 0.9
                        bbox = (x, y, fw, fh)
                        results.append(
                            DetectedFace(
                                bbox=bbox,
                                confidence=conf,
                                landmarks=lms,
                                raw_row=row.copy(),
                                frontal_score=self._compute_frontal_score(lms, bbox),
                                sharpness=self._compute_sharpness(frame_bgr, bbox),
                            )
                        )
            elif self.haar_cascade is not None:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                rects = self.haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48)
                )
                for (x, y, fw, fh) in rects:
                    bbox = (int(x), int(y), int(fw), int(fh))
                    results.append(
                        DetectedFace(
                            bbox=bbox,
                            confidence=0.85,
                            landmarks=None,
                            raw_row=None,
                            frontal_score=0.85,
                            sharpness=self._compute_sharpness(frame_bgr, bbox),
                        )
                    )

        # Sort largest face first so primary subject is index 0
        results.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        return results

    def _fallback_embedding(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Multi-grid normalized gradient + histogram descriptor (128-D) when ONNX unavailable."""
        x, y, fw, fh = bbox
        roi = frame_bgr[max(0, y):y + fh, max(0, x):x + fw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(gray).astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        feats: List[np.ndarray] = []
        cell = 16  # 4x4 grid = 16 cells * 8 orientation bins = 128 dims
        for r in range(0, 64, cell):
            for c in range(0, 64, cell):
                m_cell = mag[r:r + cell, c:c + cell]
                a_cell = ang[r:r + cell, c:c + cell]
                hist, _ = np.histogram(a_cell, bins=8, range=(0, 360), weights=m_cell)
                feats.append(hist.astype(np.float32))
        vec = np.concatenate(feats)
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 1e-12 else vec

    def extract_embedding(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        with self._lock:
            if self.engine_type == "sface" and self.recognizer is not None and face.raw_row is not None:
                aligned = self.recognizer.alignCrop(frame_bgr, face.raw_row)
                feat = self.recognizer.feature(aligned).flatten().copy().astype(np.float32)
                norm = float(np.linalg.norm(feat))
                return (feat / norm).astype(np.float32) if norm > 1e-12 else feat
        return self._fallback_embedding(frame_bgr, face.bbox)

    @staticmethod
    def crop_front_portrait(
        frame_bgr: np.ndarray,
        face: DetectedFace,
        target_size: Tuple[int, int] = (320, 360),
    ) -> np.ndarray:
        """Crop a clean, centered ID-card style front portrait around the detected face."""
        h, w = frame_bgr.shape[:2]
        x, y, fw, fh = face.bbox
        cx = x + fw * 0.5
        cy = y + fh * 0.48

        target_ratio = target_size[0] / float(target_size[1])
        crop_h = max(fh * 2.05, fw * 1.9 / target_ratio)
        crop_w = crop_h * target_ratio

        x1 = int(round(cx - crop_w * 0.5))
        y1 = int(round(cy - crop_h * 0.45))
        x2 = int(round(x1 + crop_w))
        y2 = int(round(y1 + crop_h))

        # Pad with dark slate border if face is close to frame edges
        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if any((pad_l, pad_t, pad_r, pad_b)):
            padded = cv2.copyMakeBorder(
                frame_bgr,
                pad_t,
                pad_b,
                pad_l,
                pad_r,
                cv2.BORDER_CONSTANT,
                value=(30, 41, 59),
            )
            crop = padded[y1 + pad_t:y2 + pad_t, x1 + pad_l:x2 + pad_l]
        else:
            crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            crop = frame_bgr[max(0, y):y + fh, max(0, x):x + fw]
        interp = cv2.INTER_AREA if crop.shape[0] > target_size[1] else cv2.INTER_CUBIC
        return cv2.resize(crop, target_size, interpolation=interp)

    def match_embedding(
        self,
        probe: np.ndarray,
        students: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Match a normalized probe embedding against all registered students."""
        thr = self.default_threshold if threshold is None else float(threshold)
        p_norm = float(np.linalg.norm(probe))
        if p_norm <= 1e-12:
            return None, 0.0
        probe_u = (probe / p_norm).astype(np.float32)

        best_student: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for st in students:
            if st.get("engine_type") and st["engine_type"] != self.engine_type:
                continue
            embs = st.get("embeddings") or []
            if not embs:
                continue

            sample_scores = [float(np.dot(probe_u, e)) for e in embs if e.shape == probe_u.shape]
            if not sample_scores:
                continue
            max_sample = max(sample_scores)
            centroid = st.get("centroid")
            if centroid is not None and centroid.shape == probe_u.shape:
                cent_score = float(np.dot(probe_u, centroid))
                score = 0.5 * cent_score + 0.5 * max_sample
            else:
                score = max_sample

            if score > best_score:
                best_score = score
                best_student = st

        if best_student is not None and best_score >= thr:
            return best_student, best_score
        return None, max(0.0, best_score)

    def identify_all_faces(
        self,
        frame_bgr: np.ndarray,
        students: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        evaluation_mode: Any = "current",
    ) -> List[IdentifiedFaceResult]:
        """Detect and identify EVERY face present in `frame_bgr` simultaneously."""
        faces = self.detect_faces(frame_bgr)
        identified: List[IdentifiedFaceResult] = []
        for face in faces:
            emb = self.extract_embedding(frame_bgr, face)
            matched_student, score = self.match_embedding(emb, students, threshold=threshold)
            if matched_student is not None:
                age_info = calculate_and_verify_age(
                    matched_student["date_of_birth"],
                    declared_age=matched_student.get("declared_age"),
                )
                access = evaluate_tuition_access(
                    matched_student["amount_paid"],
                    matched_student["payment_date"],
                    evaluation_mode=evaluation_mode,
                )
                identified.append(
                    IdentifiedFaceResult(
                        face=face,
                        recognized=True,
                        confidence=score,
                        student=matched_student,
                        age_info=age_info,
                        access=access,
                    )
                )
            else:
                identified.append(
                    IdentifiedFaceResult(
                        face=face,
                        recognized=False,
                        confidence=score,
                        student=None,
                        age_info=None,
                        access=None,
                    )
                )
        return identified


# =============================================================================
# 6. IMAGE, PIXMAP & VIDEO HUD RENDERING UTILITIES
# =============================================================================

def bgr_to_qpixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def make_placeholder_portrait(name: str, width: int = 136, height: int = 156) -> QPixmap:
    pix = QPixmap(width, height)
    pix.fill(QColor("#1e293b"))
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)

    # Subtle rounded border
    painter.setPen(QPen(QColor("#334155"), 2))
    painter.setBrush(QColor("#0f172a"))
    painter.drawRoundedRect(2, 2, width - 4, height - 4, 10, 10)

    # Initials circle
    parts = [p for p in (name or "ST").strip().split() if p]
    initials = (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()
    circ_d = min(width, height) // 2
    cx = (width - circ_d) // 2
    cy = (height - circ_d) // 2 - 8
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#2563eb"))
    painter.drawEllipse(cx, cy, circ_d, circ_d)

    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Helvetica Neue", max(14, circ_d // 3), QFont.Bold))
    painter.drawText(cx, cy, circ_d, circ_d, Qt.AlignCenter, initials)

    painter.setPen(QColor("#94a3b8"))
    painter.setFont(QFont("Helvetica Neue", 9))
    painter.drawText(0, height - 28, width, 20, Qt.AlignCenter, "No Front Photo")
    painter.end()
    return pix


def load_portrait_pixmap(
    photo_path: str,
    fallback_name: str = "Student",
    width: int = 136,
    height: int = 156,
    border_color: Optional[str] = None,
) -> QPixmap:
    if photo_path and os.path.isfile(photo_path):
        raw = QPixmap(photo_path)
        if not raw.isNull():
            scaled = raw.scaled(
                width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            # Center-crop to exact target rectangle
            sx = max(0, (scaled.width() - width) // 2)
            sy = max(0, (scaled.height() - height) // 2)
            cropped = scaled.copy(sx, sy, width, height)

            out = QPixmap(width, height)
            out.fill(Qt.transparent)
            painter = QPainter(out)
            painter.setRenderHint(QPainter.Antialiasing)
            path = QPainterPath()
            path.addRoundedRect(1, 1, width - 2, height - 2, 10, 10)
            painter.setClipPath(path)
            painter.drawPixmap(0, 0, cropped)
            painter.setClipping(False)
            if border_color:
                painter.setPen(QPen(QColor(border_color), 3))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(2, 2, width - 4, height - 4, 10, 10)
            painter.end()
            return out
    return make_placeholder_portrait(fallback_name, width, height)


def draw_scan_overlays(
    frame_bgr: np.ndarray,
    results: List[IdentifiedFaceResult],
) -> np.ndarray:
    """Draw sleek bounding boxes, corner accents, and non-overlapping status banners."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    placed_banners: List[Tuple[int, int, int, int]] = []

    for res in results:
        x, y, fw, fh = res.face.bbox
        if res.recognized and res.student and res.access:
            if res.access.granted:
                color = (129, 185, 16)   # Emerald BGR (#10b981)
                tag = "ACCESS GRANTED"
            else:
                color = (11, 158, 245)   # Amber BGR (#f59e0b)
                tag = "DENIED (TUITION)"
            name_ascii = sanitize_ascii(res.student["full_name"])
            if len(name_ascii) > 22:
                name_ascii = name_ascii[:21] + "."
            mat_ascii = sanitize_ascii(res.student["matricule"])
            pdate_ascii = sanitize_ascii(res.student["payment_date"])
            line1 = f"{name_ascii} ({res.confidence * 100:.0f}%)"
            line2 = f"{tag} | {mat_ascii} | {pdate_ascii}"
        else:
            color = (68, 68, 239)        # Red BGR (#ef4444)
            line1 = "UNREGISTERED / UNKNOWN"
            line2 = f"ACCESS DENIED ({res.confidence * 100:.0f}%)"

        # Outer rectangle + corner brackets
        cv2.rectangle(canvas, (x, y), (x + fw, y + fh), color, 2)
        cl = max(12, min(26, fw // 5))
        t = 4
        cv2.line(canvas, (x, y), (x + cl, y), color, t)
        cv2.line(canvas, (x, y), (x, y + cl), color, t)
        cv2.line(canvas, (x + fw, y), (x + fw - cl, y), color, t)
        cv2.line(canvas, (x + fw, y), (x + fw, y + cl), color, t)
        cv2.line(canvas, (x, y + fh), (x + cl, y + fh), color, t)
        cv2.line(canvas, (x, y + fh), (x, y + fh - cl), color, t)
        cv2.line(canvas, (x + fw, y + fh), (x + fw - cl, y + fh), color, t)
        cv2.line(canvas, (x + fw, y + fh), (x + fw, y + fh - cl), color, t)

        # Compact two-line banner with overlap avoidance
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w1, h1), _ = cv2.getTextSize(line1, font, 0.48, 2)
        (w2, h2), _ = cv2.getTextSize(line2, font, 0.40, 1)
        box_w = max(w1, w2) + 14
        box_h = h1 + h2 + 16

        bx1 = max(4, min(x, w - box_w - 4))
        by1 = y - box_h - 6
        if by1 < 4:
            by1 = min(h - box_h - 4, y + fh + 6)

        # If overlapping a previously placed banner, try below the face box first
        def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])

        cand = (bx1, by1, min(w - 4, bx1 + box_w), min(h - 4, by1 + box_h))
        if any(_overlaps(cand, prev) for prev in placed_banners):
            alt_y = min(h - box_h - 4, y + fh + 6)
            alt_cand = (bx1, alt_y, min(w - 4, bx1 + box_w), min(h - 4, alt_y + box_h))
            if not any(_overlaps(alt_cand, prev) for prev in placed_banners):
                cand = alt_cand
        bx1, by1, bx2, by2 = cand
        placed_banners.append(cand)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.84, canvas, 0.16, 0, canvas)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)

        cv2.putText(
            canvas, line1, (bx1 + 7, by1 + h1 + 5),
            font, 0.48, (248, 250, 252), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, line2, (bx1 + 7, by1 + h1 + h2 + 11),
            font, 0.40, color, 1, cv2.LINE_AA,
        )

    return canvas


def draw_enrollment_guide(
    frame_bgr: np.ndarray,
    faces: List[DetectedFace],
    collected: int,
    total_needed: int,
    active: bool,
) -> np.ndarray:
    """Draw a clean oval/box frontal face alignment guide during registration."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # Center target box guide
    gw, gh = int(w * 0.38), int(h * 0.62)
    gx1, gy1 = (w - gw) // 2, (h - gh) // 2
    gx2, gy2 = gx1 + gw, gy1 + gh
    guide_color = (59, 130, 246) if not active else (16, 185, 129)
    cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), guide_color, 2)

    if len(faces) == 1:
        f = faces[0]
        x, y, fw, fh = f.bbox
        col = (129, 185, 16) if f.frontal_score >= 0.6 else (11, 158, 245)
        cv2.rectangle(canvas, (x, y), (x + fw, y + fh), col, 2)
        if f.landmarks is not None:
            for pt in f.landmarks:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, (246, 180, 59), -1)
        status_txt = (
            f"Capturing Front Face: {collected}/{total_needed}"
            if active
            else f"Face Ready (Frontal: {int(f.frontal_score * 100)}%, Sharp: {int(f.sharpness)})"
        )
    elif len(faces) > 1:
        for f in faces:
            x, y, fw, fh = f.bbox
            cv2.rectangle(canvas, (x, y), (x + fw, y + fh), (68, 68, 239), 2)
        status_txt = "Multiple faces detected - keep ONLY 1 student in frame!"
    else:
        status_txt = "Position student's front face inside the frame"

    cv2.rectangle(canvas, (0, 0), (w, 34), (15, 23, 42), -1)
    cv2.putText(
        canvas, status_txt, (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 2, cv2.LINE_AA,
    )
    return canvas


# =============================================================================
# 7. GLOBAL MODERN DARK UNIVERSITY THEME (QSS)
# =============================================================================

APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #f1f5f9;
    font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QFrame#Card QLabel, QFrame#LoginCard QLabel, QFrame#StudentCard QLabel, QFrame#IdentCard QLabel, QFrame#TopBar QLabel {
    background-color: transparent;
}

QCheckBox {
    background-color: transparent;
    spacing: 6px;
}

QFrame#Sidebar {
    background-color: #0b1120;
    border-right: 1px solid #1e293b;
}

QFrame#TopBar {
    background-color: #0b1120;
    border-bottom: 1px solid #1e293b;
}

QFrame#Card, QFrame#LoginCard, QFrame#StudentCard, QFrame#IdentCard {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}

QFrame#StudentCard:hover {
    border: 1px solid #3b82f6;
}

QFrame#KpiCard {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}

QLabel#PageTitle {
    font-size: 19px;
    font-weight: 700;
    color: #f8fafc;
    background: transparent;
}

QLabel#SectionTitle {
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
    background: transparent;
}

QLabel#MutedLabel {
    color: #94a3b8;
    font-size: 12px;
    background: transparent;
}

QLabel#FieldLabel {
    color: #cbd5e1;
    font-weight: 600;
    font-size: 12px;
    background: transparent;
}

QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}

QPushButton#SecondaryBtn {
    background-color: #334155;
    color: #f1f5f9;
}

QPushButton#SecondaryBtn:hover {
    background-color: #475569;
}

QPushButton#SuccessBtn {
    background-color: #059669;
    color: #ffffff;
}

QPushButton#SuccessBtn:hover {
    background-color: #047857;
}

QPushButton#DangerBtn {
    background-color: #dc2626;
    color: #ffffff;
}

QPushButton#DangerBtn:hover {
    background-color: #b91c1c;
}

QPushButton#WarningBtn {
    background-color: #d97706;
    color: #ffffff;
}

QPushButton#WarningBtn:hover {
    background-color: #b45309;
}

QPushButton#NavBtn {
    background-color: transparent;
    color: #94a3b8;
    text-align: left;
    padding: 11px 14px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
}

QPushButton#NavBtn:hover {
    background-color: #1e293b;
    color: #f8fafc;
}

QPushButton#NavBtn[active="true"] {
    background-color: #1d4ed8;
    color: #ffffff;
}

QPushButton#FieldChip {
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 5px 12px;
    font-size: 12px;
    font-weight: 600;
}

QPushButton#FieldChip:hover {
    border-color: #3b82f6;
    color: #ffffff;
}

QPushButton#FieldChip[selected="true"] {
    background-color: #2563eb;
    border-color: #60a5fa;
    color: #ffffff;
}

QLineEdit, QComboBox, QSpinBox, QDateEdit, QTextEdit {
    background-color: #0b1220;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 7px 10px;
    selection-background-color: #2563eb;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDateEdit:focus, QTextEdit:focus {
    border: 1px solid #3b82f6;
}

QComboBox::drop-down, QDateEdit::drop-down {
    border: none;
    width: 22px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f8fafc;
    selection-background-color: #2563eb;
    border: 1px solid #334155;
}

QTableWidget {
    background-color: #0b1220;
    alternate-background-color: #131d31;
    gridline-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    color: #e2e8f0;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #cbd5e1;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #334155;
    font-weight: 700;
}

QProgressBar {
    background-color: #0b1220;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #f8fafc;
    font-weight: 600;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #10b981;
    border-radius: 5px;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #0f172a;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #334155;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #3b82f6;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QDialog {
    background-color: #0f172a;
}

QMessageBox {
    background-color: #0f172a;
}
"""


# =============================================================================
# 8. REUSABLE UI COMPONENTS & DIALOGS
# =============================================================================

def make_badge_label(text: str, bg: str, fg: str = "#ffffff") -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background-color: {bg}; color: {fg}; border-radius: 6px; "
        f"padding: 3px 9px; font-weight: 700; font-size: 11px;"
    )
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return lbl


def make_milestone_strip(milestones: Dict[str, bool]) -> QWidget:
    """Compact visual strip showing Sep / Oct / Nov / Dec tuition clearance."""
    container = QWidget()
    container.setStyleSheet("background: transparent;")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    items = [
        ("Sep 55k", milestones.get("Sep", False)),
        ("Oct 305k", milestones.get("Oct", False)),
        ("Nov 445k", milestones.get("Nov", False)),
        ("Dec 485k", milestones.get("Dec", False)),
    ]
    for label, cleared in items:
        prefix = "✓ " if cleared else "✕ "
        bg = "#065f46" if cleared else "#334155"
        fg = "#a7f3d0" if cleared else "#94a3b8"
        b = make_badge_label(prefix + label, bg, fg)
        layout.addWidget(b)
    layout.addStretch()
    return container


class ClickablePhotoLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PhotoZoomDialog(QDialog):
    """Full-resolution preview of a student's registered front picture."""

    def __init__(self, student: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Front Photo — {student['full_name']} ({student['matricule']})")
        self.setMinimumSize(420, 480)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel(f"{student['full_name']}  •  {student['matricule']}")
        title.setObjectName("SectionTitle")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        sub = QLabel(f"{student['field']} — {student['option_name']} ({student['academic_year']})")
        sub.setObjectName("MutedLabel")
        sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub)

        photo_lbl = QLabel()
        photo_lbl.setAlignment(Qt.AlignCenter)
        pix = load_portrait_pixmap(
            student.get("front_photo_path", ""),
            student["full_name"],
            width=340,
            height=380,
            border_color="#3b82f6",
        )
        photo_lbl.setPixmap(pix)
        layout.addWidget(photo_lbl)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("SecondaryBtn")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class AddPaymentDialog(QDialog):
    """Record an additional tuition installment (FCFA) with live cumulative access preview."""

    def __init__(
        self,
        db: DatabaseManager,
        student: Dict[str, Any],
        admin_username: str,
        evaluation_mode: Any = "current",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.student = student
        self.admin_username = admin_username
        self.evaluation_mode = evaluation_mode
        self.setWindowTitle(f"Add Tuition Payment — {student['full_name']}")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel(f"Record Payment for {student['full_name']} ({student['matricule']})")
        header.setObjectName("SectionTitle")
        layout.addWidget(header)

        current_paid = int(student["amount_paid"])
        summary = QLabel(
            f"Current Cumulative Paid: <b>{format_fcfa(current_paid)}</b>  |  "
            f"Last Payment Date: <b>{student['payment_date']}</b>  |  "
            f"Annual Total: <b>{format_fcfa(TOTAL_ANNUAL_TUITION)}</b>"
        )
        summary.setStyleSheet("background: #1e293b; padding: 10px; border-radius: 8px;")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # Quick tranche helper buttons
        quick_lbl = QLabel("Quick Fill Remaining Balance for Monthly Milestone:")
        quick_lbl.setObjectName("FieldLabel")
        layout.addWidget(quick_lbl)

        quick_row = QHBoxLayout()
        for item in TUITION_SCHEDULE:
            diff = max(0, item["cumulative"] - current_paid)
            btn_txt = (
                f"{item['short']} (+{diff:,})"
                if diff > 0
                else f"{item['short']} ✓"
            )
            btn = QPushButton(btn_txt)
            btn.setObjectName("SecondaryBtn")
            btn.setEnabled(diff > 0)
            btn.clicked.connect(lambda _, d=diff: self.amount_spin.setValue(d))
            quick_row.addWidget(btn)
        layout.addLayout(quick_row)

        form = QFormLayout()
        form.setSpacing(10)

        self.amount_spin = QSpinBox()
        self.amount_spin.setRange(0, 5_000_000)
        self.amount_spin.setSingleStep(5_000)
        self.amount_spin.setSuffix(" FCFA")
        # Suggest next unpaid tranche delta by default
        next_delta = 0
        for item in TUITION_SCHEDULE:
            if current_paid < item["cumulative"]:
                next_delta = item["cumulative"] - current_paid
                break
        self.amount_spin.setValue(next_delta if next_delta > 0 else 40_000)
        self.amount_spin.valueChanged.connect(self._update_preview)
        form.addRow("Amount to Add:", self.amount_spin)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.dateChanged.connect(self._update_preview)
        form.addRow("Date of Payment:", self.date_edit)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Receipt number / payment method (e.g. Mobile Money, Bank)")
        form.addRow("Notes / Receipt:", self.notes_edit)

        layout.addLayout(form)

        self.preview_label = QLabel()
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("SecondaryBtn")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Payment")
        save_btn.setObjectName("SuccessBtn")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._update_preview()

    def _update_preview(self) -> None:
        new_total = int(self.student["amount_paid"]) + self.amount_spin.value()
        pdate = self.date_edit.date().toString("yyyy-MM-dd")
        dec = evaluate_tuition_access(new_total, pdate, evaluation_mode=self.evaluation_mode)
        bg = "#064e3b" if dec.granted else "#7f1d1d"
        fg = "#d1fae5" if dec.granted else "#fee2e2"
        self.preview_label.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 10px; border-radius: 8px;"
        )
        self.preview_label.setText(
            f"<b>After Payment Total: {format_fcfa(new_total)}</b>  •  "
            f"<b>{dec.status_label}</b> ({dec.evaluation_month_name})<br>"
            f"Cleared Milestone: {dec.highest_cleared}<br>{dec.reason}"
        )

    def _save(self) -> None:
        amt = self.amount_spin.value()
        if amt <= 0:
            QMessageBox.warning(self, "Invalid Amount", "Please enter an amount greater than 0 FCFA.")
            return
        pdate = self.date_edit.date().toString("yyyy-MM-dd")
        ok, msg, _ = self.db.add_student_payment(
            self.student["id"],
            amt,
            pdate,
            self.admin_username,
            self.notes_edit.text(),
        )
        if ok:
            QMessageBox.information(self, "Payment Recorded", msg)
            self.accept()
        else:
            QMessageBox.warning(self, "Error", msg)


class EditStudentDialog(QDialog):
    """Edit registered student information with live DOB age accuracy checking & optional photo update."""

    def __init__(
        self,
        db: DatabaseManager,
        engine: FaceRecognitionEngine,
        student: Dict[str, Any],
        admin_username: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.student = student
        self.admin_username = admin_username
        self.new_photo_path: Optional[str] = None
        self.new_embeddings: Optional[List[np.ndarray]] = None
        self.new_engine_type: Optional[str] = None

        self.setWindowTitle(f"Edit Student — {student['full_name']} ({student['matricule']})")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        hdr = QLabel(f"Edit Student Profile — {student['matricule']}")
        hdr.setObjectName("SectionTitle")
        layout.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(8)

        self.name_edit = QLineEdit(student["full_name"])
        form.addRow("Full Name *:", self.name_edit)

        dob_row = QHBoxLayout()
        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        parsed = parse_iso_date(student["date_of_birth"])
        if parsed:
            self.dob_edit.setDate(QDate(parsed.year, parsed.month, parsed.day))
        else:
            self.dob_edit.setDate(QDate(2004, 1, 1))
        self.dob_edit.dateChanged.connect(self._recheck_age)
        dob_row.addWidget(self.dob_edit, 2)

        self.declared_age_spin = QSpinBox()
        self.declared_age_spin.setRange(0, 110)
        self.declared_age_spin.setSpecialValueText("Auto (Not Set)")
        self.declared_age_spin.setValue(int(student["declared_age"]) if student.get("declared_age") else 0)
        self.declared_age_spin.valueChanged.connect(self._recheck_age)
        dob_row.addWidget(QLabel("Declared Age Check:"))
        dob_row.addWidget(self.declared_age_spin, 1)
        form.addRow("Date of Birth *:", dob_row)

        self.age_status_lbl = QLabel()
        self.age_status_lbl.setWordWrap(True)
        form.addRow("Age Verification:", self.age_status_lbl)

        self.phone_edit = QLineEdit(student["phone_number"])
        form.addRow("Phone Number *:", self.phone_edit)

        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        self.field_combo.addItems(COMMON_FIELDS)
        self.field_combo.setCurrentText(student["field"])
        form.addRow("Field *:", self.field_combo)

        self.option_combo = QComboBox()
        self.option_combo.setEditable(True)
        self.option_combo.addItems(COMMON_OPTIONS)
        self.option_combo.setCurrentText(student["option_name"])
        form.addRow("Option *:", self.option_combo)

        self.year_combo = QComboBox()
        self.year_combo.setEditable(True)
        self.year_combo.addItems(COMMON_YEARS)
        self.year_combo.setCurrentText(student["academic_year"])
        form.addRow("Year *:", self.year_combo)

        self.school_combo = QComboBox()
        self.school_combo.setEditable(True)
        self.school_combo.addItems(COMMON_SCHOOLS)
        self.school_combo.setCurrentText(student["school"])
        form.addRow("School *:", self.school_combo)

        self.address_edit = QLineEdit(student["address"])
        form.addRow("Address *:", self.address_edit)

        # Optional replacement front photo from file
        photo_row = QHBoxLayout()
        self.photo_status_lbl = QLabel("Keeping current front photo & face biometric")
        self.photo_status_lbl.setObjectName("MutedLabel")
        replace_photo_btn = QPushButton("Replace Photo from File...")
        replace_photo_btn.setObjectName("SecondaryBtn")
        replace_photo_btn.clicked.connect(self._replace_photo)
        photo_row.addWidget(self.photo_status_lbl, 1)
        photo_row.addWidget(replace_photo_btn)
        form.addRow("Front Photo:", photo_row)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("SecondaryBtn")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Changes")
        save_btn.setObjectName("SuccessBtn")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._recheck_age()

    def _recheck_age(self) -> None:
        dob_s = self.dob_edit.date().toString("yyyy-MM-dd")
        decl = self.declared_age_spin.value() or None
        res = calculate_and_verify_age(dob_s, declared_age=decl)
        colors = {
            "VERIFIED": ("#064e3b", "#a7f3d0"),
            "ACCURATE": ("#1e3a5f", "#bfdbfe"),
            "WARNING": ("#78350f", "#fde68a"),
            "MISMATCH": ("#7f1d1d", "#fecaca"),
            "INVALID": ("#7f1d1d", "#fecaca"),
        }
        bg, fg = colors.get(res.status, ("#1e293b", "#e2e8f0"))
        self.age_status_lbl.setStyleSheet(
            f"background: {bg}; color: {fg}; padding: 6px 10px; border-radius: 6px; font-weight: 600;"
        )
        self.age_status_lbl.setText(f"[{res.status}] {res.accuracy_note}")

    def _replace_photo(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Front Face Photo", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not fpath:
            return
        img = cv2.imread(fpath)
        if img is None:
            QMessageBox.warning(self, "Image Error", "Could not read the selected image.")
            return
        faces = self.engine.detect_faces(img)
        if not faces:
            QMessageBox.warning(
                self, "No Face Detected", "No clear frontal face detected in this image."
            )
            return
        face = faces[0]
        emb = self.engine.extract_embedding(img, face)
        portrait = self.engine.crop_front_portrait(img, face)
        safe_mat = re.sub(r"[^A-Za-z0-9_-]", "_", self.student["matricule"])
        dest = os.path.join(PHOTOS_DIR, f"{safe_mat}_front_{int(time.time())}.jpg")
        cv2.imwrite(dest, portrait, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        self.new_photo_path = dest
        self.new_embeddings = [emb]
        self.new_engine_type = self.engine.engine_type
        self.photo_status_lbl.setText(f"New front photo & embedding ready ({os.path.basename(fpath)})")
        self.photo_status_lbl.setStyleSheet("color: #34d399; font-weight: 600;")

    def _save(self) -> None:
        name = self.name_edit.text().strip()
        field = self.field_combo.currentText().strip()
        option = self.option_combo.currentText().strip()
        year = self.year_combo.currentText().strip()
        school = self.school_combo.currentText().strip()
        address = self.address_edit.text().strip()
        phone_ok, phone_msg = validate_phone_number(self.phone_edit.text())

        if not all([name, field, option, year, school, address]):
            QMessageBox.warning(self, "Missing Fields", "Please fill in all required fields.")
            return
        if not phone_ok:
            QMessageBox.warning(self, "Invalid Phone", phone_msg)
            return

        dob_s = self.dob_edit.date().toString("yyyy-MM-dd")
        decl = self.declared_age_spin.value() or None
        age_res = calculate_and_verify_age(dob_s, declared_age=decl)
        if not age_res.is_valid:
            QMessageBox.warning(self, "Invalid Date of Birth", age_res.accuracy_note)
            return
        if age_res.status in ("MISMATCH", "WARNING"):
            reply = QMessageBox.question(
                self,
                "Age Accuracy Alert",
                f"{age_res.accuracy_note}\n\n"
                f"Save using DOB-calculated age ({age_res.age_years} years)?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        ok, msg = self.db.update_student_info(
            student_id=self.student["id"],
            full_name=name,
            date_of_birth=age_res.dob_str,
            age_years=age_res.age_years,
            declared_age=decl,
            age_verification_status=age_res.status,
            field=field,
            option_name=option,
            academic_year=year,
            school=school,
            address=address,
            phone_number=phone_msg,
            updated_by=self.admin_username,
            front_photo_path=self.new_photo_path,
            embeddings=self.new_embeddings,
            engine_type=self.new_engine_type,
        )
        if ok:
            QMessageBox.information(self, "Updated", msg)
            self.accept()
        else:
            QMessageBox.warning(self, "Error", msg)


class StudentDetailDialog(QDialog):
    """Comprehensive Student Dossier showing Front Photo, Personal Info, Payment History & Scan Logs."""

    def __init__(
        self,
        db: DatabaseManager,
        student: Dict[str, Any],
        evaluation_mode: Any = "current",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.student = student
        self.setWindowTitle(f"Student Profile — {student['full_name']} ({student['matricule']})")
        self.setMinimumSize(780, 620)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        age_res = calculate_and_verify_age(
            student["date_of_birth"], declared_age=student.get("declared_age")
        )
        access = evaluate_tuition_access(
            student["amount_paid"], student["payment_date"], evaluation_mode=evaluation_mode
        )

        # Top header card: Photo beside comprehensive info
        top_card = QFrame()
        top_card.setObjectName("Card")
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(16, 16, 16, 16)
        top_layout.setSpacing(18)

        border_col = "#10b981" if access.granted else "#ef4444"
        photo_lbl = ClickablePhotoLabel()
        photo_lbl.setPixmap(
            load_portrait_pixmap(
                student.get("front_photo_path", ""),
                student["full_name"],
                width=160,
                height=188,
                border_color=border_col,
            )
        )
        photo_lbl.setToolTip("Click to view full front photo")
        photo_lbl.clicked.connect(lambda: PhotoZoomDialog(student, self).exec_())
        top_layout.addWidget(photo_lbl, 0, Qt.AlignTop)

        info_col = QVBoxLayout()
        info_col.setSpacing(6)

        title_row = QHBoxLayout()
        name_lbl = QLabel(student["full_name"])
        name_lbl.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; background: transparent;")
        title_row.addWidget(name_lbl)
        title_row.addWidget(make_badge_label(student["matricule"], "#1d4ed8"))
        title_row.addWidget(
            make_badge_label(
                access.status_label,
                "#059669" if access.granted else "#dc2626",
            )
        )
        title_row.addStretch()
        info_col.addLayout(title_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(5)
        details = [
            ("Field:", student["field"]),
            ("Option:", student["option_name"]),
            ("Year:", student["academic_year"]),
            ("School:", student["school"]),
            ("Date of Birth:", f"{student['date_of_birth']}  •  {age_res.badge_text}"),
            ("Age Check:", age_res.accuracy_note),
            ("Phone Number:", student["phone_number"]),
            ("Address:", student["address"]),
            ("Amount Paid:", f"{format_fcfa(student['amount_paid'])} / {format_fcfa(TOTAL_ANNUAL_TUITION)}"),
            ("Last Payment Date:", student["payment_date"]),
            ("Biometric Samples:", f"{len(student.get('embeddings', []))} ({student.get('engine_type', 'sface')})"),
            ("Access Evaluation:", f"{access.evaluation_month_name} — {access.reason}"),
        ]
        for idx, (k, v) in enumerate(details):
            r = idx // 2
            c = (idx % 2) * 2
            kl = QLabel(k)
            kl.setObjectName("MutedLabel")
            vl = QLabel(str(v))
            vl.setWordWrap(True)
            vl.setStyleSheet("color: #f1f5f9; font-weight: 600; background: transparent;")
            grid.addWidget(kl, r, c, Qt.AlignTop)
            grid.addWidget(vl, r, c + 1, Qt.AlignTop)
        info_col.addLayout(grid)
        info_col.addWidget(make_milestone_strip(access.milestones))
        top_layout.addLayout(info_col, 1)
        layout.addWidget(top_card)

        # Payment History Table
        pay_hdr = QLabel("Payment History Ledger")
        pay_hdr.setObjectName("SectionTitle")
        layout.addWidget(pay_hdr)

        pay_table = QTableWidget()
        pay_table.setColumnCount(6)
        pay_table.setHorizontalHeaderLabels(
            ["Payment Date", "Amount Added", "Cumulative Total", "Tranche", "Recorded By", "Notes"]
        )
        pay_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pay_table.verticalHeader().setVisible(False)
        pay_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pay_table.setAlternatingRowColors(True)

        history = db.get_payment_history(student["id"])
        pay_table.setRowCount(len(history))
        for r, hrow in enumerate(history):
            pay_table.setItem(r, 0, QTableWidgetItem(hrow["payment_date"]))
            pay_table.setItem(r, 1, QTableWidgetItem(format_fcfa(hrow["amount_added"])))
            pay_table.setItem(r, 2, QTableWidgetItem(format_fcfa(hrow["cumulative_after"])))
            pay_table.setItem(r, 3, QTableWidgetItem(hrow["tranche_label"]))
            pay_table.setItem(r, 4, QTableWidgetItem(hrow["recorded_by"]))
            pay_table.setItem(r, 5, QTableWidgetItem(hrow.get("notes", "")))
        layout.addWidget(pay_table, 1)

        # Recent Scan Logs for this student
        log_hdr = QLabel("Recent Face Recognition Scans")
        log_hdr.setObjectName("SectionTitle")
        layout.addWidget(log_hdr)

        log_table = QTableWidget()
        log_table.setColumnCount(5)
        log_table.setHorizontalHeaderLabels(
            ["Timestamp", "Confidence", "Check Month", "Decision", "Reason"]
        )
        log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        log_table.verticalHeader().setVisible(False)
        log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        log_table.setAlternatingRowColors(True)

        logs = db.get_access_logs(limit=25, student_id=student["id"])
        log_table.setRowCount(len(logs))
        for r, lrow in enumerate(logs):
            log_table.setItem(r, 0, QTableWidgetItem(lrow["timestamp"]))
            log_table.setItem(r, 1, QTableWidgetItem(f"{lrow['confidence'] * 100:.1f}%"))
            log_table.setItem(r, 2, QTableWidgetItem(lrow["evaluation_month"]))
            log_table.setItem(r, 3, QTableWidgetItem(lrow["access_status"]))
            log_table.setItem(r, 4, QTableWidgetItem(lrow["reason"]))
        layout.addWidget(log_table, 1)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("SecondaryBtn")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, Qt.AlignRight)


class StudentResearchCard(QFrame):
    """
    Signature Research Screen Card:
    Displays the student's registered FRONT PICTURE / PHOTO on the LEFT,
    and ALL student personal, academic, and payment information neatly listed BESIDE it!
    """

    payment_requested = pyqtSignal(dict)
    edit_requested = pyqtSignal(dict)
    detail_requested = pyqtSignal(dict)
    delete_requested = pyqtSignal(dict)

    def __init__(
        self,
        student: Dict[str, Any],
        evaluation_mode: Any = "current",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.student = student
        self.setObjectName("StudentCard")

        age_res = calculate_and_verify_age(
            student["date_of_birth"], declared_age=student.get("declared_age")
        )
        access = evaluate_tuition_access(
            student["amount_paid"], student["payment_date"], evaluation_mode=evaluation_mode
        )

        outer = QHBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(16)

        # LEFT: Registered Front Picture / Photo
        photo_col = QVBoxLayout()
        photo_col.setSpacing(6)
        border_col = "#10b981" if access.granted else "#ef4444"
        photo_lbl = ClickablePhotoLabel()
        photo_lbl.setPixmap(
            load_portrait_pixmap(
                student.get("front_photo_path", ""),
                student["full_name"],
                width=136,
                height=160,
                border_color=border_col,
            )
        )
        photo_lbl.setFixedSize(136, 160)
        photo_lbl.setToolTip("Click to enlarge registered front photo")
        photo_lbl.clicked.connect(lambda: PhotoZoomDialog(student, self).exec_())
        photo_col.addWidget(photo_lbl, 0, Qt.AlignTop | Qt.AlignHCenter)

        photo_caption = QLabel("Front Photo")
        photo_caption.setObjectName("MutedLabel")
        photo_caption.setAlignment(Qt.AlignCenter)
        photo_col.addWidget(photo_caption)
        photo_col.addStretch()
        outer.addLayout(photo_col)

        # RIGHT (BESIDE THE PHOTO): Complete Student Info
        info_col = QVBoxLayout()
        info_col.setSpacing(6)

        # Header row: Name + Matricule + Field + Access Badge
        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(8)
        name_lbl = QLabel(student["full_name"])
        name_lbl.setStyleSheet(
            "font-size: 16px; font-weight: 800; color: #f8fafc; background: transparent;"
        )
        hdr_row.addWidget(name_lbl)
        hdr_row.addWidget(make_badge_label(student["matricule"], "#1d4ed8"))
        hdr_row.addWidget(make_badge_label(student["field"], "#312e81", "#c7d2fe"))
        hdr_row.addStretch()
        hdr_row.addWidget(
            make_badge_label(
                f"● {access.status_label} ({access.evaluation_month_name[:3]})",
                "#065f46" if access.granted else "#991b1b",
                "#a7f3d0" if access.granted else "#fecaca",
            )
        )
        info_col.addLayout(hdr_row)

        # Structured 2-column info grid beside the front photo
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        rows_data = [
            ("Option:", student["option_name"], "Academic Year:", student["academic_year"]),
            ("School:", student["school"], "Date of Birth / Age:", f"{student['date_of_birth']} ({age_res.badge_text})"),
            ("Phone Number:", student["phone_number"], "Address:", student["address"]),
            (
                "Amount Paid:",
                f"{format_fcfa(student['amount_paid'])}  (Remaining: {format_fcfa(access.remaining_annual)})",
                "Date of Payment:",
                student["payment_date"],
            ),
        ]
        for r_idx, (k1, v1, k2, v2) in enumerate(rows_data):
            l_k1 = QLabel(k1)
            l_k1.setObjectName("MutedLabel")
            l_v1 = QLabel(str(v1))
            l_v1.setWordWrap(True)
            l_v1.setStyleSheet("color: #e2e8f0; font-weight: 600; background: transparent;")

            l_k2 = QLabel(k2)
            l_k2.setObjectName("MutedLabel")
            l_v2 = QLabel(str(v2))
            l_v2.setWordWrap(True)
            l_v2.setStyleSheet("color: #e2e8f0; font-weight: 600; background: transparent;")

            grid.addWidget(l_k1, r_idx, 0, Qt.AlignTop)
            grid.addWidget(l_v1, r_idx, 1, Qt.AlignTop)
            grid.addWidget(l_k2, r_idx, 2, Qt.AlignTop)
            grid.addWidget(l_v2, r_idx, 3, Qt.AlignTop)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        info_col.addLayout(grid)

        # Milestone clearance strip + Access summary
        milestone_row = QHBoxLayout()
        milestone_row.addWidget(make_milestone_strip(access.milestones))
        if not access.granted:
            due_lbl = QLabel(f"Due for {access.evaluation_month_name}: {format_fcfa(access.shortfall)}")
            due_lbl.setStyleSheet("color: #fca5a5; font-weight: 700; font-size: 12px; background: transparent;")
            milestone_row.addWidget(due_lbl)
        info_col.addLayout(milestone_row)

        # Action buttons row
        actions_row = QHBoxLayout()
        actions_row.setSpacing(8)
        pay_btn = QPushButton("+ Add Payment")
        pay_btn.setObjectName("SuccessBtn")
        pay_btn.setFixedHeight(30)
        pay_btn.clicked.connect(lambda: self.payment_requested.emit(self.student))

        detail_btn = QPushButton("Full Dossier")
        detail_btn.setObjectName("SecondaryBtn")
        detail_btn.setFixedHeight(30)
        detail_btn.clicked.connect(lambda: self.detail_requested.emit(self.student))

        edit_btn = QPushButton("Edit")
        edit_btn.setObjectName("SecondaryBtn")
        edit_btn.setFixedHeight(30)
        edit_btn.clicked.connect(lambda: self.edit_requested.emit(self.student))

        del_btn = QPushButton("Delete")
        del_btn.setObjectName("DangerBtn")
        del_btn.setFixedHeight(30)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self.student))

        actions_row.addWidget(pay_btn)
        actions_row.addWidget(detail_btn)
        actions_row.addWidget(edit_btn)
        actions_row.addStretch()
        actions_row.addWidget(del_btn)
        info_col.addLayout(actions_row)

        outer.addLayout(info_col, 1)


class IdentifiedStudentCard(QFrame):
    """
    Live Scan Identity Card displayed beside the camera feed for EVERY detected person,
    prominently surfacing Identity, Phone Number, Date of Payment, Amount Paid & Access Decision.
    """

    def __init__(self, res: IdentifiedFaceResult, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("IdentCard")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        if res.recognized and res.student and res.access and res.age_info:
            st = res.student
            acc = res.access
            border = "#10b981" if acc.granted else "#f59e0b"
            self.setStyleSheet(
                f"QFrame#IdentCard {{ background: #1e293b; border: 2px solid {border}; border-radius: 10px; }}"
            )

            photo = QLabel()
            photo.setPixmap(
                load_portrait_pixmap(
                    st.get("front_photo_path", ""),
                    st["full_name"],
                    width=82,
                    height=96,
                    border_color=border,
                )
            )
            photo.setFixedSize(82, 96)
            layout.addWidget(photo, 0, Qt.AlignTop)

            col = QVBoxLayout()
            col.setSpacing(3)

            top = QHBoxLayout()
            nm = QLabel(st["full_name"])
            nm.setStyleSheet("font-size: 14px; font-weight: 800; color: #f8fafc; background: transparent;")
            top.addWidget(nm)
            top.addStretch()
            top.addWidget(
                make_badge_label(
                    acc.status_label,
                    "#059669" if acc.granted else "#d97706",
                )
            )
            col.addLayout(top)

            meta1 = QLabel(
                f"<b>Matricule:</b> {st['matricule']}  •  "
                f"<b>Match:</b> {res.confidence * 100:.1f}%  •  "
                f"<b>Age:</b> {res.age_info.age_years} yrs"
            )
            meta1.setStyleSheet("color: #cbd5e1; font-size: 12px; background: transparent;")
            meta1.setWordWrap(True)
            col.addWidget(meta1)

            meta2 = QLabel(
                f"<b>Phone:</b> <span style='color:#60a5fa;'>{st['phone_number']}</span>  •  "
                f"<b>Payment Date:</b> <span style='color:#fbbf24;'>{st['payment_date']}</span>"
            )
            meta2.setStyleSheet("color: #f1f5f9; font-size: 12px; background: transparent;")
            meta2.setWordWrap(True)
            col.addWidget(meta2)

            meta3 = QLabel(
                f"<b>Field:</b> {st['field']} ({st['option_name']}) • {st['academic_year']} • {st['school']}"
            )
            meta3.setStyleSheet("color: #94a3b8; font-size: 11px; background: transparent;")
            meta3.setWordWrap(True)
            col.addWidget(meta3)

            pay_line = QLabel(
                f"<b>Paid:</b> {format_fcfa(st['amount_paid'])} / {format_fcfa(acc.required_amount)} "
                f"({acc.evaluation_month_name}) — {acc.highest_cleared}"
            )
            pay_line.setStyleSheet(
                f"color: {'#6ee7b7' if acc.granted else '#fcd34d'}; font-size: 11px; background: transparent;"
            )
            pay_line.setWordWrap(True)
            col.addWidget(pay_line)

            layout.addLayout(col, 1)
        else:
            self.setStyleSheet(
                "QFrame#IdentCard { background: #1e293b; border: 2px solid #ef4444; border-radius: 10px; }"
            )
            icon = QLabel("⚠")
            icon.setStyleSheet("font-size: 28px; color: #ef4444; background: transparent;")
            icon.setFixedWidth(44)
            icon.setAlignment(Qt.AlignCenter)
            layout.addWidget(icon)

            col = QVBoxLayout()
            col.setSpacing(3)
            hdr = QLabel("UNREGISTERED / UNKNOWN INDIVIDUAL")
            hdr.setStyleSheet("font-size: 13px; font-weight: 800; color: #fca5a5; background: transparent;")
            sub = QLabel(
                f"ACCESS DENIED — Face not matched in student database "
                f"(Best similarity: {res.confidence * 100:.1f}%)."
            )
            sub.setStyleSheet("color: #cbd5e1; font-size: 12px; background: transparent;")
            sub.setWordWrap(True)
            col.addWidget(hdr)
            col.addWidget(sub)
            layout.addLayout(col, 1)


# =============================================================================
# 9. ADMIN LOGIN SCREEN
# =============================================================================

class AdminLoginScreen(QWidget):
    """Branded Admin Authentication Screen shown before accessing the management console."""

    login_succeeded = pyqtSignal(dict)

    def __init__(self, db: DatabaseManager, engine: FaceRecognitionEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self._failed_attempts = 0
        self._lockout_until = 0.0

        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignCenter)
        outer.setContentsMargins(24, 24, 24, 24)

        card = QFrame()
        card.setObjectName("LoginCard")
        card.setFixedWidth(440)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(32, 32, 32, 28)
        card_layout.setSpacing(14)

        # Emblem & Title
        emblem = QLabel("PFMS")
        emblem.setAlignment(Qt.AlignCenter)
        emblem.setFixedSize(68, 68)
        emblem.setStyleSheet(
            "background-color: #2563eb; color: #ffffff; font-size: 18px; font-weight: 900; "
            "border-radius: 34px;"
        )
        card_layout.addWidget(emblem, 0, Qt.AlignHCenter)

        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; background: transparent;")
        card_layout.addWidget(title)

        subtitle = QLabel("University Biometric Face Recognition & Tuition Access Control")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setObjectName("MutedLabel")
        card_layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #334155;")
        card_layout.addWidget(sep)

        login_hdr = QLabel("Administrator Sign In")
        login_hdr.setStyleSheet("font-size: 15px; font-weight: 700; color: #e2e8f0; background: transparent;")
        card_layout.addWidget(login_hdr)

        u_lbl = QLabel("Admin Username")
        u_lbl.setObjectName("FieldLabel")
        card_layout.addWidget(u_lbl)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Enter administrator username")
        self.username_edit.setText(DEFAULT_ADMIN_USER)
        self.username_edit.returnPressed.connect(self._attempt_login)
        card_layout.addWidget(self.username_edit)

        p_lbl = QLabel("Password")
        p_lbl.setObjectName("FieldLabel")
        card_layout.addWidget(p_lbl)

        pw_row = QHBoxLayout()
        pw_row.setSpacing(6)
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.returnPressed.connect(self._attempt_login)
        pw_row.addWidget(self.password_edit, 1)

        self.show_pw_btn = QPushButton("Show")
        self.show_pw_btn.setObjectName("SecondaryBtn")
        self.show_pw_btn.setCheckable(True)
        self.show_pw_btn.setFixedWidth(64)
        self.show_pw_btn.toggled.connect(self._toggle_pw)
        pw_row.addWidget(self.show_pw_btn)
        card_layout.addLayout(pw_row)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        self.error_label.setStyleSheet(
            "background-color: #7f1d1d; color: #fecaca; padding: 8px 10px; "
            "border-radius: 7px; font-weight: 600;"
        )
        card_layout.addWidget(self.error_label)

        self.login_btn = QPushButton("Sign In to Management Console")
        self.login_btn.setFixedHeight(42)
        self.login_btn.setStyleSheet("font-size: 14px; font-weight: 700;")
        self.login_btn.clicked.connect(self._attempt_login)
        card_layout.addWidget(self.login_btn)

        # Helpful default login hint & engine status footer
        hint_box = QLabel(
            f"Default Admin Credentials: <b>{DEFAULT_ADMIN_USER}</b> / <b>{DEFAULT_ADMIN_PASS}</b><br>"
            f"<i>You can update passwords & add staff admins in Settings after signing in.</i><br>"
            f"Biometric Engine: <b>{self.engine.engine_label}</b>"
        )
        hint_box.setWordWrap(True)
        hint_box.setAlignment(Qt.AlignCenter)
        hint_box.setStyleSheet(
            "background-color: #0b1220; color: #94a3b8; border: 1px solid #1e293b; "
            "border-radius: 8px; padding: 10px; font-size: 11px;"
        )
        card_layout.addWidget(hint_box)

        outer.addWidget(card)

    def _toggle_pw(self, checked: bool) -> None:
        self.password_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.show_pw_btn.setText("Hide" if checked else "Show")

    def reset_fields(self) -> None:
        self.password_edit.clear()
        self.error_label.setVisible(False)
        self.password_edit.setFocus()

    def _attempt_login(self) -> None:
        now = time.time()
        if now < self._lockout_until:
            wait = int(math.ceil(self._lockout_until - now))
            self._show_error(f"Too many failed attempts. Please wait {wait}s before trying again.")
            return

        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        if not username or not password:
            self._show_error("Please enter both your admin username and password.")
            return

        admin = self.db.authenticate_admin(username, password)
        if admin is None:
            self._failed_attempts += 1
            if self._failed_attempts >= 5:
                self._lockout_until = time.time() + 15.0
                self._failed_attempts = 0
                self._show_error("5 consecutive failed attempts. Login temporarily paused for 15 seconds.")
            else:
                self._show_error(
                    f"Invalid username or password (Attempt {self._failed_attempts}/5)."
                )
            return

        self._failed_attempts = 0
        self.error_label.setVisible(False)
        self.password_edit.clear()
        self.login_succeeded.emit(admin)

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(True)


# =============================================================================
# 10. APPLICATION PAGES (DASHBOARD, LIVE SCANNER, REGISTER STUDENT)
# =============================================================================

class DashboardPage(QWidget):
    """Executive overview: KPIs, Cumulative Tuition Rules Reference, Field Breakdown & Recent Scans."""

    navigate_requested = pyqtSignal(str)

    def __init__(self, db: DatabaseManager, engine: FaceRecognitionEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.evaluation_mode: Any = "current"

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.main_layout = QVBoxLayout(content)
        self.main_layout.setContentsMargins(20, 18, 20, 20)
        self.main_layout.setSpacing(16)
        scroll.setWidget(content)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Quick action bar
        action_bar = QHBoxLayout()
        welcome = QLabel("Campus Biometric & Tuition Overview")
        welcome.setObjectName("SectionTitle")
        action_bar.addWidget(welcome)
        action_bar.addStretch()

        btn_scan = QPushButton("Open Live Face Scanner")
        btn_scan.setObjectName("SuccessBtn")
        btn_scan.clicked.connect(lambda: self.navigate_requested.emit("scanner"))
        btn_reg = QPushButton("+ Register New Student")
        btn_reg.clicked.connect(lambda: self.navigate_requested.emit("register"))
        btn_search = QPushButton("Student Research")
        btn_search.setObjectName("SecondaryBtn")
        btn_search.clicked.connect(lambda: self.navigate_requested.emit("research"))
        action_bar.addWidget(btn_scan)
        action_bar.addWidget(btn_reg)
        action_bar.addWidget(btn_search)
        self.main_layout.addLayout(action_bar)

        # KPI Cards Row
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(12)
        self.kpi_total = self._make_kpi_card("Registered Students", "0", "#3b82f6")
        self.kpi_granted = self._make_kpi_card("Access Granted", "0", "#10b981")
        self.kpi_denied = self._make_kpi_card("Access Denied (Tuition Due)", "0", "#ef4444")
        self.kpi_collected = self._make_kpi_card("Total Tuition Collected", "0 FCFA", "#f59e0b")
        self.kpi_fields = self._make_kpi_card("Registered Fields", "0", "#8b5cf6")

        kpi_grid.addWidget(self.kpi_total[0], 0, 0)
        kpi_grid.addWidget(self.kpi_granted[0], 0, 1)
        kpi_grid.addWidget(self.kpi_denied[0], 0, 2)
        kpi_grid.addWidget(self.kpi_collected[0], 0, 3)
        kpi_grid.addWidget(self.kpi_fields[0], 0, 4)
        self.main_layout.addLayout(kpi_grid)

        # Middle split: Tuition Access Schedule Reference + Fields Breakdown
        mid_row = QHBoxLayout()
        mid_row.setSpacing(14)

        # Tuition Schedule Reference Card
        sched_card = QFrame()
        sched_card.setObjectName("Card")
        sched_layout = QVBoxLayout(sched_card)
        sched_layout.setContentsMargins(16, 14, 16, 14)
        sched_layout.setSpacing(8)

        shdr = QLabel("Official Monthly Tuition Access Rules (FCFA)")
        shdr.setObjectName("SectionTitle")
        sched_layout.addWidget(shdr)

        self.active_rule_lbl = QLabel("")
        self.active_rule_lbl.setStyleSheet(
            "background: #172554; color: #93c5fd; padding: 7px 10px; border-radius: 7px; font-weight: 600;"
        )
        self.active_rule_lbl.setWordWrap(True)
        sched_layout.addWidget(self.active_rule_lbl)

        sched_table = QTableWidget(len(TUITION_SCHEDULE), 4)
        sched_table.setHorizontalHeaderLabels(
            ["Month", "Monthly Tranche", "Cumulative Required", "Access Status"]
        )
        sched_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sched_table.verticalHeader().setVisible(False)
        sched_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        sched_table.setFixedHeight(175)
        for r, item in enumerate(TUITION_SCHEDULE):
            sched_table.setItem(r, 0, QTableWidgetItem(item["name"]))
            sched_table.setItem(r, 1, QTableWidgetItem(f"+ {format_fcfa(item['tranche'])}"))
            sched_table.setItem(r, 2, QTableWidgetItem(format_fcfa(item["cumulative"])))
            st_item = QTableWidgetItem("ACCESS GRANTED")
            st_item.setForeground(QColor("#34d399"))
            sched_table.setItem(r, 3, st_item)
        sched_layout.addWidget(sched_table)
        mid_row.addWidget(sched_card, 3)

        # Field Breakdown Card
        field_card = QFrame()
        field_card.setObjectName("Card")
        field_layout = QVBoxLayout(field_card)
        field_layout.setContentsMargins(16, 14, 16, 14)
        field_layout.setSpacing(8)

        fhdr = QLabel("Students by Field")
        fhdr.setObjectName("SectionTitle")
        field_layout.addWidget(fhdr)

        self.field_table = QTableWidget(0, 4)
        self.field_table.setHorizontalHeaderLabels(["Field", "Total", "Granted", "Denied"])
        fh = self.field_table.horizontalHeader()
        fh.setSectionResizeMode(0, QHeaderView.Stretch)
        for ci in (1, 2, 3):
            fh.setSectionResizeMode(ci, QHeaderView.Fixed)
            self.field_table.setColumnWidth(ci, 68)
        self.field_table.verticalHeader().setVisible(False)
        self.field_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.field_table.setAlternatingRowColors(True)
        self.field_table.setMinimumHeight(175)
        field_layout.addWidget(self.field_table)
        mid_row.addWidget(field_card, 2)

        self.main_layout.addLayout(mid_row)

        # Recent Access Scan Activity
        recent_card = QFrame()
        recent_card.setObjectName("Card")
        recent_layout = QVBoxLayout(recent_card)
        recent_layout.setContentsMargins(16, 14, 16, 14)
        recent_layout.setSpacing(8)

        rhdr_row = QHBoxLayout()
        rhdr = QLabel("Recent Face Recognition Scans")
        rhdr.setObjectName("SectionTitle")
        rhdr_row.addWidget(rhdr)
        rhdr_row.addStretch()
        view_all_btn = QPushButton("View All Logs")
        view_all_btn.setObjectName("SecondaryBtn")
        view_all_btn.clicked.connect(lambda: self.navigate_requested.emit("logs"))
        rhdr_row.addWidget(view_all_btn)
        recent_layout.addLayout(rhdr_row)

        self.recent_table = QTableWidget(0, 7)
        self.recent_table.setHorizontalHeaderLabels(
            ["Timestamp", "Student", "Matricule", "Phone Number", "Payment Date", "Status", "Reason"]
        )
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recent_table.setAlternatingRowColors(True)
        self.recent_table.setMinimumHeight(210)
        recent_layout.addWidget(self.recent_table)
        self.main_layout.addWidget(recent_card)

    @staticmethod
    def _make_kpi_card(title: str, value: str, accent: str) -> Tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("KpiCard")
        card.setStyleSheet(
            f"QFrame#KpiCard {{ background: #1e293b; border: 1px solid #334155; "
            f"border-left: 4px solid {accent}; border-radius: 10px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)
        t_lbl = QLabel(title)
        t_lbl.setObjectName("MutedLabel")
        v_lbl = QLabel(value)
        v_lbl.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {accent}; background: transparent;")
        lay.addWidget(t_lbl)
        lay.addWidget(v_lbl)
        return card, v_lbl

    def set_evaluation_mode(self, mode: Any) -> None:
        self.evaluation_mode = mode
        self.refresh()

    def refresh(self) -> None:
        stats = self.db.get_dashboard_stats(evaluation_mode=self.evaluation_mode)
        self.kpi_total[1].setText(str(stats["total_students"]))
        self.kpi_granted[1].setText(str(stats["access_granted"]))
        self.kpi_denied[1].setText(str(stats["access_denied"]))
        self.kpi_collected[1].setText(format_fcfa(stats["total_collected"]))
        self.kpi_fields[1].setText(str(stats["distinct_fields"]))

        sample_dec = evaluate_tuition_access(0, "", evaluation_mode=self.evaluation_mode)
        if self.evaluation_mode == "payment_date":
            self.active_rule_lbl.setText(
                "Active Mode: Evaluating each student against their own Payment Date's month."
            )
        else:
            self.active_rule_lbl.setText(f"Active Check: {sample_dec.rule_description}")

        fs = stats["fields_summary"]
        self.field_table.setRowCount(len(fs))
        for r, (fname, counts) in enumerate(sorted(fs.items())):
            self.field_table.setItem(r, 0, QTableWidgetItem(fname))
            self.field_table.setItem(r, 1, QTableWidgetItem(str(counts["total"])))
            g_item = QTableWidgetItem(str(counts["granted"]))
            g_item.setForeground(QColor("#34d399"))
            d_item = QTableWidgetItem(str(counts["denied"]))
            d_item.setForeground(QColor("#f87171"))
            self.field_table.setItem(r, 2, g_item)
            self.field_table.setItem(r, 3, d_item)

        logs = stats["recent_logs"]
        self.recent_table.setRowCount(len(logs))
        for r, lrow in enumerate(logs):
            self.recent_table.setItem(r, 0, QTableWidgetItem(lrow["timestamp"]))
            self.recent_table.setItem(r, 1, QTableWidgetItem(lrow["student_name"]))
            self.recent_table.setItem(r, 2, QTableWidgetItem(lrow.get("matricule") or ""))
            self.recent_table.setItem(r, 3, QTableWidgetItem(lrow.get("phone_number") or ""))
            self.recent_table.setItem(r, 4, QTableWidgetItem(lrow.get("payment_date") or ""))
            st_item = QTableWidgetItem(lrow["access_status"])
            if lrow["access_status"] == "GRANTED":
                st_item.setForeground(QColor("#34d399"))
            elif lrow["access_status"] == "DENIED_TUITION":
                st_item.setForeground(QColor("#fbbf24"))
            else:
                st_item.setForeground(QColor("#f87171"))
            self.recent_table.setItem(r, 5, st_item)
            self.recent_table.setItem(r, 6, QTableWidgetItem(lrow["reason"]))


class LiveScannerPage(QWidget):
    """
    Real-Time Biometric Face Scanner & Multi-Student Identity Determination Screen.
    Supports live webcam streaming AND single/group photo file scanning.
    Determines the identity of EVERY concerned registered student simultaneously,
    showing Phone Number, Date of Payment, Amount Paid, Field/Option/Year/School & Access Status.
    """

    logs_updated = pyqtSignal()

    def __init__(
        self,
        db: DatabaseManager,
        engine: FaceRecognitionEngine,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.admin_username = DEFAULT_ADMIN_USER
        self.evaluation_mode: Any = "current"

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._process_camera_frame)
        self.students_cache: List[Dict[str, Any]] = []
        self._last_logged_times: Dict[str, float] = {}
        self._last_card_signature: str = ""
        self._frame_counter = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # Controls Bar (clean 2-row layout so no button or badge ever clips)
        ctrl = QFrame()
        ctrl.setObjectName("Card")
        ctrl_outer = QVBoxLayout(ctrl)
        ctrl_outer.setContentsMargins(12, 10, 12, 10)
        ctrl_outer.setSpacing(8)

        ctrl_row1 = QHBoxLayout()
        ctrl_row1.setSpacing(10)
        ctrl_row1.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems(["Camera 0 (Default)", "Camera 1 (External)", "Camera 2"])
        self.cam_combo.setFixedWidth(165)
        ctrl_row1.addWidget(self.cam_combo)

        self.start_btn = QPushButton("▶ Start Live Camera")
        self.start_btn.setObjectName("SuccessBtn")
        self.start_btn.clicked.connect(self.start_camera)
        ctrl_row1.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ Stop Camera")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_camera)
        ctrl_row1.addWidget(self.stop_btn)

        self.scan_img_btn = QPushButton("Scan Image File...")
        self.scan_img_btn.setObjectName("SecondaryBtn")
        self.scan_img_btn.clicked.connect(self.scan_image_file)
        ctrl_row1.addWidget(self.scan_img_btn)
        ctrl_row1.addStretch()

        self.engine_badge = make_badge_label(self.engine.engine_label, "#1e3a8a", "#bfdbfe")
        ctrl_row1.addWidget(self.engine_badge)
        ctrl_outer.addLayout(ctrl_row1)

        ctrl_row2 = QHBoxLayout()
        ctrl_row2.setSpacing(10)
        ctrl_row2.addWidget(QLabel("Similarity Match Threshold:"))
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(20, 85)
        self.thresh_slider.setValue(int(round(self.engine.default_threshold * 100)))
        self.thresh_slider.setFixedWidth(180)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        ctrl_row2.addWidget(self.thresh_slider)

        self.thresh_lbl = QLabel(f"{self.engine.default_threshold:.2f}")
        self.thresh_lbl.setFixedWidth(40)
        self.thresh_lbl.setStyleSheet("font-weight: 700; color: #60a5fa;")
        ctrl_row2.addWidget(self.thresh_lbl)

        self.mirror_chk = QCheckBox("Mirror Camera Feed")
        self.mirror_chk.setChecked(True)
        ctrl_row2.addWidget(self.mirror_chk)
        ctrl_row2.addStretch()
        ctrl_outer.addLayout(ctrl_row2)

        layout.addWidget(ctrl)

        # Main Splitter: Video Feed (Left) | Identified Students Panel (Right)
        splitter = QSplitter(Qt.Horizontal)

        # Left Video Panel
        left_panel = QFrame()
        left_panel.setObjectName("Card")
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(12, 12, 12, 12)
        left_lay.setSpacing(10)

        self.video_label = QLabel(
            "Camera Idle\n\nClick '▶ Start Live Camera' for real-time face scanning,\n"
            "or 'Scan Image File...' to identify students in a photograph."
        )
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(540, 400)
        self.video_label.setStyleSheet(
            "background-color: #0b1220; border: 1px solid #334155; border-radius: 10px; "
            "color: #94a3b8; font-size: 14px;"
        )
        left_lay.addWidget(self.video_label, 1)

        self.status_banner = QLabel("READY — Waiting for face scan...")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setFixedHeight(46)
        self.status_banner.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; "
            "font-size: 15px; font-weight: 800; color: #cbd5e1;"
        )
        left_lay.addWidget(self.status_banner)
        splitter.addWidget(left_panel)

        # Right Identified Students Panel
        right_panel = QFrame()
        right_panel.setObjectName("Card")
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(12, 12, 12, 12)
        right_lay.setSpacing(10)

        rhdr = QLabel("Identified Students (Identity • Phone • Payment Date)")
        rhdr.setObjectName("SectionTitle")
        right_lay.addWidget(rhdr)

        self.ident_summary_lbl = QLabel("No faces currently detected.")
        self.ident_summary_lbl.setObjectName("MutedLabel")
        right_lay.addWidget(self.ident_summary_lbl)

        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_host = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 4, 0)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch()
        self.cards_scroll.setWidget(self.cards_host)
        right_lay.addWidget(self.cards_scroll, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([640, 460])
        layout.addWidget(splitter, 1)

    def set_context(self, admin_username: str, evaluation_mode: Any) -> None:
        self.admin_username = admin_username
        self.evaluation_mode = evaluation_mode
        self.reload_gallery()

    def reload_gallery(self) -> None:
        self.students_cache = self.db.get_all_students()
        self._last_card_signature = ""

    def _on_thresh_changed(self, value: int) -> None:
        self.thresh_lbl.setText(f"{value / 100.0:.2f}")

    @property
    def current_threshold(self) -> float:
        return self.thresh_slider.value() / 100.0

    def start_camera(self) -> None:
        self.reload_gallery()
        if self.cap is not None and self.cap.isOpened():
            return
        idx = self.cam_combo.currentIndex()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.warning(
                self,
                "Camera Unavailable",
                f"Could not open Camera {idx}.\n\n"
                "On macOS, ensure Terminal / Python has Camera permission in:\n"
                "System Settings > Privacy & Security > Camera.\n\n"
                "You can still use 'Scan Image File...' to test face identification.",
            )
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_banner.setText("LIVE SCANNING — Position student(s) in front of camera")
        self.status_banner.setStyleSheet(
            "background-color: #172554; border: 1px solid #3b82f6; border-radius: 8px; "
            "font-size: 15px; font-weight: 800; color: #93c5fd;"
        )
        self.timer.start(35)

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(
            "Camera Stopped\n\nClick '▶ Start Live Camera' or 'Scan Image File...' to scan."
        )
        self.status_banner.setText("CAMERA STOPPED")
        self.status_banner.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; "
            "font-size: 15px; font-weight: 800; color: #94a3b8;"
        )

    def _process_camera_frame(self) -> None:
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        if self.mirror_chk.isChecked():
            frame = cv2.flip(frame, 1)
        self._frame_counter += 1
        self._analyze_and_display_frame(frame, log_events=True)

    def scan_image_file(self) -> None:
        self.reload_gallery()
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Photo to Scan & Identify Students", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not fpath:
            return
        frame = cv2.imread(fpath)
        if frame is None:
            QMessageBox.warning(self, "Image Error", "Failed to read the selected image file.")
            return
        if self.cap is not None:
            self.stop_camera()
        self._last_card_signature = ""
        self._analyze_and_display_frame(frame, log_events=True, force_log=True)

    def _analyze_and_display_frame(
        self,
        frame_bgr: np.ndarray,
        log_events: bool = True,
        force_log: bool = False,
    ) -> List[IdentifiedFaceResult]:
        results = self.engine.identify_all_faces(
            frame_bgr,
            self.students_cache,
            threshold=self.current_threshold,
            evaluation_mode=self.evaluation_mode,
        )
        annotated = draw_scan_overlays(frame_bgr, results)
        pix = bgr_to_qpixmap(annotated)
        scaled = pix.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

        self._update_status_banner(results)
        self._update_identified_cards(results)
        if log_events and results:
            self._record_scan_logs(results, force_log=force_log)
        return results

    def _update_status_banner(self, results: List[IdentifiedFaceResult]) -> None:
        if not results:
            self.status_banner.setText("WAITING FOR FACE — No individual detected in frame")
            self.status_banner.setStyleSheet(
                "background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; "
                "font-size: 14px; font-weight: 700; color: #94a3b8;"
            )
            return

        granted = [r for r in results if r.recognized and r.access and r.access.granted]
        denied_tuition = [r for r in results if r.recognized and r.access and not r.access.granted]
        unknown = [r for r in results if not r.recognized]

        if granted and not denied_tuition and not unknown:
            names = ", ".join(r.student["full_name"] for r in granted if r.student)
            self.status_banner.setText(f"✓ ACCESS GRANTED — {names}")
            self.status_banner.setStyleSheet(
                "background-color: #064e3b; border: 2px solid #10b981; border-radius: 8px; "
                "font-size: 15px; font-weight: 800; color: #a7f3d0;"
            )
        elif denied_tuition and not unknown and not granted:
            names = ", ".join(r.student["full_name"] for r in denied_tuition if r.student)
            self.status_banner.setText(f"⚠ ACCESS DENIED (TUITION ARREARS) — {names}")
            self.status_banner.setStyleSheet(
                "background-color: #78350f; border: 2px solid #f59e0b; border-radius: 8px; "
                "font-size: 15px; font-weight: 800; color: #fde68a;"
            )
        elif unknown and not granted and not denied_tuition:
            self.status_banner.setText(f"✕ ACCESS DENIED — {len(unknown)} Unregistered Individual(s)")
            self.status_banner.setStyleSheet(
                "background-color: #7f1d1d; border: 2px solid #ef4444; border-radius: 8px; "
                "font-size: 15px; font-weight: 800; color: #fecaca;"
            )
        else:
            self.status_banner.setText(
                f"MULTI-FACE SCAN: {len(granted)} Granted  •  "
                f"{len(denied_tuition)} Tuition Due  •  {len(unknown)} Unregistered"
            )
            self.status_banner.setStyleSheet(
                "background-color: #1e3a8a; border: 2px solid #3b82f6; border-radius: 8px; "
                "font-size: 14px; font-weight: 800; color: #dbeafe;"
            )

    def _update_identified_cards(self, results: List[IdentifiedFaceResult]) -> None:
        # Build lightweight signature to avoid rebuilding Qt widgets 30 times/sec unnecessarily
        sig_parts = []
        for r in results:
            if r.recognized and r.student and r.access:
                sig_parts.append(
                    f"S:{r.student['id']}:{r.access.granted}:{int(r.confidence * 20)}"
                )
            else:
                sig_parts.append("U")
        signature = "|".join(sig_parts)
        if signature == self._last_card_signature:
            return
        self._last_card_signature = signature

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        if not results:
            self.ident_summary_lbl.setText("No faces currently detected.")
            return

        rec_cnt = sum(1 for r in results if r.recognized)
        self.ident_summary_lbl.setText(
            f"Detected {len(results)} face(s): {rec_cnt} registered student(s), "
            f"{len(results) - rec_cnt} unregistered."
        )
        for r in results:
            card = IdentifiedStudentCard(r)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _record_scan_logs(self, results: List[IdentifiedFaceResult], force_log: bool = False) -> None:
        now = time.time()
        logged_any = False
        for r in results:
            if r.recognized and r.student and r.access:
                key = f"st_{r.student['id']}_{r.access.granted}"
                status = "GRANTED" if r.access.granted else "DENIED_TUITION"
                month_name = r.access.evaluation_month_name
                reason = r.access.reason
            else:
                key = "unknown"
                status = "DENIED_UNKNOWN"
                dec = evaluate_tuition_access(0, "", evaluation_mode=self.evaluation_mode)
                month_name = dec.evaluation_month_name
                reason = "Face not matched in registered student database."

            last_t = self._last_logged_times.get(key, 0.0)
            if force_log or (now - last_t >= LOG_COOLDOWN_SECONDS):
                self._last_logged_times[key] = now
                self.db.log_access(
                    student=r.student,
                    evaluation_month_name=month_name,
                    confidence=r.confidence,
                    access_status=status,
                    reason=reason,
                    scanned_by=self.admin_username,
                )
                logged_any = True
        if logged_any:
            self.logs_updated.emit()


class RegisterStudentPage(QWidget):
    """
    Complete Student & Biometric Front-Face Enrollment Screen:
      - Left : Guided 5-sample webcam face capture OR Front Photo File upload + Front Portrait Preview
      - Right: Matricule, Full Name, DOB (exact age determination + declared age accuracy check),
               Phone Number, Field, Option, Year, School, Address, Amount Paid (FCFA),
               Date of Payment, and live Monthly Tuition Access Preview.
    """

    student_registered = pyqtSignal(dict)

    def __init__(
        self,
        db: DatabaseManager,
        engine: FaceRecognitionEngine,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.admin_username = DEFAULT_ADMIN_USER
        self.evaluation_mode: Any = "current"

        self.cap: Optional[cv2.VideoCapture] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_cam_tick)

        self.capturing = False
        self.captured_embeddings: List[np.ndarray] = []
        self.best_portrait_bgr: Optional[np.ndarray] = None
        self.best_portrait_quality: float = -1.0
        self._last_sample_time: float = 0.0

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(16)

        # ---- LEFT COLUMN: FACE BIOMETRIC & FRONT PHOTO CAPTURE ----
        left_card = QFrame()
        left_card.setObjectName("Card")
        left_card.setMinimumWidth(420)
        left_lay = QVBoxLayout(left_card)
        left_lay.setContentsMargins(14, 14, 14, 14)
        left_lay.setSpacing(10)

        lhdr = QLabel("Step 1 — Scan & Capture Student Front Face")
        lhdr.setObjectName("SectionTitle")
        left_lay.addWidget(lhdr)

        cam_ctrl = QHBoxLayout()
        cam_ctrl.setSpacing(8)
        self.cam_idx_combo = QComboBox()
        self.cam_idx_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.cam_idx_combo.setFixedWidth(105)
        cam_ctrl.addWidget(self.cam_idx_combo)

        self.open_cam_btn = QPushButton("Open Camera")
        self.open_cam_btn.clicked.connect(self.start_camera)
        cam_ctrl.addWidget(self.open_cam_btn, 1)

        self.close_cam_btn = QPushButton("Stop")
        self.close_cam_btn.setObjectName("SecondaryBtn")
        self.close_cam_btn.setEnabled(False)
        self.close_cam_btn.clicked.connect(self.stop_camera)
        cam_ctrl.addWidget(self.close_cam_btn)
        left_lay.addLayout(cam_ctrl)

        self.upload_btn = QPushButton("Upload Front Photo from File...")
        self.upload_btn.setObjectName("SecondaryBtn")
        self.upload_btn.clicked.connect(self.upload_front_photo)
        left_lay.addWidget(self.upload_btn)

        self.preview_video = QLabel(
            "Start the webcam to scan the student's face live,\n"
            "or click 'Upload Photo...' to enroll from a front portrait image."
        )
        self.preview_video.setAlignment(Qt.AlignCenter)
        self.preview_video.setMinimumSize(380, 260)
        self.preview_video.setStyleSheet(
            "background-color: #0b1220; border: 1px solid #334155; border-radius: 10px; color: #94a3b8;"
        )
        left_lay.addWidget(self.preview_video, 1)

        enroll_row = QHBoxLayout()
        self.capture_btn = QPushButton(f"Capture {NUM_ENROLL_SAMPLES} Face Samples")
        self.capture_btn.setObjectName("SuccessBtn")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self._start_sample_capture)
        enroll_row.addWidget(self.capture_btn, 1)

        self.clear_face_btn = QPushButton("Reset Face")
        self.clear_face_btn.setObjectName("SecondaryBtn")
        self.clear_face_btn.clicked.connect(self._reset_face_capture)
        enroll_row.addWidget(self.clear_face_btn)
        left_lay.addLayout(enroll_row)

        self.enroll_progress = QProgressBar()
        self.enroll_progress.setRange(0, NUM_ENROLL_SAMPLES)
        self.enroll_progress.setValue(0)
        self.enroll_progress.setFormat("Face Samples Captured: %v / %m")
        left_lay.addWidget(self.enroll_progress)

        # Captured Front Portrait Card
        portrait_box = QFrame()
        portrait_box.setStyleSheet(
            "background-color: #0b1220; border: 1px solid #334155; border-radius: 10px;"
        )
        pb_lay = QHBoxLayout(portrait_box)
        pb_lay.setContentsMargins(10, 10, 10, 10)
        pb_lay.setSpacing(12)

        self.front_photo_preview = QLabel()
        self.front_photo_preview.setFixedSize(104, 120)
        self.front_photo_preview.setPixmap(make_placeholder_portrait("NEW", 104, 120))
        pb_lay.addWidget(self.front_photo_preview)

        p_info = QVBoxLayout()
        self.face_status_title = QLabel("No Front Face Enrolled Yet")
        self.face_status_title.setStyleSheet("font-weight: 700; color: #f8fafc; background: transparent;")
        self.face_status_detail = QLabel(
            "Capture 5 frontal samples via webcam (recommended for best accuracy) "
            "or upload a clear frontal portrait image."
        )
        self.face_status_detail.setObjectName("MutedLabel")
        self.face_status_detail.setWordWrap(True)
        p_info.addWidget(self.face_status_title)
        p_info.addWidget(self.face_status_detail)
        p_info.addStretch()
        pb_lay.addLayout(p_info, 1)
        left_lay.addWidget(portrait_box)

        outer.addWidget(left_card, 4)

        # ---- RIGHT COLUMN: STUDENT PERSONAL, ACADEMIC & PAYMENT FORM ----
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_card = QFrame()
        right_card.setObjectName("Card")
        right_lay = QVBoxLayout(right_card)
        right_lay.setContentsMargins(18, 16, 18, 16)
        right_lay.setSpacing(12)
        right_scroll.setWidget(right_card)

        rhdr = QLabel("Step 2 — Student Personal, Academic & Tuition Details")
        rhdr.setObjectName("SectionTitle")
        right_lay.addWidget(rhdr)

        form = QFormLayout()
        form.setSpacing(9)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Matricule
        mat_row = QHBoxLayout()
        self.matricule_edit = QLineEdit()
        self.matricule_edit.setPlaceholderText("e.g. PFMS-2026-0001 or University ID")
        auto_mat_btn = QPushButton("Auto ID")
        auto_mat_btn.setObjectName("SecondaryBtn")
        auto_mat_btn.setFixedWidth(82)
        auto_mat_btn.clicked.connect(self._fill_auto_matricule)
        mat_row.addWidget(self.matricule_edit, 1)
        mat_row.addWidget(auto_mat_btn)
        form.addRow("Matricule *:", mat_row)

        # Student Full Name
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Full University Student Name")
        form.addRow("Student Name *:", self.name_edit)

        # Date of Birth + Age Accuracy Verification
        dob_row = QHBoxLayout()
        self.dob_edit = QDateEdit()
        self.dob_edit.setCalendarPopup(True)
        self.dob_edit.setDisplayFormat("yyyy-MM-dd")
        self.dob_edit.setDate(QDate(2004, 5, 15))
        self.dob_edit.dateChanged.connect(self._update_age_check)
        dob_row.addWidget(self.dob_edit, 2)

        dob_row.addWidget(QLabel("Declared Age (Verify):"))
        self.declared_age_spin = QSpinBox()
        self.declared_age_spin.setRange(0, 110)
        self.declared_age_spin.setValue(0)
        self.declared_age_spin.setSpecialValueText("Auto")
        self.declared_age_spin.setToolTip(
            "Optional: enter student's stated/ID-card age to cross-check against Date of Birth for extra accuracy."
        )
        self.declared_age_spin.valueChanged.connect(self._update_age_check)
        dob_row.addWidget(self.declared_age_spin, 1)
        form.addRow("Date of Birth *:", dob_row)

        self.age_banner = QLabel()
        self.age_banner.setWordWrap(True)
        self.age_banner.setMinimumHeight(44)
        self.age_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        form.addRow("Age Accuracy Check:", self.age_banner)

        # Phone Number
        self.phone_edit = QLineEdit()
        self.phone_edit.setPlaceholderText("e.g. +237 6XX XXX XXX")
        form.addRow("Phone Number *:", self.phone_edit)

        # Field & Option
        self.field_combo = QComboBox()
        self.field_combo.setEditable(True)
        self.field_combo.addItems(COMMON_FIELDS)
        form.addRow("Field *:", self.field_combo)

        self.option_combo = QComboBox()
        self.option_combo.setEditable(True)
        self.option_combo.addItems(COMMON_OPTIONS)
        form.addRow("Option *:", self.option_combo)

        # Year & School
        self.year_combo = QComboBox()
        self.year_combo.setEditable(True)
        self.year_combo.addItems(COMMON_YEARS)
        form.addRow("Year *:", self.year_combo)

        self.school_combo = QComboBox()
        self.school_combo.setEditable(True)
        self.school_combo.addItems(COMMON_SCHOOLS)
        form.addRow("School *:", self.school_combo)

        # Address
        self.address_edit = QLineEdit()
        self.address_edit.setPlaceholderText("Neighborhood, City, Street / Residence")
        form.addRow("Address *:", self.address_edit)

        # Tuition Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #334155;")
        right_lay.addLayout(form)
        right_lay.addWidget(sep)

        pay_hdr = QLabel("Step 3 — Tuition Payment (FCFA) & Access Qualification")
        pay_hdr.setObjectName("SectionTitle")
        right_lay.addWidget(pay_hdr)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(6)
        for item in TUITION_SCHEDULE:
            btn = QPushButton(f"{item['short']}: {item['cumulative']:,}")
            btn.setObjectName("SecondaryBtn")
            btn.setToolTip(item["rule"])
            btn.clicked.connect(lambda _, amt=item["cumulative"]: self.amount_spin.setValue(amt))
            quick_row.addWidget(btn)
        right_lay.addLayout(quick_row)

        pay_form = QFormLayout()
        pay_form.setSpacing(9)
        pay_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.amount_spin = QSpinBox()
        self.amount_spin.setRange(0, 5_000_000)
        self.amount_spin.setSingleStep(5_000)
        self.amount_spin.setValue(55_000)
        self.amount_spin.setSuffix(" FCFA")
        self.amount_spin.valueChanged.connect(self._update_tuition_preview)
        pay_form.addRow("Amount Paid (Cumulative) *:", self.amount_spin)

        self.payment_date_edit = QDateEdit()
        self.payment_date_edit.setCalendarPopup(True)
        self.payment_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.payment_date_edit.setDate(QDate.currentDate())
        self.payment_date_edit.dateChanged.connect(self._update_tuition_preview)
        pay_form.addRow("Date of Payment *:", self.payment_date_edit)

        self.payment_notes_edit = QLineEdit()
        self.payment_notes_edit.setPlaceholderText("Optional receipt reference / bank / Mobile Money ID")
        pay_form.addRow("Payment Notes:", self.payment_notes_edit)
        right_lay.addLayout(pay_form)

        self.tuition_preview = QLabel()
        self.tuition_preview.setWordWrap(True)
        right_lay.addWidget(self.tuition_preview)

        self.milestone_host = QHBoxLayout()
        right_lay.addLayout(self.milestone_host)

        # Submit Row
        submit_row = QHBoxLayout()
        submit_row.setSpacing(10)
        reset_btn = QPushButton("Clear Form")
        reset_btn.setObjectName("SecondaryBtn")
        reset_btn.clicked.connect(self.reset_form)

        self.register_btn = QPushButton("✓ Register Student & Save Biometric Face")
        self.register_btn.setObjectName("SuccessBtn")
        self.register_btn.setFixedHeight(42)
        self.register_btn.setStyleSheet("font-size: 14px; font-weight: 700;")
        self.register_btn.clicked.connect(self._submit_registration)

        submit_row.addWidget(reset_btn)
        submit_row.addWidget(self.register_btn, 1)
        right_lay.addLayout(submit_row)
        right_lay.addStretch()

        outer.addWidget(right_scroll, 6)

        self._fill_auto_matricule()
        self._update_age_check()
        self._update_tuition_preview()

    def set_context(self, admin_username: str, evaluation_mode: Any) -> None:
        self.admin_username = admin_username
        self.evaluation_mode = evaluation_mode
        self._update_tuition_preview()

    def _fill_auto_matricule(self) -> None:
        self.matricule_edit.setText(self.db.generate_next_matricule())

    def _update_age_check(self) -> AgeVerificationResult:
        dob_s = self.dob_edit.date().toString("yyyy-MM-dd")
        decl = self.declared_age_spin.value() or None
        res = calculate_and_verify_age(dob_s, declared_age=decl)
        styles = {
            "VERIFIED": ("#064e3b", "#a7f3d0"),
            "ACCURATE": ("#172554", "#bfdbfe"),
            "WARNING": ("#78350f", "#fde68a"),
            "MISMATCH": ("#7f1d1d", "#fecaca"),
            "INVALID": ("#7f1d1d", "#fecaca"),
        }
        bg, fg = styles.get(res.status, ("#1e293b", "#e2e8f0"))
        self.age_banner.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 7px 10px; border-radius: 7px; font-weight: 600;"
        )
        self.age_banner.setText(f"[{res.status}] {res.accuracy_note}")
        return res

    def _update_tuition_preview(self) -> None:
        amt = self.amount_spin.value()
        pdate = self.payment_date_edit.date().toString("yyyy-MM-dd")
        dec = evaluate_tuition_access(amt, pdate, evaluation_mode=self.evaluation_mode)
        bg = "#064e3b" if dec.granted else "#7f1d1d"
        fg = "#d1fae5" if dec.granted else "#fee2e2"
        self.tuition_preview.setStyleSheet(
            f"background-color: {bg}; color: {fg}; padding: 10px 12px; border-radius: 8px;"
        )
        self.tuition_preview.setText(
            f"<b>{dec.status_label}</b> ({dec.evaluation_month_name})  •  "
            f"Cleared: <b>{dec.highest_cleared}</b><br>"
            f"{dec.reason}<br>"
            f"<i>{dec.rule_description}</i>"
        )
        while self.milestone_host.count():
            w = self.milestone_host.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self.milestone_host.addWidget(make_milestone_strip(dec.milestones))

    # -- Camera & Face Enrollment Capture ------------------------------------

    def start_camera(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            return
        idx = self.cam_idx_combo.currentIndex()
        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.warning(
                self,
                "Camera Unavailable",
                f"Unable to open Camera {idx}.\n"
                "Check camera permissions or use 'Upload Photo...' to register from a portrait image.",
            )
            return
        self.open_cam_btn.setEnabled(False)
        self.close_cam_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.timer.start(35)

    def stop_camera(self) -> None:
        self.timer.stop()
        self.capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.open_cam_btn.setEnabled(True)
        self.close_cam_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.preview_video.setPixmap(QPixmap())
        self.preview_video.setText(
            "Camera Stopped.\nClick 'Open Camera' or 'Upload Photo...' to enroll a student face."
        )

    def _start_sample_capture(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Off", "Please open the camera first.")
            return
        self.captured_embeddings = []
        self.best_portrait_bgr = None
        self.best_portrait_quality = -1.0
        self.enroll_progress.setValue(0)
        self.capturing = True
        self._last_sample_time = 0.0
        self.face_status_title.setText("Capturing frontal face samples...")
        self.face_status_detail.setText("Keep looking forward with good lighting (hold steady).")

    def _reset_face_capture(self) -> None:
        self.capturing = False
        self.captured_embeddings = []
        self.best_portrait_bgr = None
        self.best_portrait_quality = -1.0
        self.enroll_progress.setValue(0)
        self.front_photo_preview.setPixmap(make_placeholder_portrait("NEW", 104, 120))
        self.face_status_title.setText("No Front Face Enrolled Yet")
        self.face_status_detail.setText("Capture 5 frontal samples via webcam or upload a front photo.")

    def _on_cam_tick(self) -> None:
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        frame = cv2.flip(frame, 1)
        faces = self.engine.detect_faces(frame)

        if self.capturing and len(faces) == 1:
            now = time.time()
            face = faces[0]
            if now - self._last_sample_time >= 0.35 and face.frontal_score >= 0.45:
                emb = self.engine.extract_embedding(frame, face)
                self.captured_embeddings.append(emb)
                self._last_sample_time = now

                quality = face.frontal_score * 100.0 + min(face.sharpness, 300.0) * 0.25 + face.confidence * 50.0
                if quality > self.best_portrait_quality:
                    self.best_portrait_quality = quality
                    self.best_portrait_bgr = self.engine.crop_front_portrait(frame, face)
                    self._refresh_front_preview()

                self.enroll_progress.setValue(len(self.captured_embeddings))
                if len(self.captured_embeddings) >= NUM_ENROLL_SAMPLES:
                    self.capturing = False
                    self.face_status_title.setText(
                        f"✓ {NUM_ENROLL_SAMPLES} Face Samples & Front Photo Ready!"
                    )
                    self.face_status_title.setStyleSheet(
                        "font-weight: 700; color: #34d399; background: transparent;"
                    )
                    self.face_status_detail.setText(
                        f"Biometric template generated via {self.engine.engine_label}. "
                        "Complete the form and click Register."
                    )

        annotated = draw_enrollment_guide(
            frame, faces, len(self.captured_embeddings), NUM_ENROLL_SAMPLES, self.capturing
        )
        pix = bgr_to_qpixmap(annotated).scaled(
            self.preview_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_video.setPixmap(pix)

    def upload_front_photo(self) -> None:
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Select Student Front Photo", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not fpath:
            return
        img = cv2.imread(fpath)
        if img is None:
            QMessageBox.warning(self, "Image Error", "Could not read the selected image file.")
            return
        faces = self.engine.detect_faces(img)
        if not faces:
            QMessageBox.warning(
                self,
                "No Face Detected",
                "No clear frontal face was detected in this image.\n"
                "Please choose a well-lit front-facing portrait photo.",
            )
            return

        if len(faces) > 1:
            QMessageBox.information(
                self,
                "Multiple Faces Found",
                "More than one face was detected. The largest foreground face will be used.",
            )
        face = faces[0]
        emb = self.engine.extract_embedding(img, face)
        self.captured_embeddings = [emb]
        self.best_portrait_bgr = self.engine.crop_front_portrait(img, face)
        self.best_portrait_quality = 100.0
        self.enroll_progress.setValue(NUM_ENROLL_SAMPLES)
        self._refresh_front_preview()

        if self.cap is None:
            annotated = draw_enrollment_guide(img, [face], NUM_ENROLL_SAMPLES, NUM_ENROLL_SAMPLES, False)
            pix = bgr_to_qpixmap(annotated).scaled(
                self.preview_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_video.setPixmap(pix)

        self.face_status_title.setText("✓ Front Photo Loaded & Face Encoded!")
        self.face_status_title.setStyleSheet("font-weight: 700; color: #34d399; background: transparent;")
        self.face_status_detail.setText(
            f"Source: {os.path.basename(fpath)}  •  Engine: {self.engine.engine_label}"
        )

    def _refresh_front_preview(self) -> None:
        if self.best_portrait_bgr is not None:
            pix = bgr_to_qpixmap(self.best_portrait_bgr).scaled(
                104, 120, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            self.front_photo_preview.setPixmap(pix)

    def reset_form(self) -> None:
        self._reset_face_capture()
        self._fill_auto_matricule()
        self.name_edit.clear()
        self.dob_edit.setDate(QDate(2004, 5, 15))
        self.declared_age_spin.setValue(0)
        self.phone_edit.clear()
        self.address_edit.clear()
        self.amount_spin.setValue(55_000)
        self.payment_date_edit.setDate(QDate.currentDate())
        self.payment_notes_edit.clear()
        self._update_age_check()
        self._update_tuition_preview()

    def _submit_registration(self) -> None:
        if not self.captured_embeddings or self.best_portrait_bgr is None:
            QMessageBox.warning(
                self,
                "Face Not Captured",
                "Please scan the student's face with the webcam or upload a front photo before registering.",
            )
            return

        matricule = self.matricule_edit.text().strip().upper()
        full_name = self.name_edit.text().strip()
        field = self.field_combo.currentText().strip()
        option = self.option_combo.currentText().strip()
        year = self.year_combo.currentText().strip()
        school = self.school_combo.currentText().strip()
        address = self.address_edit.text().strip()

        if not all([matricule, full_name, field, option, year, school, address]):
            QMessageBox.warning(
                self,
                "Missing Information",
                "Please fill in all required fields marked with (*).",
            )
            return

        phone_ok, phone_clean = validate_phone_number(self.phone_edit.text())
        if not phone_ok:
            QMessageBox.warning(self, "Invalid Phone Number", phone_clean)
            return

        age_res = self._update_age_check()
        if not age_res.is_valid:
            QMessageBox.warning(self, "Invalid Date of Birth", age_res.accuracy_note)
            return
        if age_res.status in ("MISMATCH", "WARNING"):
            reply = QMessageBox.question(
                self,
                "Age Verification Warning",
                f"{age_res.accuracy_note}\n\n"
                f"Proceed and save with DOB-determined age ({age_res.age_years} years)?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Save official front portrait photo
        ensure_data_directories()
        safe_mat = re.sub(r"[^A-Za-z0-9_-]", "_", matricule)
        photo_path = os.path.join(PHOTOS_DIR, f"{safe_mat}_front.jpg")
        cv2.imwrite(photo_path, self.best_portrait_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        amount_paid = self.amount_spin.value()
        payment_date = self.payment_date_edit.date().toString("yyyy-MM-dd")
        decl_age = self.declared_age_spin.value() or None

        ok, msg, student_id = self.db.register_student(
            matricule=matricule,
            full_name=full_name,
            date_of_birth=age_res.dob_str,
            age_at_registration=age_res.age_years,
            declared_age=decl_age,
            age_verification_status=age_res.status,
            field=field,
            option_name=option,
            academic_year=year,
            school=school,
            address=address,
            phone_number=phone_clean,
            amount_paid=amount_paid,
            payment_date=payment_date,
            front_photo_path=photo_path,
            embeddings=self.captured_embeddings,
            engine_type=self.engine.engine_type,
            recorded_by=self.admin_username,
            payment_notes=self.payment_notes_edit.text() or "Initial registration payment",
        )
        if not ok:
            QMessageBox.warning(self, "Registration Failed", msg)
            return

        dec = evaluate_tuition_access(amount_paid, payment_date, evaluation_mode=self.evaluation_mode)
        QMessageBox.information(
            self,
            "Student Registered Successfully",
            f"{msg}\n\n"
            f"Age Determined: {age_res.badge_text}\n"
            f"Amount Paid: {format_fcfa(amount_paid)} (Date: {payment_date})\n"
            f"Access Status ({dec.evaluation_month_name}): {dec.status_label}",
        )
        st = self.db.get_student_by_id(student_id) if student_id else None
        self.reset_form()
        if st:
            self.student_registered.emit(st)


# =============================================================================
# 11. APPLICATION PAGES (RESEARCH, PAYMENTS, ACCESS LOGS, SETTINGS)
# =============================================================================

class ResearchPage(QWidget):
    """
    Dedicated Student Research Screen:
      - Search by Matricule, Name, Field, or All
      - Clickable Registered Fields bar listing every field with its student count
      - When a Field is listed/selected (or on any search), displays EVERY matching
        student's registered front photo on the left and full student info beside it!
    """

    data_changed = pyqtSignal()

    def __init__(
        self,
        db: DatabaseManager,
        engine: FaceRecognitionEngine,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.admin_username = DEFAULT_ADMIN_USER
        self.evaluation_mode: Any = "current"
        self.selected_field_chip: str = "All Fields"
        self._current_results: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # Search & Filter Controls Card
        search_card = QFrame()
        search_card.setObjectName("Card")
        sc_lay = QVBoxLayout(search_card)
        sc_lay.setContentsMargins(14, 12, 14, 12)
        sc_lay.setSpacing(10)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.search_by_combo = QComboBox()
        self.search_by_combo.addItem("All (Matricule / Name / Field)", "all")
        self.search_by_combo.addItem("By Matricule", "matricule")
        self.search_by_combo.addItem("By Student Name", "name")
        self.search_by_combo.addItem("By Field", "field")
        self.search_by_combo.setFixedWidth(235)
        self.search_by_combo.currentIndexChanged.connect(self.run_search)
        top_row.addWidget(self.search_by_combo)

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Type Matricule, Student Name, or Field...")
        self.query_edit.textChanged.connect(self.run_search)
        top_row.addWidget(self.query_edit, 1)

        self.field_combo = QComboBox()
        self.field_combo.setFixedWidth(205)
        self.field_combo.currentIndexChanged.connect(self._on_field_combo_changed)
        top_row.addWidget(self.field_combo)

        self.status_combo = QComboBox()
        self.status_combo.addItems(["All Access Status", "Access Granted", "Access Denied"])
        self.status_combo.setFixedWidth(150)
        self.status_combo.currentIndexChanged.connect(self.run_search)
        top_row.addWidget(self.status_combo)

        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("SecondaryBtn")
        reset_btn.clicked.connect(self.reset_filters)
        top_row.addWidget(reset_btn)

        export_btn = QPushButton("Export CSV")
        export_btn.setObjectName("SecondaryBtn")
        export_btn.clicked.connect(self._export_csv)
        top_row.addWidget(export_btn)
        sc_lay.addLayout(top_row)

        # Registered Fields Quick-Select Strip (click any Field to list all its students with front photos + info!)
        fields_hdr_row = QHBoxLayout()
        flbl = QLabel("Registered Fields (click a field to list all its students with front photo & info):")
        flbl.setObjectName("MutedLabel")
        fields_hdr_row.addWidget(flbl)
        fields_hdr_row.addStretch()

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Cards: Photo + Info", "Compact Table View"])
        self.view_mode_combo.setFixedWidth(185)
        self.view_mode_combo.currentIndexChanged.connect(self._render_results)
        fields_hdr_row.addWidget(self.view_mode_combo)
        sc_lay.addLayout(fields_hdr_row)

        self.chips_scroll = QScrollArea()
        self.chips_scroll.setWidgetResizable(True)
        self.chips_scroll.setFixedHeight(44)
        self.chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.chips_host = QWidget()
        self.chips_layout = QHBoxLayout(self.chips_host)
        self.chips_layout.setContentsMargins(0, 2, 0, 2)
        self.chips_layout.setSpacing(8)
        self.chips_layout.addStretch()
        self.chips_scroll.setWidget(self.chips_host)
        sc_lay.addWidget(self.chips_scroll)

        layout.addWidget(search_card)

        # Result summary header
        self.summary_lbl = QLabel("Showing 0 students")
        self.summary_lbl.setStyleSheet("font-weight: 700; color: #93c5fd;")
        layout.addWidget(self.summary_lbl)

        # Stacked Result Area: Index 0 = Photo+Info Cards | Index 1 = Compact Table
        self.result_stack = QStackedWidget()

        # Cards scroll area
        self.cards_scroll = QScrollArea()
        self.cards_scroll.setWidgetResizable(True)
        self.cards_host = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_host)
        self.cards_layout.setContentsMargins(0, 0, 6, 0)
        self.cards_layout.setSpacing(12)
        self.cards_layout.addStretch()
        self.cards_scroll.setWidget(self.cards_host)
        self.result_stack.addWidget(self.cards_scroll)

        # Compact table
        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Matricule",
                "Student Name",
                "Field",
                "Option",
                "Year",
                "School",
                "DOB / Age",
                "Phone",
                "Amount Paid",
                "Payment Date",
                "Access",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_table_double_click)
        self.result_stack.addWidget(self.table)

        layout.addWidget(self.result_stack, 1)

    def set_context(self, admin_username: str, evaluation_mode: Any) -> None:
        self.admin_username = admin_username
        self.evaluation_mode = evaluation_mode
        self.refresh()

    def refresh(self) -> None:
        fields = self.db.get_fields_with_counts()

        # Refresh Field Combo without triggering duplicate searches
        self.field_combo.blockSignals(True)
        prev = self.selected_field_chip
        self.field_combo.clear()
        self.field_combo.addItem("All Fields", "All Fields")
        for fname, cnt in fields:
            self.field_combo.addItem(f"{fname} ({cnt})", fname)
        idx = self.field_combo.findData(prev)
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        else:
            self.selected_field_chip = "All Fields"
            self.field_combo.setCurrentIndex(0)
        self.field_combo.blockSignals(False)

        # Rebuild Field Chips
        while self.chips_layout.count() > 1:
            w = self.chips_layout.takeAt(0).widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        total_count = sum(c for _, c in fields)
        all_chip = self._make_field_chip(f"All Fields ({total_count})", "All Fields")
        self.chips_layout.insertWidget(self.chips_layout.count() - 1, all_chip)
        for fname, cnt in fields:
            chip = self._make_field_chip(f"{fname} ({cnt})", fname)
            self.chips_layout.insertWidget(self.chips_layout.count() - 1, chip)

        self.run_search()

    def _make_field_chip(self, label: str, field_value: str) -> QPushButton:
        btn = QPushButton(label.replace("&", "&&"))
        btn.setObjectName("FieldChip")
        btn.setProperty("selected", field_value.lower() == self.selected_field_chip.lower())
        btn.clicked.connect(lambda _, fv=field_value: self._select_field(fv))
        return btn

    def _select_field(self, field_value: str) -> None:
        self.selected_field_chip = field_value
        self.field_combo.blockSignals(True)
        idx = self.field_combo.findData(field_value)
        if idx >= 0:
            self.field_combo.setCurrentIndex(idx)
        self.field_combo.blockSignals(False)
        self.refresh()

    def _on_field_combo_changed(self) -> None:
        val = self.field_combo.currentData()
        if val:
            self.selected_field_chip = val
            self.refresh()

    def reset_filters(self) -> None:
        self.query_edit.blockSignals(True)
        self.query_edit.clear()
        self.query_edit.blockSignals(False)
        self.search_by_combo.setCurrentIndex(0)
        self.status_combo.setCurrentIndex(0)
        self.selected_field_chip = "All Fields"
        self.refresh()

    def run_search(self) -> None:
        query = self.query_edit.text().strip()
        search_by = self.search_by_combo.currentData() or "all"
        field_filter = self.selected_field_chip
        students = self.db.search_students(
            query=query,
            search_by=search_by,
            field_filter=field_filter,
        )

        status_idx = self.status_combo.currentIndex()
        filtered: List[Dict[str, Any]] = []
        for st in students:
            dec = evaluate_tuition_access(
                st["amount_paid"], st["payment_date"], evaluation_mode=self.evaluation_mode
            )
            if status_idx == 1 and not dec.granted:
                continue
            if status_idx == 2 and dec.granted:
                continue
            filtered.append(st)

        self._current_results = filtered
        scope = (
            f"in Field '{field_filter}'"
            if field_filter and field_filter != "All Fields"
            else "across All Fields"
        )
        q_note = f" matching '{query}'" if query else ""
        self.summary_lbl.setText(
            f"Found {len(filtered)} registered student(s) {scope}{q_note}"
        )
        self._render_results()

    def _render_results(self) -> None:
        mode = self.view_mode_combo.currentIndex()
        self.result_stack.setCurrentIndex(mode)

        if mode == 0:
            # Clear existing cards immediately (detach parent to prevent paint ghosts)
            while self.cards_layout.count() > 1:
                w = self.cards_layout.takeAt(0).widget()
                if w:
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()

            if not self._current_results:
                empty = QLabel(
                    "No registered students match your search criteria.\n"
                    "Try clearing filters or register a new student."
                )
                empty.setAlignment(Qt.AlignCenter)
                empty.setStyleSheet("color: #64748b; padding: 48px; font-size: 14px;")
                self.cards_layout.insertWidget(0, empty)
                return

            # Group header when viewing multiple fields, or direct list when one field is selected
            last_field: Optional[str] = None
            multi_field = len({s["field"].lower() for s in self._current_results}) > 1
            for st in self._current_results:
                if multi_field and st["field"].lower() != (last_field or "").lower():
                    last_field = st["field"]
                    cnt = sum(1 for x in self._current_results if x["field"].lower() == last_field.lower())
                    grp_hdr = QLabel(f"FIELD: {last_field}  ({cnt} student{'s' if cnt != 1 else ''})")
                    grp_hdr.setStyleSheet(
                        "background-color: #172554; color: #93c5fd; padding: 6px 12px; "
                        "border-radius: 6px; font-weight: 800; font-size: 12px;"
                    )
                    self.cards_layout.insertWidget(self.cards_layout.count() - 1, grp_hdr)

                card = StudentResearchCard(st, evaluation_mode=self.evaluation_mode)
                card.payment_requested.connect(self._open_add_payment)
                card.detail_requested.connect(self._open_detail)
                card.edit_requested.connect(self._open_edit)
                card.delete_requested.connect(self._confirm_delete)
                self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        else:
            self.table.setRowCount(len(self._current_results))
            for r, st in enumerate(self._current_results):
                age_res = calculate_and_verify_age(st["date_of_birth"], st.get("declared_age"))
                dec = evaluate_tuition_access(
                    st["amount_paid"], st["payment_date"], evaluation_mode=self.evaluation_mode
                )
                self.table.setItem(r, 0, QTableWidgetItem(st["matricule"]))
                self.table.setItem(r, 1, QTableWidgetItem(st["full_name"]))
                self.table.setItem(r, 2, QTableWidgetItem(st["field"]))
                self.table.setItem(r, 3, QTableWidgetItem(st["option_name"]))
                self.table.setItem(r, 4, QTableWidgetItem(st["academic_year"]))
                self.table.setItem(r, 5, QTableWidgetItem(st["school"]))
                self.table.setItem(r, 6, QTableWidgetItem(f"{st['date_of_birth']} ({age_res.age_years}y)"))
                self.table.setItem(r, 7, QTableWidgetItem(st["phone_number"]))
                self.table.setItem(r, 8, QTableWidgetItem(format_fcfa(st["amount_paid"])))
                self.table.setItem(r, 9, QTableWidgetItem(st["payment_date"]))
                acc_item = QTableWidgetItem(dec.status_label)
                acc_item.setForeground(QColor("#34d399" if dec.granted else "#f87171"))
                self.table.setItem(r, 10, acc_item)

    def _on_table_double_click(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._current_results):
            self._open_detail(self._current_results[row])

    def _open_add_payment(self, student: Dict[str, Any]) -> None:
        dlg = AddPaymentDialog(self.db, student, self.admin_username, self.evaluation_mode, self)
        if dlg.exec_() == QDialog.Accepted:
            self.refresh()
            self.data_changed.emit()

    def _open_detail(self, student: Dict[str, Any]) -> None:
        fresh = self.db.get_student_by_id(student["id"]) or student
        StudentDetailDialog(self.db, fresh, self.evaluation_mode, self).exec_()

    def _open_edit(self, student: Dict[str, Any]) -> None:
        fresh = self.db.get_student_by_id(student["id"]) or student
        dlg = EditStudentDialog(self.db, self.engine, fresh, self.admin_username, self)
        if dlg.exec_() == QDialog.Accepted:
            self.refresh()
            self.data_changed.emit()

    def _confirm_delete(self, student: Dict[str, Any]) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Registered Student",
            f"Permanently delete student '{student['full_name']}' ({student['matricule']}) "
            "and their biometric face data?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            ok, msg = self.db.delete_student(student["id"], self.admin_username)
            if ok:
                self.refresh()
                self.data_changed.emit()
            else:
                QMessageBox.warning(self, "Error", msg)

    def _export_csv(self) -> None:
        if not self._current_results:
            QMessageBox.information(self, "Nothing to Export", "No students in current search results.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Export Students CSV", "pondi_students_export.csv", "CSV Files (*.csv)"
        )
        if not fpath:
            return
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Matricule", "Full Name", "Date of Birth", "Age", "Age Status",
                    "Field", "Option", "Year", "School", "Address", "Phone Number",
                    "Amount Paid (FCFA)", "Date of Payment", "Access Status",
                    "Evaluation Month", "Front Photo Path",
                ]
            )
            for st in self._current_results:
                age_res = calculate_and_verify_age(st["date_of_birth"], st.get("declared_age"))
                dec = evaluate_tuition_access(
                    st["amount_paid"], st["payment_date"], evaluation_mode=self.evaluation_mode
                )
                writer.writerow(
                    [
                        st["matricule"], st["full_name"], st["date_of_birth"],
                        age_res.age_years, age_res.status,
                        st["field"], st["option_name"], st["academic_year"], st["school"],
                        st["address"], st["phone_number"], st["amount_paid"],
                        st["payment_date"], dec.status_label, dec.evaluation_month_name,
                        st["front_photo_path"],
                    ]
                )
        QMessageBox.information(self, "Export Complete", f"Exported {len(self._current_results)} student(s) to:\n{fpath}")


class PaymentsPage(QWidget):
    """Tuition Management & Monthly Tranche Tracker (Sep 55k / Oct 305k / Nov 445k / Dec 485k)."""

    data_changed = pyqtSignal()

    def __init__(self, db: DatabaseManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.db = db
        self.admin_username = DEFAULT_ADMIN_USER
        self.evaluation_mode: Any = "current"
        self._rows_data: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        # Rules Banner
        rules_card = QFrame()
        rules_card.setObjectName("Card")
        rl = QHBoxLayout(rules_card)
        rl.setContentsMargins(14, 10, 14, 10)
        rl.setSpacing(12)
        for item in TUITION_SCHEDULE:
            box = QLabel(
                f"<b>{item['name']}</b><br>"
                f"<span style='color:#93c5fd;'>+{item['tranche']:,}</span> → "
                f"<b style='color:#34d399;'>{item['cumulative']:,} FCFA</b>"
            )
            box.setAlignment(Qt.AlignCenter)
            box.setStyleSheet(
                "background-color: #0b1220; border: 1px solid #334155; border-radius: 8px; padding: 8px;"
            )
            rl.addWidget(box, 1)
        layout.addWidget(rules_card)

        # Filter bar
        bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter by matricule, name, field, or phone...")
        self.search_edit.textChanged.connect(self.refresh)
        bar.addWidget(self.search_edit, 1)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Students", "Access Granted", "Tuition Due / Arrears"])
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.filter_combo)

        self.add_pay_btn = QPushButton("+ Record Payment for Selected")
        self.add_pay_btn.setObjectName("SuccessBtn")
        self.add_pay_btn.clicked.connect(self._record_for_selected)
        bar.addWidget(self.add_pay_btn)

        export_btn = QPushButton("Export Tuition Report CSV")
        export_btn.setObjectName("SecondaryBtn")
        export_btn.clicked.connect(self._export_csv)
        bar.addWidget(export_btn)
        layout.addLayout(bar)

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet("color: #cbd5e1; font-weight: 600;")
        layout.addWidget(self.summary_lbl)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Matricule",
                "Student Name",
                "Field",
                "Phone",
                "Paid",
                "Pay Date",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
                "Access & Balance",
            ]
        )
        ph = self.table.horizontalHeader()
        ph.setSectionResizeMode(QHeaderView.Interactive)
        col_widths = [115, 165, 130, 115, 95, 88, 56, 56, 56, 56]
        for ci, cw in enumerate(col_widths):
            self.table.setColumnWidth(ci, cw)
        ph.setSectionResizeMode(10, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open_payment_row(r))
        layout.addWidget(self.table, 1)

    def set_context(self, admin_username: str, evaluation_mode: Any) -> None:
        self.admin_username = admin_username
        self.evaluation_mode = evaluation_mode
        self.refresh()

    def refresh(self) -> None:
        q = self.search_edit.text().strip()
        students = self.db.search_students(query=q, search_by="all")
        f_idx = self.filter_combo.currentIndex()

        rows: List[Tuple[Dict[str, Any], AccessDecision]] = []
        total_paid = 0
        total_due = 0
        for st in students:
            dec = evaluate_tuition_access(
                st["amount_paid"], st["payment_date"], evaluation_mode=self.evaluation_mode
            )
            if f_idx == 1 and not dec.granted:
                continue
            if f_idx == 2 and dec.granted:
                continue
            total_paid += int(st["amount_paid"])
            total_due += dec.shortfall
            rows.append((st, dec))

        self._rows_data = [s for s, _ in rows]
        self.summary_lbl.setText(
            f"Showing {len(rows)} student(s)  •  Total Paid: {format_fcfa(total_paid)}  •  "
            f"Active Month Arrears: {format_fcfa(total_due)}  "
            "(Double-click any row to add a payment installment)"
        )

        self.table.setRowCount(len(rows))
        for r, (st, dec) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(st["matricule"]))
            self.table.setItem(r, 1, QTableWidgetItem(st["full_name"]))
            self.table.setItem(r, 2, QTableWidgetItem(st["field"]))
            self.table.setItem(r, 3, QTableWidgetItem(st["phone_number"]))
            self.table.setItem(r, 4, QTableWidgetItem(format_fcfa(st["amount_paid"])))
            self.table.setItem(r, 5, QTableWidgetItem(st["payment_date"]))
            for ci, key in enumerate(("Sep", "Oct", "Nov", "Dec"), start=6):
                ok = dec.milestones.get(key, False)
                item = QTableWidgetItem("✓ Paid" if ok else "✕ Due")
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QColor("#34d399" if ok else "#94a3b8"))
                self.table.setItem(r, ci, item)
            status_txt = (
                f"GRANTED ({dec.evaluation_month_name[:3]})"
                if dec.granted
                else f"DENIED — Due {format_fcfa(dec.shortfall)}"
            )
            s_item = QTableWidgetItem(status_txt)
            s_item.setForeground(QColor("#34d399" if dec.granted else "#f87171"))
            self.table.setItem(r, 10, s_item)

    def _record_for_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows_data):
            QMessageBox.information(self, "Select Student", "Please click a student row first.")
            return
        self._open_payment_row(row)

    def _open_payment_row(self, row: int) -> None:
        if 0 <= row < len(self._rows_data):
            st = self.db.get_student_by_id(self._rows_data[row]["id"])
            if st:
                dlg = AddPaymentDialog(self.db, st, self.admin_username, self.evaluation_mode, self)
                if dlg.exec_() == QDialog.Accepted:
                    self.refresh()
                    self.data_changed.emit()

    def _export_csv(self) -> None:
        if not self._rows_data:
            QMessageBox.information(self, "No Data", "No tuition records to export.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Export Tuition Report", "pondi_tuition_report.csv", "CSV Files (*.csv)"
        )
        if not fpath:
            return
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Matricule", "Student Name", "Field", "Phone", "Amount Paid (FCFA)",
                    "Date of Payment", "Sep Cleared", "Oct Cleared", "Nov Cleared", "Dec Cleared",
                    "Access Status", "Shortfall (FCFA)", "Remaining Annual (FCFA)",
                ]
            )
            for st in self._rows_data:
                dec = evaluate_tuition_access(
                    st["amount_paid"], st["payment_date"], evaluation_mode=self.evaluation_mode
                )
                writer.writerow(
                    [
                        st["matricule"], st["full_name"], st["field"], st["phone_number"],
                        st["amount_paid"], st["payment_date"],
                        dec.milestones["Sep"], dec.milestones["Oct"],
                        dec.milestones["Nov"], dec.milestones["Dec"],
                        dec.status_label, dec.shortfall, dec.remaining_annual,
                    ]
                )
        QMessageBox.information(self, "Export Complete", f"Saved tuition report to:\n{fpath}")


class AccessLogsPage(QWidget):
    """Timestamped Audit Log of all Face Recognition Access Scans."""

    def __init__(self, db: DatabaseManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.db = db
        self.admin_username = DEFAULT_ADMIN_USER
        self._logs_cache: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search logs by name, matricule, phone, or reason...")
        self.search_edit.textChanged.connect(self.refresh)
        bar.addWidget(self.search_edit, 1)

        self.status_combo = QComboBox()
        self.status_combo.addItem("All Events", "ALL")
        self.status_combo.addItem("Access Granted", "GRANTED")
        self.status_combo.addItem("Denied (Tuition Due)", "DENIED_TUITION")
        self.status_combo.addItem("Denied (Unknown Face)", "DENIED_UNKNOWN")
        self.status_combo.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.status_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("SecondaryBtn")
        refresh_btn.clicked.connect(self.refresh)
        bar.addWidget(refresh_btn)

        export_btn = QPushButton("Export CSV")
        export_btn.setObjectName("SecondaryBtn")
        export_btn.clicked.connect(self._export_csv)
        bar.addWidget(export_btn)

        clear_btn = QPushButton("Clear Logs")
        clear_btn.setObjectName("DangerBtn")
        clear_btn.clicked.connect(self._clear_logs)
        bar.addWidget(clear_btn)
        layout.addLayout(bar)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Timestamp",
                "Student Name",
                "Matricule",
                "Phone",
                "Pay Date",
                "Paid",
                "Month",
                "Match",
                "Decision",
                "Reason",
            ]
        )
        lh = self.table.horizontalHeader()
        lh.setSectionResizeMode(QHeaderView.Interactive)
        l_widths = [135, 145, 112, 112, 86, 86, 74, 58, 110]
        for ci, cw in enumerate(l_widths):
            self.table.setColumnWidth(ci, cw)
        lh.setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

    def set_admin(self, username: str) -> None:
        self.admin_username = username

    def refresh(self) -> None:
        status = self.status_combo.currentData() or "ALL"
        query = self.search_edit.text().strip()
        logs = self.db.get_access_logs(limit=400, status_filter=status, query=query)
        self._logs_cache = logs
        self.table.setRowCount(len(logs))
        for r, row in enumerate(logs):
            self.table.setItem(r, 0, QTableWidgetItem(row["timestamp"]))
            self.table.setItem(r, 1, QTableWidgetItem(row["student_name"]))
            self.table.setItem(r, 2, QTableWidgetItem(row.get("matricule") or ""))
            self.table.setItem(r, 3, QTableWidgetItem(row.get("phone_number") or ""))
            self.table.setItem(r, 4, QTableWidgetItem(row.get("payment_date") or ""))
            self.table.setItem(r, 5, QTableWidgetItem(format_fcfa(row.get("amount_paid") or 0)))
            self.table.setItem(r, 6, QTableWidgetItem(row["evaluation_month"]))
            self.table.setItem(r, 7, QTableWidgetItem(f"{row['confidence'] * 100:.1f}%"))
            d_item = QTableWidgetItem(row["access_status"])
            if row["access_status"] == "GRANTED":
                d_item.setForeground(QColor("#34d399"))
            elif row["access_status"] == "DENIED_TUITION":
                d_item.setForeground(QColor("#fbbf24"))
            else:
                d_item.setForeground(QColor("#f87171"))
            self.table.setItem(r, 8, d_item)
            self.table.setItem(r, 9, QTableWidgetItem(row["reason"]))

    def _clear_logs(self) -> None:
        reply = QMessageBox.question(
            self, "Clear Access Logs", "Delete all recorded face scan logs?", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.db.clear_access_logs(self.admin_username)
            self.refresh()

    def _export_csv(self) -> None:
        if not self._logs_cache:
            QMessageBox.information(self, "No Logs", "No logs available to export.")
            return
        fpath, _ = QFileDialog.getSaveFileName(
            self, "Export Access Logs", "pondi_access_logs.csv", "CSV Files (*.csv)"
        )
        if not fpath:
            return
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp", "Student Name", "Matricule", "Phone Number", "Payment Date",
                    "Amount Paid (FCFA)", "Evaluation Month", "Confidence", "Decision", "Reason", "Scanned By",
                ]
            )
            for row in self._logs_cache:
                writer.writerow(
                    [
                        row["timestamp"], row["student_name"], row.get("matricule", ""),
                        row.get("phone_number", ""), row.get("payment_date", ""),
                        row.get("amount_paid", 0), row["evaluation_month"],
                        f"{row['confidence']:.4f}", row["access_status"],
                        row["reason"], row["scanned_by"],
                    ]
                )
        QMessageBox.information(self, "Export Complete", f"Saved {len(self._logs_cache)} logs to:\n{fpath}")


class SettingsPage(QWidget):
    """Admin Account Security, Password Management, Biometric Engine Status & Database Backup."""

    def __init__(
        self,
        db: DatabaseManager,
        engine: FaceRecognitionEngine,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.db = db
        self.engine = engine
        self.current_admin: Dict[str, Any] = {"username": DEFAULT_ADMIN_USER}

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        host = QWidget()
        layout = QVBoxLayout(host)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(16)
        scroll.setWidget(host)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # ---- 1. Change Current Admin Password ----
        pw_card = QFrame()
        pw_card.setObjectName("Card")
        pw_lay = QVBoxLayout(pw_card)
        pw_lay.setContentsMargins(16, 14, 16, 14)
        pw_lay.setSpacing(10)

        pw_hdr = QLabel("Change My Administrator Password")
        pw_hdr.setObjectName("SectionTitle")
        pw_lay.addWidget(pw_hdr)

        pw_form = QFormLayout()
        self.old_pw_edit = QLineEdit()
        self.old_pw_edit.setEchoMode(QLineEdit.Password)
        self.new_pw_edit = QLineEdit()
        self.new_pw_edit.setEchoMode(QLineEdit.Password)
        self.confirm_pw_edit = QLineEdit()
        self.confirm_pw_edit.setEchoMode(QLineEdit.Password)
        pw_form.addRow("Current Password:", self.old_pw_edit)
        pw_form.addRow("New Password (min 6 chars):", self.new_pw_edit)
        pw_form.addRow("Confirm New Password:", self.confirm_pw_edit)
        pw_lay.addLayout(pw_form)

        change_pw_btn = QPushButton("Update Password")
        change_pw_btn.setObjectName("SuccessBtn")
        change_pw_btn.setFixedWidth(180)
        change_pw_btn.clicked.connect(self._change_password)
        pw_lay.addWidget(change_pw_btn)
        layout.addWidget(pw_card)

        # ---- 2. Admin Accounts Management ----
        adm_card = QFrame()
        adm_card.setObjectName("Card")
        adm_lay = QVBoxLayout(adm_card)
        adm_lay.setContentsMargins(16, 14, 16, 14)
        adm_lay.setSpacing(10)

        adm_hdr = QLabel("Administrator Accounts")
        adm_hdr.setObjectName("SectionTitle")
        adm_lay.addWidget(adm_hdr)

        self.admins_table = QTableWidget(0, 5)
        self.admins_table.setHorizontalHeaderLabels(
            ["Username", "Full Name", "Role", "Created", "Last Login"]
        )
        self.admins_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.admins_table.verticalHeader().setVisible(False)
        self.admins_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.admins_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.admins_table.setFixedHeight(160)
        adm_lay.addWidget(self.admins_table)

        add_row = QHBoxLayout()
        self.new_adm_user = QLineEdit()
        self.new_adm_user.setPlaceholderText("New Username")
        self.new_adm_name = QLineEdit()
        self.new_adm_name.setPlaceholderText("Full Name")
        self.new_adm_role = QComboBox()
        self.new_adm_role.addItems(["Registrar Admin", "Security Officer", "Finance Admin", "Super Admin"])
        self.new_adm_pw = QLineEdit()
        self.new_adm_pw.setPlaceholderText("Password")
        self.new_adm_pw.setEchoMode(QLineEdit.Password)

        add_adm_btn = QPushButton("+ Add Admin")
        add_adm_btn.clicked.connect(self._add_admin)
        del_adm_btn = QPushButton("Delete Selected")
        del_adm_btn.setObjectName("DangerBtn")
        del_adm_btn.clicked.connect(self._delete_selected_admin)

        add_row.addWidget(self.new_adm_user)
        add_row.addWidget(self.new_adm_name)
        add_row.addWidget(self.new_adm_role)
        add_row.addWidget(self.new_adm_pw)
        add_row.addWidget(add_adm_btn)
        add_row.addWidget(del_adm_btn)
        adm_lay.addLayout(add_row)
        layout.addWidget(adm_card)

        # ---- 3. Biometric Engine & Database Maintenance ----
        sys_card = QFrame()
        sys_card.setObjectName("Card")
        sys_lay = QVBoxLayout(sys_card)
        sys_lay.setContentsMargins(16, 14, 16, 14)
        sys_lay.setSpacing(10)

        sys_hdr = QLabel("Biometric Engine & System Storage")
        sys_hdr.setObjectName("SectionTitle")
        sys_lay.addWidget(sys_hdr)

        self.sys_info_lbl = QLabel()
        self.sys_info_lbl.setWordWrap(True)
        self.sys_info_lbl.setStyleSheet(
            "background: #0b1220; padding: 12px; border-radius: 8px; color: #cbd5e1;"
        )
        sys_lay.addWidget(self.sys_info_lbl)

        btn_row = QHBoxLayout()
        self.dl_models_btn = QPushButton("Download / Verify YuNet & SFace ONNX Models")
        self.dl_models_btn.setObjectName("SecondaryBtn")
        self.dl_models_btn.clicked.connect(self._download_models)
        btn_row.addWidget(self.dl_models_btn)

        backup_btn = QPushButton("Backup SQLite Database...")
        backup_btn.clicked.connect(self._backup_db)
        btn_row.addWidget(backup_btn)
        btn_row.addStretch()
        sys_lay.addLayout(btn_row)
        layout.addWidget(sys_card)
        layout.addStretch()

        self._admins_cache: List[Dict[str, Any]] = []

    def set_admin(self, admin: Dict[str, Any]) -> None:
        self.current_admin = admin
        self.refresh()

    def refresh(self) -> None:
        admins = self.db.list_admins()
        self._admins_cache = admins
        self.admins_table.setRowCount(len(admins))
        for r, a in enumerate(admins):
            self.admins_table.setItem(r, 0, QTableWidgetItem(a["username"]))
            self.admins_table.setItem(r, 1, QTableWidgetItem(a["full_name"]))
            self.admins_table.setItem(r, 2, QTableWidgetItem(a["role"]))
            self.admins_table.setItem(r, 3, QTableWidgetItem(a["created_at"]))
            self.table_login = a.get("last_login") or "Never"
            self.admins_table.setItem(r, 4, QTableWidgetItem(self.table_login))

        self.sys_info_lbl.setText(
            f"<b>Active Face Engine:</b> {self.engine.engine_label}<br>"
            f"<b>YuNet Detector Path:</b> {self.engine.yunet_path or 'Not found (using Haar fallback)'}<br>"
            f"<b>SFace Recognizer Path:</b> {self.engine.sface_path or 'Not found (using descriptor fallback)'}<br>"
            f"<b>Database Path:</b> {self.db.db_path}<br>"
            f"<b>Front Photos Folder:</b> {PHOTOS_DIR}"
        )

    def _change_password(self) -> None:
        old_pw = self.old_pw_edit.text()
        new_pw = self.new_pw_edit.text()
        conf_pw = self.confirm_pw_edit.text()
        if new_pw != conf_pw:
            QMessageBox.warning(self, "Mismatch", "New password and confirmation do not match.")
            return
        ok, msg = self.db.change_admin_password(self.current_admin["username"], old_pw, new_pw)
        if ok:
            self.old_pw_edit.clear()
            self.new_pw_edit.clear()
            self.confirm_pw_edit.clear()
            QMessageBox.information(self, "Password Updated", msg)
        else:
            QMessageBox.warning(self, "Error", msg)

    def _add_admin(self) -> None:
        ok, msg = self.db.create_admin(
            self.new_adm_user.text(),
            self.new_adm_name.text(),
            self.new_adm_pw.text(),
            self.new_adm_role.currentText(),
            created_by=self.current_admin["username"],
        )
        if ok:
            self.new_adm_user.clear()
            self.new_adm_name.clear()
            self.new_adm_pw.clear()
            self.refresh()
            QMessageBox.information(self, "Admin Created", msg)
        else:
            QMessageBox.warning(self, "Error", msg)

    def _delete_selected_admin(self) -> None:
        row = self.admins_table.currentRow()
        if row < 0 or row >= len(self._admins_cache):
            QMessageBox.information(self, "Select Admin", "Select an admin row to delete.")
            return
        adm = self._admins_cache[row]
        reply = QMessageBox.question(
            self, "Delete Admin", f"Delete admin account '{adm['username']}'?", QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            ok, msg = self.db.delete_admin(adm["id"], self.current_admin["username"])
            if ok:
                self.refresh()
                QMessageBox.information(self, "Deleted", msg)
            else:
                QMessageBox.warning(self, "Cannot Delete", msg)

    def _download_models(self) -> None:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, msg = self.engine.download_onnx_models()
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh()
        if ok:
            QMessageBox.information(self, "Models Ready", msg)
        else:
            QMessageBox.warning(self, "Model Download", msg)

    def _backup_db(self) -> None:
        default_name = f"pondi_students_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        dest, _ = QFileDialog.getSaveFileName(self, "Save Database Backup", default_name, "SQLite DB (*.db)")
        if not dest:
            return
        try:
            shutil.copy2(self.db.db_path, dest)
            QMessageBox.information(self, "Backup Saved", f"Database successfully backed up to:\n{dest}")
        except OSError as exc:
            QMessageBox.warning(self, "Backup Error", f"Failed to create backup: {exc}")


# =============================================================================
# 12. MAIN APPLICATION WINDOW SHELL
# =============================================================================

class PondiFacialManagementApp(QMainWindow):
    """Main Window hosting Admin Login Screen and full Sidebar Management Console."""

    PAGE_META = [
        ("dashboard", "📊  Dashboard", "Dashboard Overview"),
        ("scanner", "🎥  Face Scanner", "Live Face Recognition & Access Control"),
        ("register", "📝  Register Student", "Register Student Face & Personal Information"),
        ("research", "🔍  Student Research", "Student Research (Matricule / Name / Field)"),
        ("payments", "💰  Tuition & Payments", "University Tuition & FCFA Payment Ledger"),
        ("logs", "📋  Access Logs", "Biometric Access Scan Audit Logs"),
        ("settings", "⚙️  Admin Settings", "Administrator Accounts & System Settings"),
    ]

    def __init__(self, db_path: str = DB_FILE):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} (v{APP_VERSION})")
        self.resize(1340, 840)
        self.setMinimumSize(1080, 700)

        self.db = DatabaseManager(db_path=db_path)
        self.engine = FaceRecognitionEngine()
        self.current_admin: Optional[Dict[str, Any]] = None
        self.evaluation_mode: Any = "current"

        self.root_stack = QStackedWidget()
        self.setCentralWidget(self.root_stack)

        # Stack 0: Admin Login Screen
        self.login_screen = AdminLoginScreen(self.db, self.engine)
        self.login_screen.login_succeeded.connect(self._on_login_success)
        self.root_stack.addWidget(self.login_screen)

        # Stack 1: Authenticated Management Console
        self.console_widget = QWidget()
        self._build_console()
        self.root_stack.addWidget(self.console_widget)

        self.root_stack.setCurrentIndex(0)

    def _build_console(self) -> None:
        outer = QHBoxLayout(self.console_widget)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Left Sidebar ----
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(235)
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(14, 18, 14, 16)
        sb_lay.setSpacing(6)

        brand_title = QLabel("PONDI FMS")
        brand_title.setStyleSheet("font-size: 20px; font-weight: 900; color: #60a5fa; background: transparent;")
        brand_sub = QLabel("University Facial Access System")
        brand_sub.setObjectName("MutedLabel")
        sb_lay.addWidget(brand_title)
        sb_lay.addWidget(brand_sub)
        sb_lay.addSpacing(14)

        self.nav_buttons: Dict[str, QPushButton] = {}
        for key, label, _title in self.PAGE_META:
            btn = QPushButton(label)
            btn.setObjectName("NavBtn")
            btn.setProperty("active", False)
            btn.clicked.connect(lambda _, k=key: self.switch_page(k))
            sb_lay.addWidget(btn)
            self.nav_buttons[key] = btn

        sb_lay.addStretch()

        # Engine & Tuition quick note in sidebar footer
        foot = QLabel(
            "Tuition Milestones (FCFA)\n"
            "• Sep: 55,000\n"
            "• Oct: +250k (305,000)\n"
            "• Nov: +140k (445,000)\n"
            "• Dec: +40k (485,000)"
        )
        foot.setStyleSheet(
            "background-color: #111827; color: #94a3b8; border: 1px solid #1e293b; "
            "border-radius: 8px; padding: 10px; font-size: 11px;"
        )
        sb_lay.addWidget(foot)

        logout_btn = QPushButton("🔒  Logout / Lock")
        logout_btn.setObjectName("DangerBtn")
        logout_btn.clicked.connect(self._logout)
        sb_lay.addWidget(logout_btn)

        outer.addWidget(sidebar)

        # ---- Right Main Area (TopBar + Pages Stack) ----
        right_col = QWidget()
        right_lay = QVBoxLayout(right_col)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(60)
        tb_lay = QHBoxLayout(topbar)
        tb_lay.setContentsMargins(20, 0, 20, 0)
        tb_lay.setSpacing(12)

        self.page_title_lbl = QLabel("Dashboard Overview")
        self.page_title_lbl.setObjectName("PageTitle")
        tb_lay.addWidget(self.page_title_lbl)
        tb_lay.addStretch()

        eval_lbl = QLabel("Access Check Month:")
        eval_lbl.setObjectName("FieldLabel")
        tb_lay.addWidget(eval_lbl)

        self.eval_month_combo = QComboBox()
        curr_month_name = calendar.month_name[date.today().month]
        for label, val in EVAL_MONTH_OPTIONS:
            display = f"Current ({curr_month_name})" if val == "current" else label
            self.eval_month_combo.addItem(display, val)
        self.eval_month_combo.setFixedWidth(225)
        self.eval_month_combo.currentIndexChanged.connect(self._on_eval_month_changed)
        tb_lay.addWidget(self.eval_month_combo)

        self.admin_badge = make_badge_label("Admin: —", "#1e3a8a", "#dbeafe")
        tb_lay.addWidget(self.admin_badge)

        right_lay.addWidget(topbar)

        # Pages
        self.pages_stack = QStackedWidget()
        self.dashboard_page = DashboardPage(self.db, self.engine)
        self.scanner_page = LiveScannerPage(self.db, self.engine)
        self.register_page = RegisterStudentPage(self.db, self.engine)
        self.research_page = ResearchPage(self.db, self.engine)
        self.payments_page = PaymentsPage(self.db)
        self.logs_page = AccessLogsPage(self.db)
        self.settings_page = SettingsPage(self.db, self.engine)

        self.pages_map: Dict[str, QWidget] = {
            "dashboard": self.dashboard_page,
            "scanner": self.scanner_page,
            "register": self.register_page,
            "research": self.research_page,
            "payments": self.payments_page,
            "logs": self.logs_page,
            "settings": self.settings_page,
        }
        for key, _lbl, _title in self.PAGE_META:
            self.pages_stack.addWidget(self.pages_map[key])

        # Wire inter-page signals
        self.dashboard_page.navigate_requested.connect(self.switch_page)
        self.register_page.student_registered.connect(self._on_data_mutated)
        self.research_page.data_changed.connect(self._on_data_mutated)
        self.payments_page.data_changed.connect(self._on_data_mutated)
        self.scanner_page.logs_updated.connect(self._on_logs_updated)

        right_lay.addWidget(self.pages_stack, 1)
        outer.addWidget(right_col, 1)

    def _on_login_success(self, admin: Dict[str, Any]) -> None:
        self.current_admin = admin
        self.admin_badge.setText(f"👤 {admin['username']} ({admin['role']})")
        self._propagate_context()
        self.root_stack.setCurrentIndex(1)
        self.switch_page("dashboard")

    def _logout(self) -> None:
        self._stop_all_cameras()
        if self.current_admin:
            self.db.log_admin_action(self.current_admin["username"], "LOGOUT", "Admin logged out")
        self.current_admin = None
        self.login_screen.reset_fields()
        self.root_stack.setCurrentIndex(0)

    def _stop_all_cameras(self) -> None:
        self.scanner_page.stop_camera()
        self.register_page.stop_camera()

    def _on_eval_month_changed(self) -> None:
        self.evaluation_mode = self.eval_month_combo.currentData()
        self._propagate_context()

    def _propagate_context(self) -> None:
        uname = self.current_admin["username"] if self.current_admin else DEFAULT_ADMIN_USER
        self.dashboard_page.set_evaluation_mode(self.evaluation_mode)
        self.scanner_page.set_context(uname, self.evaluation_mode)
        self.register_page.set_context(uname, self.evaluation_mode)
        self.research_page.set_context(uname, self.evaluation_mode)
        self.payments_page.set_context(uname, self.evaluation_mode)
        self.logs_page.set_admin(uname)
        if self.current_admin:
            self.settings_page.set_admin(self.current_admin)

    def _on_data_mutated(self, *_args) -> None:
        self.scanner_page.reload_gallery()
        self.dashboard_page.refresh()
        self.research_page.refresh()
        self.payments_page.refresh()

    def _on_logs_updated(self) -> None:
        self.dashboard_page.refresh()
        self.logs_page.refresh()

    def switch_page(self, key: str) -> None:
        if key not in self.pages_map:
            return
        # Always release cameras when switching pages to prevent device lock conflicts
        if key != "scanner":
            self.scanner_page.stop_camera()
        if key != "register":
            self.register_page.stop_camera()

        for k, btn in self.nav_buttons.items():
            btn.setProperty("active", k == key)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        for k, _lbl, title in self.PAGE_META:
            if k == key:
                self.page_title_lbl.setText(title)
                break

        self.pages_stack.setCurrentWidget(self.pages_map[key])

        if key == "dashboard":
            self.dashboard_page.refresh()
        elif key == "scanner":
            self.scanner_page.reload_gallery()
        elif key == "research":
            self.research_page.refresh()
        elif key == "payments":
            self.payments_page.refresh()
        elif key == "logs":
            self.logs_page.refresh()
        elif key == "settings":
            self.settings_page.refresh()

    def closeEvent(self, event) -> None:
        self._stop_all_cameras()
        super().closeEvent(event)


def main() -> int:
    ensure_data_directories()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyleSheet(APP_STYLESHEET)
    window = PondiFacialManagementApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

