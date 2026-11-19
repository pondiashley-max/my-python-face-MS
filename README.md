# python-projects

Collection of Python applications and tools by **Pondi Maxim**.

---

## Featured Project: PONDI Facial Management System (PFMS)

**File:** [`pondi_facial_management_system.py`](./pondi_facial_management_system.py)

An all-in-one, single-file **University Biometric Face Recognition, Student Registration, Tuition Access Control (FCFA), and Student Research** desktop application built with **Python**, **PyQt5**, **OpenCV (YuNet + SFace ONNX)**, **NumPy**, and **SQLite**.

### Key Features

- **Administrator Login & Account Security**
  - Dedicated Admin Login Screen on startup using salted **PBKDF2-SHA256** (`200,000` iterations) password hashing and constant-time verification.
  - Failed-attempt rate-limiting, one-click **Logout / Lock**, password updates, and multi-admin account management (Super Admin, Registrar Admin, Security Officer, Finance Admin) in **Admin Settings**.
- **Deep Biometric Face Detection & Simultaneous Multi-Student Recognition**
  - Uses **OpenCV YuNet** (`face_detection_yunet_2023mar.onnx`) for face + 5-landmark detection and **OpenCV SFace** (`face_recognition_sface_2021dec.onnx`) for 128-D L2-normalized face embeddings (cosine similarity matching against enrolled samples + centroid).
  - Automatic model discovery (`Mes models/`, script folder, or `pondi_fms_data/models/`), built-in one-click ONNX model downloader in Settings, and an automatic **Haar Cascade + gradient descriptor** fallback when offline without ONNX files.
  - Supports **live webcam scanning** and **image/group photo file scanning**, identifying **every detected student simultaneously** with bounding-box overlays and live identity cards displaying **Name, Matricule, Match Confidence, Phone Number, Date of Payment, Amount Paid, Academic Info, and Access Status**.
- **Guided Student & Front-Face Registration**
  - Captures **5 frontal face samples** via live webcam (selecting the sharpest, most frontal frame via YuNet landmarks & Laplacian sharpness to crop a clean ID-style front portrait) or enrolls from a front portrait image file.
  - Captures complete student records: **Matricule**, **Student Name**, **Date of Birth**, **Phone Number**, **Field**, **Option**, **Academic Year**, **School**, **Address**, **Amount Paid (FCFA)**, and **Date of Payment**.
  - **Age Determination & Accuracy Verification**: Automatically calculates exact chronological age (years, months, days, leap-year aware), validates against future dates and university age ranges, and supports optional **Declared Age Cross-Check** to flag exact matches (`VERIFIED`), off-by-1 birthday mistakes (`MISMATCH`), or out-of-range warnings (`WARNING`).
- **Cumulative Monthly Tuition Access Control (FCFA)**
  - Evaluates campus access strictly against the university's cumulative monthly tuition schedule:

    | Month | Monthly Tranche | Cumulative Required | Status when Met |
    | --- | ---: | ---: | --- |
    | **September** | 55,000 FCFA | **55,000 FCFA** | `ACCESS GRANTED` |
    | **October** (+ September) | + 250,000 FCFA | **305,000 FCFA** | `ACCESS GRANTED` |
    | **November** (+ October cumulative) | + 140,000 FCFA | **445,000 FCFA** | `ACCESS GRANTED` |
    | **December** (+ November cumulative) | + 40,000 FCFA | **485,000 FCFA** | `ACCESS GRANTED` (Full Annual Tuition) |

  - Top-bar **Access Check Month** switcher (`Current Month`, `September`, `October`, `November`, `December`, or `Student's Payment Date Month`).
  - Full **Tuition & Payments Ledger** to record incremental monthly payments, track shortfalls/arrears, view per-student payment history, and export CSV reports.
- **Student Research Screen (Matricule / Name / Field)**
  - Filter and search registered students by **Matricule**, **Student Name**, **Field**, **Access Status**, or across **All Fields**.
  - Clickable **Registered Fields** filter chips displaying counts per field; selecting a field (or running any search) lists every matching student's **registered front picture/photo on the left** (click to enlarge) with all personal, academic, payment, and access information displayed beside it.
  - Inline actions for **+ Add Payment**, **Full Dossier**, **Edit**, **Delete**, compact table toggle, and **CSV export**.
- **Dashboard, Access Audit Logs & Database Backup**
  - Real-time KPI summary (Total Students, Access Granted, Access Denied, Total Collected in FCFA, Distinct Fields), Field breakdown, timestamped face-scan **Access Logs** with filtering & CSV export, and one-click **SQLite Database Backup**.

---

### Requirements

- **Python 3.10+** (tested on Python 3.12)
- Dependencies:
  ```bash
  pip install opencv-contrib-python numpy PyQt5
  ```

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/pondiashley-max/python-projects.git
   cd python-projects
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-contrib-python numpy PyQt5
   ```

3. **Run the application**
   ```bash
   python3 pondi_facial_management_system.py
   ```

4. **Sign in as Administrator**
   - **Default Username:** `admin`
   - **Default Password:** `admin123`
   - *(Recommended: change the default password inside **Admin Settings** after your first login.)*

5. **Deep Learning ONNX Models (Optional First-Run Setup)**
   - If `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` are already placed in `Mes models/`, the script directory, or `pondi_fms_data/models/`, PFMS activates **YuNet + SFace (128-D)** automatically.
   - Otherwise, open **Admin Settings → Download / Verify YuNet & SFace ONNX Models** to fetch them automatically from OpenCV Zoo (or use the built-in Haar fallback offline).

---

### Runtime Data Storage

At runtime, the application automatically creates a local `pondi_fms_data/` directory (excluded from Git via `.gitignore`):

```text
pondi_fms_data/
├── pondi_students.db   # SQLite database (admins, students, payment_history, access_logs)
├── photos/             # Saved front ID portraits (<MATRICULE>_front.jpg)
└── models/             # Cached YuNet & SFace ONNX models (if downloaded via Settings)
```

---

### Author

**Pondi Maxim** ([@pondiashley-max](https://github.com/pondiashley-max))
