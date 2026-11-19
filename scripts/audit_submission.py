#!/usr/bin/env python3
"""
Submission audit: teslimden once repo'nun eksiksiz, temiz ve GitHub ile senkron
oldugunu kontrol eder. Sadece standart kutuphane kullanir.

Kullanim (proje kokunde, .venv aktifken):
    python scripts/audit_submission.py --fetch --run
Baska bir gorev icin:
    python scripts/audit_submission.py --fetch --run --notebook notebooks/task4.ipynb \
        --expect "0.41,0.55" --required "src/x/a.py,main.py"
Cikis kodu: 0 = FAIL yok, 1 = en az bir FAIL var.
"""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path.cwd()
RES = []  # (level, section, message)


def rec(level, section, msg):
    RES.append((level, section, msg))


def ok(s, m): rec("PASS", s, m)
def warn(s, m): rec("WARN", s, m)
def fail(s, m): rec("FAIL", s, m)


def git(*args):
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout.rstrip(), p.stderr.strip()


def check(cond, section, good, bad, level="FAIL"):
    rec("PASS" if cond else level, section, good if cond else bad)
    return cond


# --------------------------------------------------------------------------
def audit_files(a):
    S = "1. Dosyalar"
    for f in a.required:
        check((ROOT / f).exists(), S, f"mevcut: {f}", f"EKSIK: {f}")
    for f in a.forbidden:
        check(not (ROOT / f).exists(), S, f"yok (dogru): {f}", f"KALMIS, silinmeli: {f}")
    for p in ROOT.glob("scratch_*"):
        fail(S, f"gecici dosya kalmis: {p.name}")
    check(any((ROOT / "tests").glob("test_*.py")), S, "tests/ icinde test dosyalari var",
          "tests/ icinde test_*.py yok")


def audit_git(a):
    S = "2. Git / GitHub"
    rc, _, _ = git("rev-parse", "--is-inside-work-tree")
    if rc != 0:
        fail(S, "burasi bir git reposu degil"); return None
    rc, br, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    check(br == "main", S, "dal: main", f"dal 'main' degil: {br}", "WARN")
    rc, remotes, _ = git("remote", "-v")
    if not check(bool(remotes), S, "remote tanimli", "remote (origin) YOK: GitHub'a bagli degil"):
        return None
    ok(S, "remote: " + remotes.splitlines()[0].split()[1])
    if a.fetch:
        rc, _, err = git("fetch", "origin")
        check(rc == 0, S, "git fetch basarili", f"git fetch basarisiz (internet/giris?): {err[:120]}", "WARN")
    rc, st, _ = git("status", "--porcelain")
    if st:
        fail(S, f"commit edilmemis degisiklik var ({len(st.splitlines())} dosya): "
                + ", ".join(l[3:] for l in st.splitlines()[:6]))
    else:
        ok(S, "calisma dizini temiz (commit edilmemis degisiklik yok)")
    rc, up, _ = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if rc != 0:
        fail(S, "dal icin upstream yok (git push -u origin main yapilmamis)"); return None
    rc, ahead, _ = git("rev-list", "--count", f"{up}..HEAD")
    rc, behind, _ = git("rev-list", "--count", f"HEAD..{up}")
    check(ahead == "0", S, "push edilmemis commit yok",
          f"{ahead} commit PUSH EDILMEMIS (git push calistir)")
    check(behind == "0", S, "uzak repo ile senkron", f"uzak repo {behind} commit onde (git pull)", "WARN")
    _, h, _ = git("rev-parse", "HEAD"); _, r, _ = git("rev-parse", up)
    check(h == r, S, f"HEAD == {up} ({h[:7]})", f"HEAD ({h[:7]}) != {up} ({r[:7]})")
    _, files, _ = git("ls-files")
    return set(files.splitlines())


def audit_tracked(a, tracked):
    S = "3. Takip edilen dosyalar (GitHub'a gidecekler)"
    for f in a.required:
        if f in tracked:
            ok(S, f"takipte: {f}")
        else:
            _, why, _ = git("check-ignore", "-v", f)
            fail(S, f"GITHUB'A GITMIYOR: {f}" + (f"  <- .gitignore kurali: {why}" if why else "  (git add edilmemis)"))
    for f in a.forbidden:
        check(f not in tracked, S, f"takipte degil: {f}", f"repoda duruyor, silinmeli: {f}")
    dataset = "data/raw/Dataset.csv"
    history_status, dataset_history, _ = git("log", "--branches", "--remotes", "--oneline", "--", dataset)
    check(
        history_status == 0 and dataset not in tracked and not dataset_history,
        S,
        f"ham veri Git dizininde veya dal gecmisinde bulunmadi: {dataset}",
        f"veri lisansi dogrulanmadi: {dataset} takipte veya dal gecmisinde",
    )
    bad = [f for f in tracked if re.search(r"(^|/)(\.venv|__pycache__|\.pytest_cache|\.ipynb_checkpoints)(/|$)|\.DS_Store$|\.pyc$", f)]
    check(not bad, S, "venv/cache dosyasi takipte degil", "takipte olmamasi gerekenler: " + ", ".join(bad[:5]))
    gi = ROOT / ".gitignore"
    if gi.exists():
        lines = [l.strip() for l in gi.read_text(encoding="utf-8", errors="replace").splitlines()]
        for risky in ("*.ipynb", "*.png", "*.csv", "outputs/", "data/", "*.md"):
            check(risky not in lines, S, f".gitignore '{risky}' icermiyor",
                  f".gitignore '{risky}' iceriyor: teslim dosyalarini dislayabilir")
    for f in tracked:
        p = ROOT / f
        if p.exists():
            mb = p.stat().st_size / 1e6
            if mb > 50: fail(S, f"cok buyuk dosya ({mb:.0f} MB): {f}")
            elif mb > 10: warn(S, f"buyuk dosya ({mb:.0f} MB): {f}")


def load_nb(path, from_head):
    if from_head:
        rc, out, _ = git("show", f"HEAD:{path}")
        if rc == 0:
            return json.loads(out), "commit edilmis (HEAD) surum"
    p = ROOT / path
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")), "diskteki surum"
    return None, None


def audit_notebook(a, tracked):
    S = "4. Notebook"
    nb, src = load_nb(a.notebook, a.notebook in tracked)
    if nb is None:
        fail(S, f"notebook okunamadi: {a.notebook}"); return
    ok(S, f"gecerli JSON ({src})")
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    md = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    ok(S, f"{len(code)} kod hucresi, {len(md)} markdown hucresi")
    check(len(md) >= 8, S, "yeterli aciklama hucresi", "az markdown hucresi (<8): adim adim aciklama ekle", "WARN")
    unexec = sum(1 for c in code if c.get("execution_count") is None)
    check(unexec == 0, S, "calistirilmamis hucre yok", f"{unexec} hucre CALISTIRILMAMIS")
    errs = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
    check(errs == 0, S, "hatali hucre yok", f"{errs} hucrede HATA ciktisi var")
    figs = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    check(figs >= a.min_figures, S, f"{figs} grafik ciktisi kayitli", f"yalnizca {figs} grafik (en az {a.min_figures} beklenir)")
    text = json.dumps(nb, ensure_ascii=False)
    for e in a.expect:
        check(e in text, S, f"beklenen deger notebook ciktisinda: {e}", f"beklenen deger BULUNAMADI: {e}")
    check("Traceback" not in text, S, "Traceback izi yok", "notebook icinde Traceback metni var")


def audit_readme(a, tracked):
    S = "5. README"
    p = ROOT / "README.md"
    if not p.exists():
        fail(S, "README.md yok"); return
    t = p.read_text(encoding="utf-8", errors="replace")
    check(len(t) > 1500, S, f"yeterli uzunluk ({len(t)} karakter)", f"README cok kisa ({len(t)} karakter)", "WARN")
    for name, pat in {"kurulum": r"install|setup|kurulum", "calistirma": r"usage|how to run|run|kullan",
                      "sonuclar": r"result|sonu", "sinirlamalar": r"limitation|sinirlam|s.n.rlama"}.items():
        check(re.search(pat, t, re.I) is not None, S, f"'{name}' bolumu var", f"'{name}' bolumu eksik", "WARN")
    for w in ("TODO", "TBD", "lorem ipsum", "<your", "XXX"):
        check(w.lower() not in t.lower(), S, f"yer tutucu yok: {w}", f"yer tutucu metin: {w}")
    imgs = re.findall(r"!\[[^\]]*\]\(([^)\s]+)", t)
    local = [i for i in imgs if not i.startswith(("http", "data:"))]
    check(len(local) >= 3, S, f"{len(local)} yerel gorsel referansi", "README'de en az 3 gorsel olmali", "WARN")
    for i in local:
        rel = i.split("#")[0].lstrip("./")
        check(rel in tracked, S, f"gorsel GitHub'da gorunecek: {rel}",
              f"README'deki gorsel repoda YOK/takipte degil: {rel}")
    for e in a.expect:
        check(e in t, S, f"README'de sonuc degeri var: {e}", f"README'de sonuc degeri yok: {e}", "WARN")


def audit_requirements(a):
    S = "6. requirements.txt"
    p = ROOT / "requirements.txt"
    if not p.exists():
        fail(S, "requirements.txt yok"); return
    t = p.read_text(encoding="utf-8", errors="replace").lower()
    for pkg in a.packages:
        check(re.search(rf"^{re.escape(pkg)}\b", t, re.M) is not None, S, f"paket listede: {pkg}", f"paket EKSIK: {pkg}")
    unpinned = [l for l in t.splitlines() if l.strip() and not l.startswith("#") and not re.search(r"[=<>~]", l)]
    check(not unpinned, S, "tum paketlerde surum belirtilmis", "surumsuz paketler: " + ", ".join(unpinned), "WARN")


def audit_hygiene(tracked):
    S = "7. Gizlilik ve guvenlik"
    pats = {"GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}", "AWS anahtari": r"AKIA[0-9A-Z]{16}",
            "API anahtari": r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]"}
    leaks = {"kisisel yol": r"[A-Za-z]:\\\\?Users\\\\?\w+|/Users/\w+|/home/\w+"}
    texts = [f for f in tracked if f.endswith((".py", ".md", ".ipynb", ".toml", ".txt", ".json", ".yml", ".yaml"))]
    for label, pat in pats.items():
        hit = [f for f in texts if (ROOT / f).exists() and re.search(pat, (ROOT / f).read_text(encoding="utf-8", errors="ignore"), re.I)]
        check(not hit, S, f"{label} sizintisi yok", f"{label} bulundu: {', '.join(hit)}")
    for label, pat in leaks.items():
        hit = [f for f in texts if (ROOT / f).exists() and re.search(pat, (ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
        check(not hit, S, f"{label} (bilgisayar yolu) yok", f"{label} geciyor: {', '.join(hit)} (kullanici adin gorunur)", "WARN")


def audit_run(a):
    S = "8. Yeniden uretilebilirlik (--run)"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"(\d+) passed", p.stdout)
    check(p.returncode == 0 and m is not None, S, f"pytest: {m.group(1) if m else '?'} test gecti",
          "pytest BASARISIZ:\n" + (p.stdout + p.stderr)[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, a.main, "--data", a.data, "--out", tmp]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        check(p.returncode == 0, S, f"{a.main} hatasiz bitti (cikis kodu 0)",
              f"{a.main} hata verdi (kod {p.returncode}): " + p.stderr[-300:])
        got = re.findall(a.metric_regex, p.stdout + p.stderr)
        exp = a.expect
        if got:
            # The Task 3 CLI logs test metrics before training metrics.
            g = got[0] if isinstance(got[0], tuple) else (got[0],)
            ok(S, "main ciktisindan okunan metrikler: " + ", ".join(g))
            for val, e in zip(g, exp):
                check(abs(float(val) - float(e)) < 0.0006, S, f"metrik {val} ~ beklenen {e}", f"metrik SAPTI: {val} != {e}")
        else:
            warn(S, "main ciktisindan metrik okunamadi (--metric-regex kontrol et)")
        outs = [f.name for f in Path(tmp).glob("*")]
        check(len(outs) >= 5, S, f"{len(outs)} cikti dosyasi uretildi", f"yalnizca {len(outs)} cikti dosyasi uretildi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/task3_rating_classification.ipynb")
    ap.add_argument("--required", default=",".join([
        "README.md", "requirements.txt", "main.py", ".gitignore",
        "notebooks/task3_rating_classification.ipynb", "outputs/01_class_distribution.png",
        "outputs/02_model_comparison.png", "outputs/03_confusion_matrix.png",
        "outputs/04_feature_importance.png", "outputs/model_comparison.csv",
        "src/rating_classifier/pipeline.py", "tests/test_pipeline.py"]))
    ap.add_argument("--forbidden", default="Dataset.csv,task3_rating_classification.py,summary.md")
    ap.add_argument("--expect", default="0.6383,0.6376,0.5870", help="notebook/README'de gecmesi gereken degerler")
    ap.add_argument("--packages", default="pandas,numpy,scikit-learn,matplotlib,seaborn,pytest,jupyter")
    ap.add_argument("--min-figures", type=int, default=4)
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--data", default="data/raw/Dataset.csv")
    ap.add_argument(
        "--metric-regex",
        default=r"Metrics [^\r\n]*?Accuracy: (\d+\.\d+) \| Balanced Accuracy: \d+\.\d+ \| Macro F1: (\d+\.\d+)",
    )
    ap.add_argument("--fetch", action="store_true", help="once git fetch (GitHub ile gercek senkron kontrolu)")
    ap.add_argument("--run", action="store_true", help="pytest ve main.py'yi calistirip sonuclari dogrula (~1 dk)")
    a = ap.parse_args()
    for k in ("required", "forbidden", "expect", "packages"):
        setattr(a, k, [x.strip() for x in getattr(a, k).split(",") if x.strip()])

    tracked = audit_git(a) or set()
    audit_files(a)
    if tracked:
        audit_tracked(a, tracked)
    audit_notebook(a, tracked)
    audit_readme(a, tracked)
    audit_requirements(a)
    audit_hygiene(tracked)
    if a.run:
        audit_run(a)

    cur = None
    for lvl, sec, msg in sorted(RES, key=lambda r: r[1]):
        if sec != cur:
            print(f"\n=== {sec} ==="); cur = sec
        icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[lvl]
        print(f"{icon} {msg}")
    n = {k: sum(1 for r in RES if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
    print(f"\n{'=' * 50}\nOZET: {n['PASS']} gecti | {n['WARN']} uyari | {n['FAIL']} HATA")
    print("SONUC: " + ("TESLIME HAZIR" if n["FAIL"] == 0 else "TESLIM ETME - HATALARI DUZELT"))
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
