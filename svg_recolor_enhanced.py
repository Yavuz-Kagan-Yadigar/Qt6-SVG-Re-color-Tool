#!/usr/bin/env python3
"""
SVG Recolor Tool  v5
─────────────────────────────────────────────────────────────
  Performance : ThreadPoolExecutor (min(cpu*2, 16) workers)
                sliding-window submission – O(WINDOW) memory
  Log         : real-time coloured output, search box,
                "Errors only" checkbox
  UI          : dark flat buttons, coloured thin borders
  Output      : sibling folder, never touches originals
─────────────────────────────────────────────────────────────
"""

import os
import re
import sys
import shutil
import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QGroupBox, QProgressBar, QColorDialog, QGridLayout,
    QSpinBox, QMessageBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtSvgWidgets import QSvgWidget


# ══════════════════════════════════════════════════════════════════════════════
#  Colour helpers
# ══════════════════════════════════════════════════════════════════════════════

NAMED_COLORS = {
    'black':'#000000','white':'#ffffff','red':'#ff0000','green':'#008000',
    'blue':'#0000ff','yellow':'#ffff00','gray':'#808080','grey':'#808080',
    'silver':'#c0c0c0','maroon':'#800000','purple':'#800080','fuchsia':'#ff00ff',
    'lime':'#00ff00','olive':'#808000','navy':'#000080','teal':'#008080',
    'aqua':'#00ffff','orange':'#ffa500','coral':'#ff7f50','cyan':'#00ffff',
    'magenta':'#ff00ff','pink':'#ffc0cb','brown':'#a52a2a','darkgray':'#a9a9a9',
    'darkgrey':'#a9a9a9','lightgray':'#d3d3d3','lightgrey':'#d3d3d3',
    'darkblue':'#00008b','darkred':'#8b0000','darkgreen':'#006400',
    'gold':'#ffd700','indigo':'#4b0082','violet':'#ee82ee','wheat':'#f5deb3',
    'tomato':'#ff6347','salmon':'#fa8072','khaki':'#f0e68c','beige':'#f5f5dc',
    'ivory':'#fffff0','lavender':'#e6e6fa','plum':'#dda0dd',
    'turquoise':'#40e0d0','chocolate':'#d2691e','sienna':'#a0522d',
    'peru':'#cd853f',
}
_SKIP = frozenset(['none','transparent','currentcolor','inherit','unset','initial'])


def normalize_color(raw: str):
    if not raw: return None
    s = raw.strip().lower()
    if not s or s in _SKIP or s.startswith('url(') or s.startswith('var('): return None
    m = re.match(r'rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', s)
    if m: return '#{:02x}{:02x}{:02x}'.format(int(m[1]),int(m[2]),int(m[3]))
    m = re.match(r'rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\)', s)
    if m:
        if float(m[4]) == 0: return None
        return '#{:02x}{:02x}{:02x}'.format(int(m[1]),int(m[2]),int(m[3]))
    if re.match(r'^#[0-9a-f]{3}$', s): return '#'+''.join(c*2 for c in s[1:])
    if re.match(r'^#[0-9a-f]{6}$', s): return s
    if re.match(r'^#[0-9a-f]{8}$', s): return '#'+s[1:7]
    return NAMED_COLORS.get(s)


def hex_to_rgb(h):
    h = h.lstrip('#')
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)


def interpolate(c1, c2, t):
    r1,g1,b1 = hex_to_rgb(c1); r2,g2,b2 = hex_to_rgb(c2)
    return '#{:02x}{:02x}{:02x}'.format(
        round(r1+(r2-r1)*t), round(g1+(g2-g1)*t), round(b1+(b2-b1)*t))


def build_gradient(base, n):
    k = len(base)
    if n <= 0: return []
    if n == 1: return [base[0]]
    if n <= k: return base[:n]
    res = []
    for i in range(n):
        t = i/(n-1)
        seg = min(int(t*(k-1)), k-2)
        res.append(interpolate(base[seg], base[seg+1], (t-seg/(k-1))*(k-1)))
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  File I/O
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_svg_path(path: str) -> str:
    """
    Some icon packs (Papirus on Leap) use text-redirect files:
    the file contains only a relative path like 'yast.svg' or '../base/icon.svg'.
    Resolve up to 8 hops to find the real SVG.
    """
    seen = set()
    current = path
    for _ in range(8):
        real = os.path.realpath(current)
        if real in seen:
            break
        seen.add(real)
        try:
            with open(current, 'rb') as fh:
                head = fh.read(256)
        except OSError:
            break
        # Strip BOM and whitespace
        if head.startswith(b'\xef\xbb\xbf'):
            head = head[3:]
        head = head.strip()
        # Looks like a real SVG or binary — stop resolving
        if head[:2] == b'\x1f\x8b' or head[:4] in (b'<svg', b'<?xm', b'<!--'):
            break
        if b'<svg' in head[:64].lower():
            break
        # Check if entire content is a short path-like string (text redirect)
        try:
            text = head.decode('utf-8').strip()
        except Exception:
            break
        # Must be a single token that looks like a filename (no spaces, ends with .svg)
        if '\n' not in text and ' ' not in text and len(text) < 256 and text.lower().endswith('.svg'):
            candidate = os.path.join(os.path.dirname(current), text)
            if os.path.isfile(candidate):
                current = candidate
                continue
            # Try stripping leading directory components
            candidate2 = os.path.join(os.path.dirname(current), os.path.basename(text))
            if os.path.isfile(candidate2):
                current = candidate2
                continue
        break
    return current


def read_svg(path):
    with open(path,'rb') as fh: raw = fh.read()
    # Strip UTF-8 BOM
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    # SVGZ: gzip-compressed SVG (magic bytes 1f 8b)
    elif raw[:2] == b'\x1f\x8b':
        try:
            import gzip
            raw = gzip.decompress(raw)
            if raw.startswith(b'\xef\xbb\xbf'): raw = raw[3:]
        except Exception:
            pass
    # UTF-16 BOM
    elif raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
        try: return raw.decode('utf-16')
        except Exception: pass
    for enc in ('utf-8','latin-1','cp1252'):
        try: return raw.decode(enc)
        except UnicodeDecodeError: continue
    return raw.decode('utf-8', errors='replace')


def preprocess(text):
    text = re.sub(r'<!DOCTYPE\b[^>\[]*(?:\[[^\]]*\])?\s*>', '', text,
                  flags=re.DOTALL|re.IGNORECASE)
    for e,n in (('&nbsp;','&#160;'),('&copy;','&#169;'),('&reg;','&#174;'),
                ('&trade;','&#8482;'),('&mdash;','&#8212;'),('&ndash;','&#8211;'),
                ('&hellip;','&#8230;'),('&ldquo;','&#8220;'),('&rdquo;','&#8221;'),
                ('&lsquo;','&#8216;'),('&rsquo;','&#8217;')):
        text = text.replace(e, n)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  Colour collection  (tracks raw forms – the v3 fix)
# ══════════════════════════════════════════════════════════════════════════════

_NAMED_ALT = '|'.join(re.escape(k) for k in sorted(NAMED_COLORS, key=len, reverse=True))
_COLOR_VAL = (r'#[0-9a-fA-F]{8}|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}'
              r'|rgba?\s*\([^)]+\)|(?:'+_NAMED_ALT+r')')
_COLLECT_RE = re.compile(
    r'(?:fill|stroke|stop-color|color|flood-color|lighting-color)'
    r'\s*[=:]\s*["\']?\s*('+_COLOR_VAL+r')', re.IGNORECASE)

# CSS custom property definitions: --foo-bar: #colour;
_CSS_VAR_DEF_RE = re.compile(
    r'(--[\w-]+\s*:\s*)('+_COLOR_VAL+r')', re.IGNORECASE)

_STYLE_RE = re.compile(r'(<style[^>]*>)(.*?)(</style\s*>)', re.DOTALL|re.IGNORECASE)
_ATTR_RE  = re.compile(
    r'((?:fill|stroke|stop-color|color|flood-color|lighting-color)\s*=\s*["\'])([^"\']*?)(["\'])'
    r'|(style\s*=\s*["\'])([^"\']*?)(["\'])', re.IGNORECASE)


def collect_colors(text):
    r2n = {}
    # Standard SVG colour attributes and CSS properties
    for m in _COLLECT_RE.finditer(text):
        raw = m.group(1).strip(); norm = normalize_color(raw)
        if norm:
            rl = raw.lower()
            if rl not in r2n: r2n[rl] = norm
    # CSS custom property definitions: --primary: #5294e2
    for m in _CSS_VAR_DEF_RE.finditer(text):
        raw = m.group(2).strip(); norm = normalize_color(raw)
        if norm:
            rl = raw.lower()
            if rl not in r2n: r2n[rl] = norm
    return r2n, set(r2n.values())


def _make_pattern(r2n, n2new):
    rel = {r:n for r,n in r2n.items() if n in n2new}
    if not rel: return None, None
    pat = re.compile(
        r'(?<![0-9a-zA-Z#])('
        + '|'.join(re.escape(r) for r in sorted(rel, key=len, reverse=True))
        + r')(?![0-9a-zA-Z])', re.IGNORECASE)
    def rep(m):
        n = normalize_color(m.group(1))
        return n2new.get(n, m.group(1)) if n else m.group(1)
    return pat, rep


def recolor_text(text, r2n, n2new):
    pat, rep = _make_pattern(r2n, n2new)
    if pat is None: return text
    def in_style(m): return m.group(1)+pat.sub(rep, m.group(2))+m.group(3)
    res = _STYLE_RE.sub(in_style, text)
    def in_attr(m):
        if m.group(1):
            n = normalize_color(m.group(2).strip())
            return m.group(1)+(n2new.get(n,m.group(2)) if n else m.group(2))+m.group(3)
        return m.group(4)+pat.sub(rep, m.group(5))+m.group(6)
    return _ATTR_RE.sub(in_attr, res)

# ══════════════════════════════════════════════════════════════════════════════
#  Module-level worker  (called from thread pool)
# ══════════════════════════════════════════════════════════════════════════════

def _read_redirect_target(path: str):
    """
    If `path` is a text-redirect file (contains only a bare filename like 'folder-new.svg'),
    return the target filename string. Otherwise return None.
    """
    try:
        with open(path, 'rb') as fh:
            head = fh.read(512)
    except OSError:
        return None
    if head.startswith(b'\xef\xbb\xbf'):
        head = head[3:]
    head_stripped = head.strip()
    # Real SVG / SVGZ starts with these
    if (head_stripped[:2] == b'\x1f\x8b'
            or head_stripped[:4] in (b'<svg', b'<?xm', b'<!--')
            or b'<svg' in head_stripped[:128].lower()):
        return None
    try:
        text = head_stripped.decode('utf-8').strip()
    except Exception:
        return None
    # Single token, no whitespace, ends with .svg, short enough
    if ('\n' not in text and '\r' not in text and ' ' not in text
            and len(text) < 256 and text.lower().endswith('.svg')):
        return text
    return None


def _recolor_file(src_path, dst_path, colors, mono_color):
    """Returns (fname, success, info_str)."""
    fname = os.path.basename(src_path)
    try:
        # ── Detect text-redirect (shortcut) files ─────────────────────────
        redirect_target = _read_redirect_target(src_path)
        if redirect_target is not None:
            # Resolve the target relative to src_path's directory
            src_dir   = os.path.dirname(src_path)
            target_abs = os.path.normpath(os.path.join(src_dir, redirect_target))
            if not os.path.isfile(target_abs):
                # Try just the basename in the same dir
                target_abs = os.path.join(src_dir, os.path.basename(redirect_target))
            if not os.path.isfile(target_abs):
                return fname, False, 'Redirect target not found: {}'.format(redirect_target)

            # In the OUTPUT folder, create an OS symlink pointing at the
            # corresponding output file for the target.
            # Both files will be recolored; the symlink makes KDE happy.
            dst_dir         = os.path.dirname(dst_path)
            target_basename = os.path.basename(target_abs)
            # The target's output path will sit in the same output subdirectory
            # (same relative structure as source)
            src_root = os.path.commonpath([src_path, target_abs])
            rel_to_src_dir  = os.path.relpath(target_abs, src_dir)
            symlink_target  = rel_to_src_dir          # relative symlink

            os.makedirs(dst_dir, exist_ok=True)
            if os.path.lexists(dst_path):
                os.remove(dst_path)
            os.symlink(symlink_target, dst_path)
            return fname, True, 'symlink → {}'.format(redirect_target)

        # ── Normal SVG ─────────────────────────────────────────────────────
        text = read_svg(src_path)
        if not text.strip(): return fname, False, 'Empty file'
        text = preprocess(text)
        r2n, nset = collect_colors(text)
        if not nset:
            for h in re.findall(r'#[0-9a-fA-F]{3,8}\b', text):
                n = normalize_color(h)
                if n: r2n[h.lower()] = n; nset.add(n)
        if not nset: return fname, False, 'No colours found'

        if len(nset) == 1:
            n2new = {n: mono_color for n in nset}
            tag = 'mono → {}'.format(mono_color)
        else:
            u = sorted(nset); g = build_gradient(colors, len(u))
            n2new = dict(zip(u, g)); tag = '{} colours'.format(len(u))

        new_text = recolor_text(text, r2n, n2new)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, 'w', encoding='utf-8') as fh: fh.write(new_text)
        return fname, True, tag
    except Exception as e:
        return fname, False, str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  Worker thread  (runs the thread pool, emits signals)
# ══════════════════════════════════════════════════════════════════════════════

class ProcessThread(QThread):
    # (fname, success, info)
    file_done    = pyqtSignal(str, bool, str)
    # (done_count, total_count)
    progress     = pyqtSignal(int, int)
    finished     = pyqtSignal(int, int)
    stopped      = pyqtSignal()

    def __init__(self, src_dir, out_dir, colors, mono_color):
        super().__init__()
        self.src_dir    = src_dir
        self.out_dir    = out_dir
        self.colors     = colors
        self.mono_color = mono_color
        self._running   = True
        self._mutex     = QMutex()

    def stop(self):
        self._mutex.lock(); self._running = False; self._mutex.unlock()

    def is_running(self):
        self._mutex.lock(); v = self._running; self._mutex.unlock(); return v

    def run(self):
        # ── Phase 1: copy non-SVG files + preserve all OS symlinks ────────
        # OS symlinks (even .svg ones like @2x dirs) are recreated as symlinks.
        # Only regular (non-symlink) SVG files go into the recolor queue.
        for root, dirs, files in os.walk(self.src_dir, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for f in sorted(files):
                src = os.path.join(root, f)
                dst = os.path.join(self.out_dir, os.path.relpath(src, self.src_dir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)

                if os.path.islink(src):
                    # Always recreate OS symlinks verbatim
                    link_target = os.readlink(src)
                    if os.path.lexists(dst): os.remove(dst)
                    os.symlink(link_target, dst)
                elif not f.lower().endswith('.svg'):
                    # Non-SVG regular file → copy
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

        # ── Phase 2: collect recolor jobs (regular SVG files only) ────────
        all_jobs = []
        for root, dirs, files in os.walk(self.src_dir, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
            for f in sorted(files):
                src = os.path.join(root, f)
                if f.lower().endswith('.svg') and not os.path.islink(src):
                    dst = os.path.join(self.out_dir, os.path.relpath(src, self.src_dir))
                    all_jobs.append((src, dst))

        total = len(all_jobs)
        self.progress.emit(0, total)
        if total == 0:
            self.finished.emit(0, 0); return

        # ── Phase 3: thread pool with sliding window ───────────────────────
        n_workers = min((os.cpu_count() or 4) * 2, 16)
        WINDOW    = max(n_workers * 8, 128)

        done_count = ok_count = 0
        job_idx    = 0
        in_flight  = {}

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            while job_idx < total and len(in_flight) < WINDOW:
                src, dst = all_jobs[job_idx]
                f = ex.submit(_recolor_file, src, dst, self.colors, self.mono_color)
                in_flight[f] = src; job_idx += 1

            while in_flight:
                if not self.is_running():
                    for f in list(in_flight): f.cancel()
                    break

                completed, _ = wait(in_flight, timeout=0.08,
                                    return_when=FIRST_COMPLETED)
                for f in completed:
                    src = in_flight.pop(f)
                    done_count += 1
                    try:
                        fname, ok, info = f.result()
                    except Exception as e:
                        fname, ok, info = os.path.basename(src), False, str(e)
                    if ok: ok_count += 1
                    self.file_done.emit(fname, ok, info)
                    self.progress.emit(done_count, total)

                    if job_idx < total and self.is_running():
                        src2, dst2 = all_jobs[job_idx]
                        f2 = ex.submit(_recolor_file, src2, dst2,
                                       self.colors, self.mono_color)
                        in_flight[f2] = src2; job_idx += 1

        if self.is_running():
            self.finished.emit(done_count, ok_count)
        else:
            self.stopped.emit()


# ══════════════════════════════════════════════════════════════════════════════
#  Colour picker button
# ══════════════════════════════════════════════════════════════════════════════

class ColorButton(QPushButton):
    def __init__(self, hex_color, label='', parent=None):
        super().__init__(parent)
        self._label = label
        self.setFixedSize(46, 46)
        self.set_color(QColor(hex_color))
        self.clicked.connect(self._pick)

    def set_color(self, color):
        self._color = color
        self.setStyleSheet(
            'QPushButton{{background:{c};border:2px solid #555;border-radius:23px;}}'
            'QPushButton:hover{{border:2px solid #888;border-radius:23px;}}'
            .format(c=color.name()))
        self.setToolTip('{}: {}'.format(self._label, color.name()))

    def _pick(self):
        c = QColorDialog.getColor(self._color, self, 'Choose Colour',
                                  QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid(): self.set_color(c)

    def hex(self): return self._color.name()


# ══════════════════════════════════════════════════════════════════════════════
#  Filtered log widget
# ══════════════════════════════════════════════════════════════════════════════

# HTML colour for each category
_CAT_COLOR = {
    'ok':   '#55cc66',
    'err':  '#ff5555',
    'info': '#888888',
    'sep':  '#555566',
}
_MAX_DISPLAY = 30_000   # render at most this many lines in the widget


class FilteredLog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._msgs: list[tuple[str,str]] = []  # (category, text)
        self._search_text = ''

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Only the errors checkbox lives inside the log widget now
        top = QHBoxLayout()
        top.addStretch()
        self._errors_only = QCheckBox('Errors only')
        top.addWidget(self._errors_only)
        layout.addLayout(top)

        self._view = QTextEdit()
        self._view.setReadOnly(True)
        self._view.setStyleSheet(
            'QTextEdit{background:#0d0d0d;border:1px solid #282828;border-radius:5px;'
            'font-family:Consolas,"Courier New",monospace;font-size:11px;padding:6px;}')
        layout.addWidget(self._view)

        # debounce filter changes
        self._rebuild_timer = QTimer()
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(120)
        self._rebuild_timer.timeout.connect(self._rebuild)

        self._errors_only.stateChanged.connect(self._rebuild_timer.start)

    # ── public API ─────────────────────────────────────────────────────────

    def set_search(self, text: str):
        """Called externally when the shared search bar changes."""
        self._search_text = text.strip().lower()
        self._rebuild_timer.start()

    def append(self, category: str, text: str):
        """category: 'ok' | 'err' | 'info' | 'sep'"""
        self._msgs.append((category, text))
        if self._passes(category, text):
            self._append_html(self._to_html(category, text))

    def clear(self):
        self._msgs.clear()
        self._view.clear()

    # ── internals ──────────────────────────────────────────────────────────

    def _passes(self, category, text):
        if self._errors_only.isChecked() and category not in ('err', 'sep'):
            if not text.startswith('═') and not text.startswith('✅') \
               and not text.startswith('⚠') and not text.startswith('📁'):
                return False
        if self._search_text and self._search_text not in text.lower():
            return False
        return True

    def _to_html(self, category, text):
        safe = (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    .replace('\n', '<br>'))
        color = _CAT_COLOR.get(category, '#aaaaaa')
        return '<span style="color:{}">{}</span>'.format(color, safe)

    def _append_html(self, html):
        self._view.append(html)
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _rebuild(self):
        """Rebuild display from stored messages (called when filter changes)."""
        visible = [m for m in self._msgs if self._passes(*m)]
        truncated = len(visible) > _MAX_DISPLAY
        if truncated:
            visible = visible[-_MAX_DISPLAY:]

        self._view.clear()
        html_parts = []
        if truncated:
            html_parts.append(
                '<span style="color:#666">… earlier entries hidden '
                '(showing last {:,})</span>'.format(_MAX_DISPLAY))

        for cat, txt in visible:
            html_parts.append(self._to_html(cat, txt))

        self._view.setHtml('<br>'.join(html_parts))
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════

_APP_QSS = """
QWidget {
    background-color: #141414;
    color: #d0d0d0;
    font-size: 13px;
    font-weight: 600;
}
QLabel {
    color: #d8d8d8;
    font-size: 13px;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #2e2e2e;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 4px;
    font-size: 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #bbbbbb;
    font-weight: 700;
}

/* ── base button ── */
QPushButton {
    background-color: #222222;
    color: #cccccc;
    border: 1px solid #3a3a3a;
    border-radius: 5px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover    { background-color: #2a2a2a; border-color: #555; }
QPushButton:pressed  { background-color: #191919; }
QPushButton:disabled { color: #404040; border-color: #252525; background:#1c1c1c; }

/* ── coloured-border action buttons ── */
QPushButton#btn_start  { border-color: #2d7d45; color: #88dd99; }
QPushButton#btn_start:hover  { border-color: #3daa5d; background:#252525; }
QPushButton#btn_start:disabled { border-color: #253530; color:#3d6347; }

QPushButton#btn_stop   { border-color: #7d5a20; color: #ddaa55; }
QPushButton#btn_stop:hover   { border-color: #aa7a30; background:#252525; }

QPushButton#btn_open   { border-color: #1f5f8a; color: #66aadd; }
QPushButton#btn_open:hover   { border-color: #2d80ba; background:#252525; }
QPushButton#btn_open:disabled { border-color: #1a2d38; color:#2a4a5a; }

QPushButton#btn_delete { border-color: #7a2222; color: #dd6666; }
QPushButton#btn_delete:hover { border-color: #aa3333; background:#252525; }
QPushButton#btn_delete:disabled { border-color: #2e1a1a; color:#4a2525; }

/* ── inputs ── */
QLineEdit, QSpinBox {
    background-color: #1a1a1a;
    border: 1px solid #333;
    border-radius: 4px;
    color: #d0d0d0;
    font-size: 13px;
    font-weight: 600;
    padding: 4px 8px;
    selection-background-color: #1f5f8a;
}
QLineEdit:focus, QSpinBox:focus { border-color: #1f5f8a; }

QSpinBox { padding-right: 20px; }
QSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px; height: 13px;
    background: #2e2e2e;
    border: none;
    border-left: 1px solid #3a3a3a;
    border-bottom: 1px solid #3a3a3a;
    border-top-right-radius: 4px;
}
QSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px; height: 13px;
    background: #2e2e2e;
    border: none;
    border-left: 1px solid #3a3a3a;
    border-top: 1px solid #3a3a3a;
    border-bottom-right-radius: 4px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #3d3d3d; }
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed { background: #222; }
QSpinBox::up-arrow {
    image: none;
    width: 0; height: 0;
    border-style: solid;
    border-width: 0 4px 6px 4px;
    border-color: transparent transparent #c8c8c8 transparent;
}
QSpinBox::down-arrow {
    image: none;
    width: 0; height: 0;
    border-style: solid;
    border-width: 6px 4px 0 4px;
    border-color: #c8c8c8 transparent transparent transparent;
}
QSpinBox::up-arrow:disabled   { border-bottom-color: #555; }
QSpinBox::down-arrow:disabled { border-top-color: #555; }

/* ── progress bar ── */
QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #2e2e2e;
    border-radius: 4px;
    color: #888;
    text-align: center;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2a6a3a;
    border-radius: 3px;
}

/* ── checkbox ── */
QCheckBox { color: #c8c8c8; spacing: 6px; font-size: 13px; font-weight: 600; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #444; border-radius: 3px; background: #1a1a1a;
}
QCheckBox::indicator:checked {
    background: #2a6a3a; border-color: #3a8a4a;
    image: none;
}
QCheckBox::indicator:hover { border-color: #666; }

/* ── scrollbars ── */
QScrollBar:vertical {
    background: #151515; width: 7px; border-radius: 4px; border: none;
}
QScrollBar::handle:vertical {
    background: #333; border-radius: 3px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #444; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

/* ── message box ── */
QMessageBox { background: #1a1a1a; }
QMessageBox QLabel { color: #ccc; }
"""


class SVGRecolorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.directory   = ''
        self.output_dir  = None
        self.session_run = 0
        self._all_output_svgs = []   # full list, never filtered
        self.preview_list= []
        self.preview_idx = 0
        self.thread      = None
        self._build_ui()
        self._sync()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle('SVG Recolor Tool  v2')
        self.setGeometry(100, 100, 1240, 920)

        cw = QWidget(); self.setCentralWidget(cw)
        rh = QHBoxLayout(cw); rh.setSpacing(16); rh.setContentsMargins(18,18,18,18)

        # ── LEFT: preview + log ────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(400)
        lv = QVBoxLayout(left); lv.setSpacing(12)

        pg = QGroupBox('Preview')
        pl = QVBoxLayout(pg)
        self.svg_w = QSvgWidget()
        self.svg_w.setFixedSize(320, 320)
        self.svg_w.setStyleSheet(
            'QSvgWidget{background:#1a1a1a;border:1px solid #2e2e2e;border-radius:7px;}')
        pl.addWidget(self.svg_w, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_fname = QLabel('No preview')
        self.lbl_fname.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_fname.setWordWrap(True)
        self.lbl_fname.setStyleSheet('color:#666;font-size:11px;')
        pl.addWidget(self.lbl_fname)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton('◀')
        self.btn_prev.setFixedSize(58, 32); self.btn_prev.clicked.connect(self._prev)
        nav.addWidget(self.btn_prev)
        self.lbl_count = QLabel('0 / 0')
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setStyleSheet('color:#666;')
        nav.addWidget(self.lbl_count)
        self.btn_next = QPushButton('▶')
        self.btn_next.setFixedSize(58, 32); self.btn_next.clicked.connect(self._next)
        nav.addWidget(self.btn_next)
        pl.addLayout(nav)
        lv.addWidget(pg)

        # ── Shared search bar (between preview and log) ────────────────────
        search_w = QWidget()
        search_lay = QHBoxLayout(search_w)
        search_lay.setContentsMargins(0, 0, 0, 0)
        search_lay.setSpacing(6)
        search_lbl = QLabel('🔍')
        search_lbl.setStyleSheet('color:#888;font-size:13px;')
        search_lay.addWidget(search_lbl)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText('Search files — filters preview & log…')
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setStyleSheet(
            'QLineEdit{background:#1a1a1a;border:1px solid #333;border-radius:4px;'
            'color:#d0d0d0;padding:5px 8px;font-size:12px;}'
            'QLineEdit:focus{border-color:#1f5f8a;}')
        self.search_box.textChanged.connect(self._on_search_changed)
        search_lay.addWidget(self.search_box)
        lv.addWidget(search_w)

        lg = QGroupBox('Log')
        ll = QVBoxLayout(lg)
        self.flog = FilteredLog()
        ll.addWidget(self.flog)
        lv.addWidget(lg)
        rh.addWidget(left)

        # ── RIGHT: controls ────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setSpacing(12)

        # Directory
        dg = QGroupBox('Source Directory')
        dl = QHBoxLayout(dg)
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText('Select a folder containing SVG files…')
        dl.addWidget(self.dir_edit)
        bb = QPushButton('Browse'); bb.setFixedWidth(80); bb.clicked.connect(self._browse)
        dl.addWidget(bb)
        rv.addWidget(dg)

        # Output label
        og = QGroupBox('Output Folder')
        ol = QVBoxLayout(og)
        self.lbl_output = QLabel('— (will be created as a sibling next to source folder)')
        self.lbl_output.setWordWrap(True)
        self.lbl_output.setStyleSheet('color:#557799;font-size:11px;')
        ol.addWidget(self.lbl_output)
        rv.addWidget(og)

        # Colour count
        cg = QGroupBox('Gradient Colour Count')
        cgl = QHBoxLayout(cg)
        cgl.addWidget(QLabel('Gradient colours (2 – 6):'))
        self.spin = QSpinBox(); self.spin.setRange(2,6); self.spin.setValue(5)
        self.spin.setFixedWidth(54)
        self.spin.valueChanged.connect(self._refresh_cbts)
        cgl.addWidget(self.spin); cgl.addStretch()
        rv.addWidget(cg)

        # Gradient buttons
        gg = QGroupBox('Gradient Colours  (multi-colour icons)')
        ggl = QGridLayout(gg); ggl.setSpacing(6)
        LABELS   = ['Color 1','Color 2','Color 3','Color 4','Color 5','Color 6']
        DEFAULTS = ['#000000','#464646','#D81C4A','#b6b6b6','#ffffff','#888888']
        self.cbts = []
        for i in range(6):
            w=QWidget(); wl=QHBoxLayout(w); wl.setContentsMargins(3,3,3,3)
            b=ColorButton(DEFAULTS[i], LABELS[i])
            self.cbts.append(b); wl.addWidget(b)
            lb=QLabel(LABELS[i]); lb.setStyleSheet('color:#999;font-size:11px;')
            wl.addWidget(lb); wl.addStretch()
            ggl.addWidget(w, i//3, i%3)
        rv.addWidget(gg)

        # Mono colour
        mg = QGroupBox('Monochrome Colour  (single-colour icons)')
        ml = QHBoxLayout(mg)
        self.mono_btn = ColorButton('#acacac','Mono')
        ml.addWidget(self.mono_btn)
        ml.addWidget(QLabel('Replacement for single-colour icons'))
        ml.addStretch()
        rv.addWidget(mg)

        # Controls
        ag = QGroupBox('Controls')
        al = QGridLayout(ag); al.setSpacing(8)

        self.btn_start = QPushButton('▶  Start Processing')
        self.btn_start.setObjectName('btn_start')
        self.btn_start.setFixedHeight(42)
        self.btn_start.setStyleSheet(self.btn_start.styleSheet())
        self.btn_start.clicked.connect(self._start)
        al.addWidget(self.btn_start, 0, 0, 1, 2)

        self.btn_stop = QPushButton('■  Stop')
        self.btn_stop.setObjectName('btn_stop')
        self.btn_stop.setFixedHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        al.addWidget(self.btn_stop, 1, 0)

        self.btn_open = QPushButton('📂  Open Output Folder')
        self.btn_open.setObjectName('btn_open')
        self.btn_open.setFixedHeight(42)
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_output)
        al.addWidget(self.btn_open, 1, 1)

        self.btn_delete = QPushButton('🗑  Delete Output')
        self.btn_delete.setObjectName('btn_delete')
        self.btn_delete.setFixedHeight(42)
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_output)
        al.addWidget(self.btn_delete, 2, 0, 1, 2)

        rv.addWidget(ag)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFormat('%v / %m  files  (%p%)')
        self.progress.setValue(0); self.progress.setFixedHeight(20)
        self.progress.hide()
        rv.addWidget(self.progress)

        # Usage hint
        hint = QLabel(
            '• Originals are never modified — output goes to a sibling folder.\n'
            '• Single-colour icons → monochrome colour.   Multi-colour → gradient map.\n'
            '• Search bar filters both the preview and the log simultaneously.\n'
            '• Progress bar shows processed / total file count in real time.\n'
            '• "Delete Output" removes only the last output folder.'
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            'color:#777;font-style:italic;font-size:11px;font-weight:500;'
            'background:#1a1a1a;padding:10px;border-radius:6px;border:1px solid #2a2a2a;')
        rv.addWidget(hint)

        rv.addStretch()
        rh.addWidget(right)

        self._refresh_cbts()

    def _refresh_cbts(self):
        n = self.spin.value()
        for i,b in enumerate(self.cbts):
            b.setVisible(i<n); b.setEnabled(i<n)

    # ── state ──────────────────────────────────────────────────────────────

    def _sync(self):
        busy    = self.progress.isVisible()
        has_out = bool(self.output_dir and os.path.exists(self.output_dir))
        self.btn_start.setEnabled(not busy and bool(self.directory))
        self.btn_stop.setEnabled(busy)
        self.btn_open.setEnabled(not busy and has_out)
        self.btn_delete.setEnabled(not busy and has_out)
        n = len(self.preview_list)
        self.btn_prev.setEnabled(n>0 and self.preview_idx>0)
        self.btn_next.setEnabled(n>0 and self.preview_idx<n-1)

    def _log(self, text, category='info'):
        self.flog.append(category, text)

    def _on_file_done(self, fname, ok, info):
        cat = 'ok' if ok else 'err'
        sym = '✓' if ok else '✗'
        self._log('  {}  {}  ({})'.format(sym, fname, info), cat)

    # ── preview ────────────────────────────────────────────────────────────

    def _load_preview(self):
        self._all_output_svgs = []
        self.preview_idx = 0
        if self.output_dir and os.path.exists(self.output_dir):
            for r, _, fs in os.walk(self.output_dir):
                for f in fs:
                    if f.lower().endswith('.svg'):
                        self._all_output_svgs.append(os.path.join(r, f))
            self._all_output_svgs.sort()
        self._apply_search_to_preview()
        self._sync()

    def _on_search_changed(self, text: str):
        """Called when the shared search bar changes — updates both log and preview."""
        self.flog.set_search(text)
        self._apply_search_to_preview()

    def _apply_search_to_preview(self):
        """Filter preview_list from _all_output_svgs using current search text."""
        needle = self.search_box.text().strip().lower()
        if needle:
            self.preview_list = [
                p for p in self._all_output_svgs
                if needle in os.path.basename(p).lower()
            ]
        else:
            self.preview_list = list(self._all_output_svgs)
        self.preview_idx = 0
        (self._show_preview if self.preview_list else self._clear_preview)()
        self._sync()

    def _show_preview(self):
        if not self.preview_list: return self._clear_preview()
        p = self.preview_list[self.preview_idx]
        self.svg_w.load(p)
        self.lbl_fname.setText(os.path.basename(p))
        needle = self.search_box.text().strip()
        total_all = len(self._all_output_svgs)
        total_filtered = len(self.preview_list)
        if needle and total_filtered < total_all:
            self.lbl_count.setText('{} / {}  (filtered: {}/{})'.format(
                self.preview_idx + 1, total_filtered, total_filtered, total_all))
        else:
            self.lbl_count.setText('{} / {}'.format(
                self.preview_idx + 1, total_filtered))

    def _clear_preview(self):
        self.svg_w.load(b'')
        self.lbl_fname.setText('No preview')
        self.lbl_count.setText('0 / 0')

    def _prev(self):
        if self.preview_idx>0:
            self.preview_idx-=1; self._show_preview(); self._sync()

    def _next(self):
        if self.preview_idx<len(self.preview_list)-1:
            self.preview_idx+=1; self._show_preview(); self._sync()

    # ── output folder ──────────────────────────────────────────────────────

    def _make_out_dir(self):
        self.session_run += 1
        src_name   = os.path.basename(self.directory.rstrip('/\\'))
        parent_dir = os.path.dirname(self.directory.rstrip('/\\'))
        ts         = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name       = '{}_recolored_{}_v{}'.format(src_name, ts, self.session_run)
        return os.path.join(parent_dir, name)

    # ── actions ────────────────────────────────────────────────────────────

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Select Source Folder')
        if d:
            self.directory = d; self.dir_edit.setText(d)
            src = os.path.basename(d.rstrip('/\\'))
            par = os.path.dirname(d.rstrip('/\\'))
            self.lbl_output.setText('→  {}  (next run)'.format(
                os.path.join(par, '{}_recolored_<ts>_v{}'.format(src, self.session_run+1))))
            self._log('📁 Source selected: {}'.format(d))
            self._sync()

    def _start(self):
        if not self.directory:
            self._log('❌ No directory selected.', 'err'); return

        self.output_dir = self._make_out_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        self.lbl_output.setText('→  {}'.format(self.output_dir))

        self.flog.clear()
        self.progress.setValue(0); self.progress.setMaximum(1)
        self.progress.show()

        colors = [self.cbts[i].hex() for i in range(self.spin.value())]
        mono   = self.mono_btn.hex()
        n_workers = min((os.cpu_count() or 4)*2, 16)

        sep = '═'*58
        for t in (sep,
                  '  SVG RECOLOR TOOL  v5',
                  sep,
                  '  Source   : {}'.format(self.directory),
                  '  Output   : {}'.format(self.output_dir),
                  '  Workers  : {} threads'.format(n_workers),
                  '  Colours  : {}'.format(' → '.join(colors)),
                  '  Mono     : {}'.format(mono),
                  sep):
            self._log(t, 'sep')

        self.thread = ProcessThread(self.directory, self.output_dir, colors, mono)
        self.thread.file_done.connect(self._on_file_done)
        self.thread.progress.connect(self._on_progress)
        self.thread.finished.connect(self._on_done)
        self.thread.stopped.connect(self._on_stop)
        self.thread.start()
        self._sync()

    def _stop(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop(); self.thread.wait()
            self._log('🛑 Processing stopped.', 'err')

    def _on_progress(self, done, total):
        if total>0:
            self.progress.setMaximum(total); self.progress.setValue(done)

    def _on_done(self, total, ok):
        self.progress.hide()
        pct = ok/total*100 if total else 0
        self._log('')
        self._log('✅  Done: {} / {} recoloured  ({:.1f}%)'.format(ok, total, pct))
        if total>ok:
            self._log('⚠️   {} files failed — enable "Errors only" to review'.format(total-ok), 'err')
        self._log('📁  Output: {}'.format(self.output_dir))
        self._load_preview(); self._sync()

    def _on_stop(self):
        self.progress.hide()
        self._log('🛑  Partial output: {}'.format(self.output_dir), 'err')
        self._load_preview(); self._sync()

    def _open_output(self):
        if not (self.output_dir and os.path.exists(self.output_dir)):
            self._log('❌ Output not found.','err'); return
        try:
            if sys.platform=='win32': os.startfile(self.output_dir)
            elif sys.platform=='darwin': subprocess.Popen(['open', self.output_dir])
            else: subprocess.Popen(['xdg-open', self.output_dir])
        except Exception as e:
            self._log('⚠️ Cannot open folder: {}'.format(e),'err')

    def _delete_output(self):
        if not (self.output_dir and os.path.exists(self.output_dir)):
            self._log('❌ Output not found.','err'); return
        if QMessageBox.question(
            self, 'Delete Output',
            'Delete output folder?\n\n{}\n\nOriginals are not affected.'.format(self.output_dir),
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes: return
        shutil.rmtree(self.output_dir)
        self._log('🗑  Deleted: {}'.format(self.output_dir))
        self.output_dir = None; self.preview_list = []; self._clear_preview()
        self._sync()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if sys.platform=='linux':
        os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        os.environ.pop('QT_QPA_PLATFORM', None)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(_APP_QSS)
    win = SVGRecolorGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
