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
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QGroupBox, QProgressBar, QColorDialog, QGridLayout,
    QSpinBox, QMessageBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QTimer, QPoint
from PyQt6.QtGui import QPalette, QColor, QPainter, QLinearGradient, QPen, QBrush, QPolygon, QFontMetrics
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


def build_gradient(base, n, positions=None):
    """
    base      : list of hex colours, in stop order
    n         : how many output colours to produce (uniformly sampled in t)
    positions : optional list, same length as `base`, giving each base
                colour's position in [0,1]. If omitted, stops are assumed
                evenly spaced (old behaviour, kept for compatibility).
    """
    k = len(base)
    if n <= 0: return []
    if n == 1: return [base[0]]

    if positions is None:
        if n <= k: return base[:n]
        positions = [i/(k-1) for i in range(k)] if k > 1 else [0.0]

    pairs = sorted(zip(positions, base))
    pos   = [p for p, _ in pairs]
    cols  = [c for _, c in pairs]

    res = []
    for i in range(n):
        t = i/(n-1)
        if t <= pos[0]:
            res.append(cols[0]); continue
        if t >= pos[-1]:
            res.append(cols[-1]); continue
        seg = 0
        while seg < len(pos)-2 and t > pos[seg+1]:
            seg += 1
        span = pos[seg+1] - pos[seg]
        local_t = (t - pos[seg]) / span if span > 0 else 0.0
        res.append(interpolate(cols[seg], cols[seg+1], local_t))
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


def _recolor_file(src_path, dst_path, colors, mono_color, positions=None):
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
            u = sorted(nset); g = build_gradient(colors, len(u), positions=positions)
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

    def __init__(self, src_dir, out_dir, colors, mono_color, positions=None):
        super().__init__()
        self.src_dir    = src_dir
        self.out_dir    = out_dir
        self.colors     = colors
        self.mono_color = mono_color
        self.positions  = positions   # stop positions [0..1], same order as colors
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

        # ── Phase 3: process pool with sliding window ──────────────────────
        # CPU-bound regex work: real parallelism needs separate processes
        # (threads are GIL-serialized for this workload). Oversubscribing
        # process count past core count just adds spawn/pickle overhead.
        n_workers = max(1, min(os.cpu_count() or 4, 16))
        WINDOW    = max(n_workers * 8, 128)

        done_count = ok_count = 0
        job_idx    = 0
        in_flight  = {}

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            while job_idx < total and len(in_flight) < WINDOW:
                src, dst = all_jobs[job_idx]
                f = ex.submit(_recolor_file, src, dst, self.colors, self.mono_color,
                              self.positions)
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
                                       self.colors, self.mono_color, self.positions)
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

    def set_color(self, color: QColor):
        self._color = color
        p = self.palette()
        mid   = p.color(QPalette.ColorRole.Mid).name()
        light = p.color(QPalette.ColorRole.Light).name()
        self.setStyleSheet(
            'QPushButton{{background:{c};border:2px solid {m};border-radius:23px;}}'
            'QPushButton:hover{{border:2px solid {l};border-radius:23px;}}'
            .format(c=color.name(), m=mid, l=light))
        self.setToolTip('{}: {}'.format(self._label, color.name()))

    def _pick(self):
        c = QColorDialog.getColor(self._color, self, 'Choose Colour',
                                  QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if c.isValid(): self.set_color(c)

    def hex(self): return self._color.name()


# ══════════════════════════════════════════════════════════════════════════════
#  Blender-style gradient color ramp widget
# ══════════════════════════════════════════════════════════════════════════════

class GradientRamp(QWidget):
    """
    Blender-style gradient ramp.

    Layout (top → bottom):
      [gradient bar]          ← RAMP_H px tall
      [upward triangles]      ← HDL_H px, tip touches the bar bottom edge
      [hex labels]            ← LBL_H px

    Behaviour:
    • Drag a handle to slide its stop position (first/last stops are locked).
    • Single-click (no drag) opens the colour picker for that stop.
    """

    # ── geometry constants ────────────────────────────────────────────────
    _RAMP_H = 30
    _HDL_H  = 16   # triangle height (tip at ramp bottom, base below)
    _HDL_W  = 10   # half-width of triangle base
    _LBL_H  = 18
    _PAD    = 20   # left/right padding so edge handles fit

    def __init__(self, color_buttons: list, parent=None):
        super().__init__(parent)
        self._btns    = color_buttons
        self._active  = 2
        # Normalised positions [0.0 … 1.0] for each stop
        self._pos     = [0.0, 1.0, 0.5, 0.5, 0.5, 0.5]

        h = self._RAMP_H + self._HDL_H + self._LBL_H + 10
        self.setMinimumHeight(h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)

        self._hovered  = -1   # index of hovered stop
        self._dragging = -1   # index of stop being dragged
        self._drag_x0  = 0    # mouse x at drag start
        self._drag_t0  = 0.0  # stop t at drag start
        self._moved    = False  # did we actually move during this press?

        for b in self._btns:
            b.clicked.connect(self.update)

    # ── public API ────────────────────────────────────────────────────────

    def set_active(self, n: int):
        # Only re-distribute stops evenly when the *count* actually changes.
        # Calling this repeatedly with the same n (e.g. redundant UI syncs)
        # must NOT wipe out positions the user already dragged.
        if n == self._active:
            return
        self._active = n
        if n > 1:
            for i in range(n):
                self._pos[i] = i / (n - 1)
        elif n == 1:
            self._pos[0] = 0.0
        self.update()

    # ── coordinate helpers ────────────────────────────────────────────────

    def _ramp_x(self)  -> int:  return self._PAD
    def _ramp_w(self)  -> int:  return max(1, self.width() - 2 * self._PAD)
    def _ramp_y(self)  -> int:  return 4
    def _tip_y(self)   -> int:  return self._ramp_y() + self._RAMP_H  # tip of ▲ touches here
    def _base_y(self)  -> int:  return self._tip_y() + self._HDL_H
    def _lbl_y(self)   -> int:  return self._base_y() + 2

    def _t_to_x(self, t: float) -> int:
        return self._ramp_x() + round(t * self._ramp_w())

    def _x_to_t(self, x: int) -> float:
        t = (x - self._ramp_x()) / self._ramp_w()
        return max(0.0, min(1.0, t))

    def _sorted_stops(self):
        """Return list of (t, index) sorted by position."""
        n = self._active
        return sorted((self._pos[i], i) for i in range(n))

    def _hit(self, px: float) -> int:
        """Return index of stop whose handle contains pixel px, else -1."""
        for i in range(self._active):
            if abs(self._t_to_x(self._pos[i]) - px) <= self._HDL_W + 2:
                return i
        return -1

    # ── paint ─────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rx = self._ramp_x();  rw = self._ramp_w()
        ry = self._ramp_y();  rh = self._RAMP_H
        tip_y  = self._tip_y()
        base_y = self._base_y()
        lbl_y  = self._lbl_y()
        n = self._active

        # ── gradient bar ─────────────────────────────────────────────────
        grad = QLinearGradient(rx, 0, rx + rw, 0)
        stops = self._sorted_stops()
        for t, i in stops:
            grad.setColorAt(t, QColor(self._btns[i].hex()))

        pal      = self.palette()
        base_bg  = pal.color(QPalette.ColorRole.Base)
        border_c = pal.color(QPalette.ColorRole.Mid)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(base_bg))
        p.drawRoundedRect(rx, ry, rw, rh, 4, 4)
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(rx, ry, rw, rh, 4, 4)
        p.setPen(QPen(border_c, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rx, ry, rw, rh, 4, 4)

        # ── handles (upward triangles: tip at ramp bottom, base below) ───
        font = self.font(); font.setPointSize(8); p.setFont(font)
        fm   = QFontMetrics(font)
        hw   = self._HDL_W

        for i in range(n):
            x   = self._t_to_x(self._pos[i])
            col = QColor(self._btns[i].hex())

            # ▲ tip touches the bar; base is below
            tip  = QPoint(x,      tip_y)
            lft  = QPoint(x - hw, base_y)
            rgt  = QPoint(x + hw, base_y)
            poly = QPolygon([tip, lft, rgt])

            is_sel  = (i == self._dragging)
            is_hov  = (i == self._hovered)
            win_txt = pal.color(QPalette.ColorRole.WindowText)
            outline = (win_txt if is_sel else
                       _mix(win_txt, base_bg, 0.2) if is_hov else
                       _mix(win_txt, base_bg, 0.45))

            p.setPen(QPen(outline, 1.5))
            p.setBrush(QBrush(col))
            p.drawPolygon(poly)

            # thin vertical stem from tip up to bottom of bar
            p.setPen(QPen(outline, 1))
            p.drawLine(x, ry + rh - 1, x, tip_y)

            # hex label
            label = col.name().upper()
            lw    = fm.horizontalAdvance(label)
            lx    = max(0, min(x - lw // 2, self.width() - lw))
            lbl_col = (_mix(win_txt, base_bg, 0.15) if (is_hov or is_sel)
                       else _mix(win_txt, base_bg, 0.4))
            p.setPen(lbl_col)
            p.drawText(lx, lbl_y + fm.ascent(), label)

        p.end()

    # ── mouse events ──────────────────────────────────────────────────────

    def _mouse_x(self, event) -> float:
        return event.position().x() if hasattr(event, 'position') else float(event.x())

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px = self._mouse_x(event)
        hit = self._hit(px)
        if hit >= 0:
            self._dragging = hit
            self._drag_x0  = px
            self._drag_t0  = self._pos[hit]
            self._moved    = False
            self.update()

    def mouseMoveEvent(self, event):
        px = self._mouse_x(event)

        if self._dragging >= 0:
            dx = px - self._drag_x0
            # Only start moving if cursor has travelled > 3px
            if abs(dx) > 3 or self._moved:
                self._moved = True
                i = self._dragging
                # First and last stops are locked to 0 and 1
                if i == 0:
                    pass
                elif i == self._active - 1:
                    pass
                else:
                    new_t = self._drag_t0 + dx / self._ramp_w()
                    self._pos[i] = max(0.01, min(0.99, new_t))
                    self.update()
            return

        # Hover detection
        prev = self._hovered
        self._hovered = self._hit(px)
        if self._hovered != prev:
            self.update()
        self.setCursor(
            Qt.CursorShape.SizeHorCursor if self._hovered >= 0
            else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        i = self._dragging
        if i >= 0 and not self._moved:
            # Click without drag → open colour picker
            self._btns[i]._pick()
            self.update()
        self._dragging = -1
        self._moved    = False
        self.update()

    def mouseDoubleClickEvent(self, event):
        # Double-click also opens colour picker (convenience)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        px  = self._mouse_x(event)
        hit = self._hit(px)
        if hit >= 0:
            self._btns[hit]._pick()
            self.update()

    def leaveEvent(self, event):
        self._hovered = -1
        self.update()
        self.setCursor(Qt.CursorShape.ArrowCursor)


# ══════════════════════════════════════════════════════════════════════════════
#  Filtered log widget
# ══════════════════════════════════════════════════════════════════════════════

# HTML colour for each category
def _cat_colors(palette: QPalette) -> dict:
    """Derive log category colours from the active palette."""
    is_dark = palette.color(QPalette.ColorRole.Window).value() < 128
    if is_dark:
        return {'ok': '#55cc66', 'err': '#ff5555', 'info': '#888888', 'sep': '#555566'}
    else:
        return {'ok': '#1a7a30', 'err': '#cc2222', 'info': '#555555', 'sep': '#8888aa'}

_MAX_DISPLAY = 30_000


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
        layout.addWidget(self._view)
        self._apply_log_style()

        # debounce filter changes
        self._rebuild_timer = QTimer()
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(120)
        self._rebuild_timer.timeout.connect(self._rebuild)

        self._errors_only.stateChanged.connect(self._rebuild_timer.start)

    # ── public API ─────────────────────────────────────────────────────────

    def _apply_log_style(self):
        p   = self.palette()
        bg  = _mix(p.color(QPalette.ColorRole.Base),
                   p.color(QPalette.ColorRole.Window), 0.7)
        bdr = p.color(QPalette.ColorRole.Mid)
        self._view.setStyleSheet(
            'QTextEdit{{background:{bg};border:1px solid {bdr};border-radius:5px;'
            'font-family:Consolas,"Courier New",monospace;font-size:11px;padding:6px;}}'
            .format(bg=bg.name(), bdr=bdr.name()))

    def changeEvent(self, event):
        super().changeEvent(event)
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.PaletteChange:
            self._apply_log_style()
            self._rebuild()

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
        colors = _cat_colors(self.palette())
        color  = colors.get(category, colors['info'])
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
            muted = _cat_colors(self.palette())['info']
            html_parts.append(
                '<span style="color:{}">… earlier entries hidden '
                '(showing last {:,})</span>'.format(muted, _MAX_DISPLAY))

        for cat, txt in visible:
            html_parts.append(self._to_html(cat, txt))

        self._view.setHtml('<br>'.join(html_parts))
        sb = self._view.verticalScrollBar()
        sb.setValue(sb.maximum())


# ══════════════════════════════════════════════════════════════════════════════
#  Main window
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  Palette-driven theming  (follows the system/Qt theme automatically)
# ══════════════════════════════════════════════════════════════════════════════

def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear interpolate between two QColors (t=0 → a, t=1 → b)."""
    return QColor(
        int(a.red()   + (b.red()   - a.red())   * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue()  + (b.blue()  - a.blue())  * t),
    )

def _lighter(c: QColor, factor: float) -> QColor:
    return QColor.fromHsvF(
        c.hsvHueF(),
        max(0.0, c.hsvSaturationF() * (1 - factor * 0.3)),
        min(1.0, c.valueF() + factor * 0.25),
    )

def _col(c: QColor) -> str:
    return c.name()

def build_qss(palette: QPalette) -> str:
    """
    Generate a QSS sheet derived entirely from the system QPalette.
    Structural rules only — all colours come from palette roles.
    """
    base    = palette.color(QPalette.ColorRole.Base)
    alt     = palette.color(QPalette.ColorRole.AlternateBase)
    window  = palette.color(QPalette.ColorRole.Window)
    btn_bg  = palette.color(QPalette.ColorRole.Button)
    btn_txt = palette.color(QPalette.ColorRole.ButtonText)
    txt     = palette.color(QPalette.ColorRole.Text)
    win_txt = palette.color(QPalette.ColorRole.WindowText)
    hi      = palette.color(QPalette.ColorRole.Highlight)
    hi_txt  = palette.color(QPalette.ColorRole.HighlightedText)
    mid     = palette.color(QPalette.ColorRole.Mid)
    dark    = palette.color(QPalette.ColorRole.Dark)
    shadow  = palette.color(QPalette.ColorRole.Shadow)
    light   = palette.color(QPalette.ColorRole.Light)
    dis_txt = palette.color(QPalette.ColorGroup.Disabled,
                             QPalette.ColorRole.ButtonText)

    # Derived tones
    btn_hover   = _mix(btn_bg, light, 0.25)
    btn_pressed = _mix(btn_bg, dark,  0.35)
    border      = _mix(btn_bg, dark,  0.6)
    border_foc  = hi
    input_bg    = _mix(base, window, 0.4)
    scroll_bg   = _mix(window, dark, 0.3)
    scroll_hdl  = _mix(mid, dark, 0.3)

    # Action-button accent colours: derive from highlight, staying readable
    is_dark = window.value() < 128

    def accent(h: int, s: float, v_dark: float, v_light: float) -> tuple:
        """Return (border, text, disabled_border, disabled_text) for an action btn."""
        v  = v_dark if is_dark else v_light
        vd = max(0.0, v - 0.25)
        c_border = QColor.fromHsvF(h/360, s, v)
        c_text   = QColor.fromHsvF(h/360, s * 0.5, min(1.0, v + 0.35))
        c_dbdr   = QColor.fromHsvF(h/360, s * 0.4, vd)
        c_dtxt   = QColor.fromHsvF(h/360, s * 0.3, vd + 0.1)
        return _col(c_border), _col(c_text), _col(c_dbdr), _col(c_dtxt)

    g_bdr, g_txt, g_dbdr, g_dtxt = accent(140, 0.65, 0.40, 0.35)  # green  start
    y_bdr, y_txt, y_dbdr, y_dtxt = accent( 38, 0.70, 0.45, 0.40)  # yellow stop
    b_bdr, b_txt, b_dbdr, b_dtxt = accent(210, 0.65, 0.45, 0.38)  # blue   open
    r_bdr, r_txt, r_dbdr, r_dtxt = accent(  0, 0.65, 0.40, 0.35)  # red    delete
    p_bdr, p_txt, p_dbdr, p_dtxt = accent(275, 0.55, 0.42, 0.38)  # purple install-root
    c_bdr, c_txt, c_dbdr, c_dtxt = accent(210, 0.55, 0.42, 0.38)  # cyan   install-user

    hover_bg = _col(btn_hover)
    arr_col  = _col(_mix(win_txt, btn_bg, 0.3))
    arr_dis  = _col(dis_txt)

    # Muted / accent tones for informational labels & preview panel
    muted_txt   = _col(_mix(win_txt, window, 0.5))
    accent_txt  = _col(_mix(hi, win_txt, 0.35))
    preview_bg  = _col(_mix(base, shadow, 0.15 if is_dark else 0.05))
    preview_bdr = _col(_mix(border, shadow, 0.2))

    return f"""
QWidget {{
    background-color: {_col(window)};
    color: {_col(win_txt)};
    font-size: 13px;
    font-weight: 600;
}}
QLabel {{
    color: {_col(win_txt)};
    font-size: 13px;
    font-weight: 600;
}}
QGroupBox {{
    border: 1px solid {_col(border)};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 4px;
    font-size: 12px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {_col(_mix(win_txt, btn_bg, 0.2))};
    font-weight: 700;
}}

QPushButton {{
    background-color: {_col(btn_bg)};
    color: {_col(btn_txt)};
    border: 1px solid {_col(border)};
    border-radius: 5px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover    {{ background-color: {hover_bg}; border-color: {_col(light)}; }}
QPushButton:pressed  {{ background-color: {_col(btn_pressed)}; }}
QPushButton:disabled {{ color: {_col(dis_txt)}; border-color: {_col(_mix(border, window, 0.6))}; background: {_col(_mix(btn_bg, window, 0.5))}; }}

QPushButton#btn_start         {{ border-color: {g_bdr}; color: {g_txt}; }}
QPushButton#btn_start:hover   {{ border-color: {_col(_lighter(QColor(g_bdr), 0.3))}; background: {hover_bg}; }}
QPushButton#btn_start:disabled{{ border-color: {g_dbdr}; color: {g_dtxt}; }}

QPushButton#btn_stop          {{ border-color: {y_bdr}; color: {y_txt}; }}
QPushButton#btn_stop:hover    {{ border-color: {_col(_lighter(QColor(y_bdr), 0.3))}; background: {hover_bg}; }}

QPushButton#btn_open          {{ border-color: {b_bdr}; color: {b_txt}; }}
QPushButton#btn_open:hover    {{ border-color: {_col(_lighter(QColor(b_bdr), 0.3))}; background: {hover_bg}; }}
QPushButton#btn_open:disabled {{ border-color: {b_dbdr}; color: {b_dtxt}; }}

QPushButton#btn_delete        {{ border-color: {r_bdr}; color: {r_txt}; }}
QPushButton#btn_delete:hover  {{ border-color: {_col(_lighter(QColor(r_bdr), 0.3))}; background: {hover_bg}; }}
QPushButton#btn_delete:disabled{{ border-color: {r_dbdr}; color: {r_dtxt}; }}

QPushButton#btn_install_root        {{ border-color: {p_bdr}; color: {p_txt}; }}
QPushButton#btn_install_root:hover  {{ border-color: {_col(_lighter(QColor(p_bdr), 0.3))}; background: {hover_bg}; }}
QPushButton#btn_install_root:disabled{{ border-color: {p_dbdr}; color: {p_dtxt}; }}

QPushButton#btn_install_user        {{ border-color: {c_bdr}; color: {c_txt}; }}
QPushButton#btn_install_user:hover  {{ border-color: {_col(_lighter(QColor(c_bdr), 0.3))}; background: {hover_bg}; }}
QPushButton#btn_install_user:disabled{{ border-color: {c_dbdr}; color: {c_dtxt}; }}

QLineEdit, QSpinBox {{
    background-color: {_col(input_bg)};
    border: 1px solid {_col(border)};
    border-radius: 4px;
    color: {_col(txt)};
    font-size: 13px;
    font-weight: 600;
    padding: 4px 8px;
    selection-background-color: {_col(hi)};
    selection-color: {_col(hi_txt)};
}}
QLineEdit:focus, QSpinBox:focus {{ border-color: {_col(border_foc)}; }}

QLineEdit#search_box {{
    background-color: {preview_bg};
    border: 1px solid {_col(border)};
    border-radius: 4px;
    color: {_col(txt)};
    padding: 5px 8px;
    font-size: 12px;
}}
QLineEdit#search_box:focus {{ border-color: {_col(border_foc)}; }}

QSvgWidget#svg_preview {{
    background: {preview_bg};
    border: 1px solid {preview_bdr};
    border-radius: 7px;
}}

QLabel#lbl_muted  {{ color: {muted_txt}; font-size: 11px; font-weight: 600; }}
QLabel#lbl_accent {{ color: {accent_txt}; font-size: 11px; font-weight: 600; }}

QLabel#hint_box {{
    color: {muted_txt};
    font-style: italic;
    font-size: 11px;
    font-weight: 500;
    background: {preview_bg};
    padding: 10px;
    border-radius: 6px;
    border: 1px solid {preview_bdr};
}}

QSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right;
    width: 20px; height: 13px;
    background: {_col(btn_bg)};
    border: none;
    border-left: 1px solid {_col(border)};
    border-bottom: 1px solid {_col(border)};
    border-top-right-radius: 4px;
}}
QSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 20px; height: 13px;
    background: {_col(btn_bg)};
    border: none;
    border-left: 1px solid {_col(border)};
    border-top: 1px solid {_col(border)};
    border-bottom-right-radius: 4px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {hover_bg}; }}
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{ background: {_col(btn_pressed)}; }}
QSpinBox::up-arrow {{
    image: none; width: 0; height: 0;
    border-style: solid; border-width: 0 4px 6px 4px;
    border-color: transparent transparent {arr_col} transparent;
}}
QSpinBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-style: solid; border-width: 6px 4px 0 4px;
    border-color: {arr_col} transparent transparent transparent;
}}
QSpinBox::up-arrow:disabled   {{ border-bottom-color: {arr_dis}; }}
QSpinBox::down-arrow:disabled {{ border-top-color:    {arr_dis}; }}

QProgressBar {{
    background-color: {_col(input_bg)};
    border: 1px solid {_col(border)};
    border-radius: 4px;
    color: {_col(_mix(win_txt, window, 0.4))};
    text-align: center;
    font-size: 11px;
}}
QProgressBar::chunk {{
    background-color: {g_bdr};
    border-radius: 3px;
}}

QCheckBox {{ color: {_col(win_txt)}; spacing: 6px; font-size: 13px; font-weight: 600; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {_col(border)}; border-radius: 3px;
    background: {_col(input_bg)};
}}
QCheckBox::indicator:checked {{
    background: {g_bdr}; border-color: {g_txt};
}}
QCheckBox::indicator:hover {{ border-color: {_col(light)}; }}

QScrollBar:vertical {{
    background: {_col(scroll_bg)}; width: 7px; border-radius: 4px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {_col(scroll_hdl)}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {_col(_mix(scroll_hdl, light, 0.3))}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QMessageBox {{ background: {_col(window)}; }}
QMessageBox QLabel {{ color: {_col(win_txt)}; }}
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
        self.svg_w.setObjectName('svg_preview')
        pl.addWidget(self.svg_w, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_fname = QLabel('No preview')
        self.lbl_fname.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_fname.setWordWrap(True)
        self.lbl_fname.setObjectName('lbl_muted')
        pl.addWidget(self.lbl_fname)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton('◀')
        self.btn_prev.setFixedSize(58, 32); self.btn_prev.clicked.connect(self._prev)
        nav.addWidget(self.btn_prev)
        self.lbl_count = QLabel('0 / 0')
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setObjectName('lbl_muted')
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
        search_lbl.setObjectName('lbl_muted')
        search_lay.addWidget(search_lbl)
        self.search_box = QLineEdit()
        self.search_box.setObjectName('search_box')
        self.search_box.setPlaceholderText('Search files — filters preview & log…')
        self.search_box.setClearButtonEnabled(True)
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
        self.lbl_output.setObjectName('lbl_accent')
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

        # Gradient color ramp (Blender-style)
        gg = QGroupBox('Gradient Colours  (multi-colour icons — click a stop to change)')
        ggl = QVBoxLayout(gg); ggl.setSpacing(8)

        LABELS   = ['Color 1','Color 2','Color 3','Color 4','Color 5','Color 6']
        DEFAULTS = ['#000000','#464646','#D81C4A','#b6b6b6','#ffffff','#888888']
        self.cbts = []
        for i in range(6):
            b = ColorButton(DEFAULTS[i], LABELS[i])
            b.setVisible(False)      # hidden — ramp handles are the UI
            self.cbts.append(b)
            ggl.addWidget(b)         # keep in layout for signals; hidden

        self.grad_ramp = GradientRamp(self.cbts)
        # Wire color changes from all buttons to repaint the ramp
        for b in self.cbts:
            # Monkey-patch set_color so ramp repaints after any pick
            orig_set = b.set_color
            def make_set(btn, orig):
                def set_and_repaint(color):
                    orig(color)
                    self.grad_ramp.update()
                return set_and_repaint
            b.set_color = make_set(b, orig_set)

        ggl.addWidget(self.grad_ramp)
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

        self.btn_install_root = QPushButton('🔒  Install for Root  (/usr/share/icons/…)')
        self.btn_install_root.setObjectName('btn_install_root')
        self.btn_install_root.setFixedHeight(42)
        self.btn_install_root.setEnabled(False)
        self.btn_install_root.clicked.connect(self._install_root)
        al.addWidget(self.btn_install_root, 3, 0)

        self.btn_install_user = QPushButton('👤  Install for User  (~/.local/share/icons/…)')
        self.btn_install_user.setObjectName('btn_install_user')
        self.btn_install_user.setFixedHeight(42)
        self.btn_install_user.setEnabled(False)
        self.btn_install_user.clicked.connect(self._install_user)
        al.addWidget(self.btn_install_user, 3, 1)

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
        hint.setObjectName('hint_box')
        rv.addWidget(hint)

        rv.addStretch()
        rh.addWidget(right)

        self._refresh_cbts()

    def _refresh_cbts(self):
        n = self.spin.value()
        for i, b in enumerate(self.cbts):
            b.setEnabled(i < n)
        self.grad_ramp.set_active(n)

    # ── state ──────────────────────────────────────────────────────────────

    def _sync(self):
        busy    = self.progress.isVisible()
        has_out = bool(self.output_dir and os.path.exists(self.output_dir))
        self.btn_start.setEnabled(not busy and bool(self.directory))
        self.btn_stop.setEnabled(busy)
        self.btn_open.setEnabled(not busy and has_out)
        self.btn_delete.setEnabled(not busy and has_out)
        self.btn_install_root.setEnabled(not busy and has_out)
        self.btn_install_user.setEnabled(not busy and has_out)
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

    # ── install ────────────────────────────────────────────────────────────

    def _theme_name(self):
        """Original source folder name — used as install target name."""
        if self.directory:
            return os.path.basename(self.directory.rstrip('/\\'))
        return os.path.basename(self.output_dir) if self.output_dir else 'icons'

    def _find_theme_roots(self, folder: str) -> list:
        """
        Return the list of directories that should be installed as icon themes.
        A 'theme root' is a directory that directly contains size-dirs (16x16,
        22x22, 24x24, …) or an index.theme file.
        If the output folder itself is a theme root → [folder].
        Otherwise look one level deeper (handles zips where the top dir is a
        release name containing Papirus/, Papirus-Dark/, etc.).
        """
        import re
        size_re = re.compile(r'^\d+x\d+$')

        def is_theme_root(d):
            try:
                entries = os.listdir(d)
            except OSError:
                return False
            return (
                'index.theme' in entries or
                any(size_re.match(e) and os.path.isdir(os.path.join(d, e))
                    for e in entries)
            )

        if is_theme_root(folder):
            return [folder]

        # One level down
        roots = []
        try:
            for name in sorted(os.listdir(folder)):
                sub = os.path.join(folder, name)
                if os.path.isdir(sub) and is_theme_root(sub):
                    roots.append(sub)
        except OSError:
            pass
        return roots or [folder]   # fallback

    def _install(self, base_dir: str, root: bool):
        if not (self.output_dir and os.path.exists(self.output_dir)):
            self._log('❌ Output folder not found.', 'err'); return

        theme_roots = self._find_theme_roots(self.output_dir)
        default_name = self._theme_name()

        # Build (src_dir, dest_dir, theme_name) list
        installs = []
        for tr in theme_roots:
            if tr == self.output_dir:
                name = default_name
            else:
                name = os.path.basename(tr)
            installs.append((tr, os.path.join(base_dir, name), name))

        # Confirm
        dest_list = '\n'.join(d for _, d, _ in installs)
        confirm = QMessageBox.question(
            self, 'Install Icon Theme',
            'Install {} theme(s) to:\n\n{}'.format(len(installs), dest_list),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        # Overwrite warning for any that already exist
        existing = [d for _, d, _ in installs if os.path.exists(d)]
        if existing:
            warn = QMessageBox(self)
            warn.setWindowTitle('Theme Already Exists')
            warn.setIcon(QMessageBox.Icon.Warning)
            warn.setText(
                'The following theme(s) already exist and will be overwritten:\n\n'
                + '\n'.join(existing))
            btn_cont = warn.addButton('Continue', QMessageBox.ButtonRole.AcceptRole)
            warn.addButton('Cancel', QMessageBox.ButtonRole.RejectRole)
            warn.exec()
            if warn.clickedButton() is not btn_cont:
                return

        if root:
            self._install_root_sudo(installs, base_dir)
        else:
            for src, dest, name in installs:
                try:
                    os.makedirs(base_dir, exist_ok=True)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(src, dest)
                    self._post_install(dest, name)
                except Exception as e:
                    self._log('❌ Install error ({}): {}'.format(name, e), 'err')

    def _install_root_sudo(self, installs: list, base_dir: str):
        from PyQt6.QtWidgets import QInputDialog, QLineEdit

        password, ok = QInputDialog.getText(
            self, 'Root Password Required',
            'Enter your sudo password to install to:\n{}'.format(base_dir),
            QLineEdit.EchoMode.Password,
        )
        if not ok or not password:
            self._log('⚠️  Root install cancelled.'); return

        for src, dest, name in installs:
            shell_cmd = "rm -rf '{d}' && mkdir -p '{d}' && cp -r '{s}/.' '{d}/'".format(
                s=src, d=dest)
            try:
                result = subprocess.run(
                    ['sudo', '-S', 'sh', '-c', shell_cmd],
                    input=password + '\n',
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    err = '\n'.join(
                        l for l in (result.stderr or '').splitlines()
                        if 'password' not in l.lower() and l.strip())
                    self._log('❌ Root install failed ({}): {}'.format(
                        name, err or 'sudo error'), 'err')
                    continue
                self._post_install(dest, name)
            except subprocess.TimeoutExpired:
                self._log('❌ Root install timed out ({}).'.format(name), 'err')
            except Exception as e:
                self._log('❌ Root install error ({}): {}'.format(name, e), 'err')

    def _post_install(self, dest: str, name: str):
        self._log('✅  Installed: {}'.format(dest))
        try:
            subprocess.run(
                ['gtk-update-icon-cache', '-f', '-t', dest],
                capture_output=True, check=False, timeout=10)
            self._log('   gtk-update-icon-cache: {}'.format(name))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _install_root(self):
        self._install('/usr/share/icons', root=True)

    def _install_user(self):
        user_icons = os.path.join(os.path.expanduser('~'), '.local', 'share', 'icons')
        self._install(user_icons, root=False)

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

        # Read colours AND their positions in ramp order (handles may have been dragged)
        sorted_stops = self.grad_ramp._sorted_stops()   # [(t, idx), ...] sorted by t
        colors    = [self.cbts[i].hex() for _, i in sorted_stops]
        positions = [t for t, _ in sorted_stops]
        mono      = self.mono_btn.hex()
        n_workers = min((os.cpu_count() or 4)*2, 16)

        sep = '═'*58
        for t in (sep,
                  '  SVG RECOLOR TOOL  v5',
                  sep,
                  '  Source   : {}'.format(self.directory),
                  '  Output   : {}'.format(self.output_dir),
                  '  Workers  : {} processes'.format(n_workers),
                  '  Colours  : {}'.format(' → '.join(colors)),
                  '  Mono     : {}'.format(mono),
                  sep):
            self._log(t, 'sep')

        self.thread = ProcessThread(self.directory, self.output_dir, colors, mono, positions)
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
    if sys.platform == 'linux':
        os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        os.environ.pop('QT_QPA_PLATFORM', None)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')   # consistent cross-distro widget shapes

    win = SVGRecolorGUI()

    def apply_theme():
        app.setStyleSheet(build_qss(app.palette()))
        # Custom-painted widgets read the palette directly in paintEvent;
        # force a repaint so they pick up the new colours immediately.
        win.grad_ramp.update()
        win.flog._rebuild()

    apply_theme()

    # Re-apply whenever the system palette/theme changes
    app.paletteChanged.connect(lambda _: apply_theme())

    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
