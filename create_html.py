#!/usr/bin/env python3
"""
Generate an HTML report comparing clear face vs masked face detection results.

Usage:
    python generate_detection_report.py --clear <clear_folder> --masked <masked_folder> --output report.html

The clear folder should contain images like: image001.jpg
The masked folder should contain images like: image001_aug0_masked.jpg
"""

import os
import sys
import argparse
import base64
from pathlib import Path
from PIL import Image


def get_image_resolution(image_path):
    """Return (width, height) of an image."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        return (0, 0)


def image_to_base64(image_path):
    """Convert an image file to a base64 data URI."""
    try:
        ext = Path(image_path).suffix.lower().lstrip(".")
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "bmp": "image/bmp",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        mime = mime_map.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"
    except Exception as e:
        return ""


def get_image_folder(base_folder):
    """
    Given a top-level folder that contains exactly one subfolder of images,
    return the path to that subfolder. Falls back to base_folder itself
    if no subfolders are found.
    """
    base = Path(base_folder)
    subfolders = [p for p in base.iterdir() if p.is_dir()]
    if len(subfolders) == 1:
        print(f"   ↳ Using subfolder: {subfolders[0]}")
        return subfolders[0]
    elif len(subfolders) > 1:
        print(f"⚠️  Multiple subfolders found in '{base}', using base folder directly.")
    return base


def find_pairs(clear_folder, masked_folder):
    """
    Find matching image pairs between the clear and masked folders.
    Each folder may contain one subfolder that holds the actual images.
    Returns a list of dicts: {base_name, clear_path, masked_path}
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    clear_folder  = get_image_folder(clear_folder)
    masked_folder = get_image_folder(masked_folder)

    # Build a lookup for masked files: stem -> full_path
    masked_lookup = {}
    for f in Path(masked_folder).iterdir():
        if f.suffix.lower() in image_exts:
            masked_lookup[f.stem] = f  # e.g. "image001_aug0_masked" -> path

    pairs = []
    for f in sorted(Path(clear_folder).iterdir()):
        if f.suffix.lower() not in image_exts:
            continue
        base_name = f.stem  # e.g. "image001"
        # Find corresponding masked file by searching for base_name + "*_masked"
        masked_path = None
        for masked_stem, masked_file in masked_lookup.items():
            # masked stem should start with base_name and end with "_masked"
            if masked_stem.startswith(base_name) and masked_stem.endswith("_masked"):
                masked_path = masked_file
                break

        if masked_path:
            pairs.append({
                "base_name": base_name,
                "clear_path": f,
                "masked_path": masked_path,
            })
        else:
            # Include clear-only entry
            pairs.append({
                "base_name": base_name,
                "clear_path": f,
                "masked_path": None,
            })

    return pairs


def generate_html(pairs, output_path, use_base64=False):
    """Generate the HTML report."""

    rows_html = []
    for idx, pair in enumerate(pairs, 1):
        base_name = pair["base_name"]
        clear_path = pair["clear_path"]
        masked_path = pair["masked_path"]

        # Resolution from clear image
        w, h = get_image_resolution(clear_path)
        resolution = f"{w} × {h}" if w else "N/A"

        # Image sources
        if use_base64:
            clear_src = image_to_base64(clear_path)
            masked_src = image_to_base64(masked_path) if masked_path else ""
        else:
            clear_src = str(clear_path.resolve())
            masked_src = str(masked_path.resolve()) if masked_path else ""

        clear_img_tag = f'<img src="{clear_src}" alt="{base_name}" />' if clear_src else "<span class='missing'>Not found</span>"
        masked_img_tag = f'<img src="{masked_src}" alt="{base_name}_masked" />' if masked_src else "<span class='missing'>Not found</span>"

        row_class = "odd" if idx % 2 == 0 else "even"
        rows_html.append(f"""
        <tr class="{row_class}">
            <td class="filename">{idx}<br><span>{base_name}</span></td>
            <td class="resolution">{resolution}</td>
            <td class="img-cell">{clear_img_tag}</td>
            <td class="img-cell">{masked_img_tag}</td>
        </tr>""")

    rows = "\n".join(rows_html)
    total = len(pairs)
    matched = sum(1 for p in pairs if p["masked_path"] is not None)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Face Detection Results Comparison</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0f1117;
            color: #e0e0e0;
            min-height: 100vh;
        }}

        header {{
            background: linear-gradient(135deg, #1a1d2e 0%, #16213e 50%, #0f3460 100%);
            padding: 28px 40px;
            border-bottom: 2px solid #e94560;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 16px;
        }}

        header h1 {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.5px;
        }}

        header h1 span {{
            color: #e94560;
        }}

        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .stat-box {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px 18px;
            text-align: center;
        }}

        .stat-box .num {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #e94560;
        }}

        .stat-box .label {{
            font-size: 0.72rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 2px;
        }}

        .controls {{
            padding: 16px 40px;
            background: #13161f;
            border-bottom: 1px solid #1e2130;
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .controls label {{
            font-size: 0.85rem;
            color: #888;
        }}

        .controls input[type="range"] {{
            width: 160px;
            accent-color: #e94560;
        }}

        #size-val {{
            font-size: 0.85rem;
            color: #e94560;
            font-weight: 600;
            min-width: 50px;
        }}

        .search-box {{
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .search-box input {{
            background: #1a1d2e;
            border: 1px solid #2a2d3e;
            border-radius: 6px;
            padding: 6px 12px;
            color: #e0e0e0;
            font-size: 0.85rem;
            outline: none;
            width: 220px;
            transition: border-color 0.2s;
        }}

        .search-box input:focus {{
            border-color: #e94560;
        }}

        .table-wrapper {{
            padding: 20px 40px 40px;
            overflow-x: auto;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
        }}

        thead tr {{
            background: #1a1d2e;
            border-bottom: 2px solid #e94560;
        }}

        thead th {{
            padding: 14px 16px;
            text-align: left;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #aab;
            white-space: nowrap;
        }}

        thead th:nth-child(3),
        thead th:nth-child(4) {{
            text-align: center;
        }}

        tr.even {{ background: #13161f; }}
        tr.odd  {{ background: #111420; }}

        tr:hover {{
            background: #1a1d2e !important;
        }}

        td {{
            padding: 12px 16px;
            vertical-align: middle;
            border-bottom: 1px solid #1e2130;
        }}

        td.filename {{
            font-size: 0.75rem;
            color: #666;
            white-space: nowrap;
            min-width: 60px;
        }}

        td.filename span {{
            display: block;
            color: #c8d0e0;
            font-size: 0.82rem;
            font-weight: 500;
            margin-top: 4px;
            word-break: break-all;
        }}

        td.resolution {{
            font-size: 0.85rem;
            color: #7c8aaa;
            white-space: nowrap;
            font-family: monospace;
        }}

        td.img-cell {{
            text-align: center;
            padding: 10px;
        }}

        td.img-cell img {{
            max-width: var(--img-size, 240px);
            max-height: var(--img-size, 240px);
            width: auto;
            height: auto;
            border-radius: 6px;
            border: 1px solid #2a2d3e;
            display: block;
            margin: 0 auto;
            cursor: zoom-in;
            transition: transform 0.15s, border-color 0.15s;
        }}

        td.img-cell img:hover {{
            border-color: #e94560;
            transform: scale(1.02);
        }}

        .missing {{
            color: #555;
            font-size: 0.8rem;
            font-style: italic;
        }}

        /* Lightbox */
        #lightbox {{
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            cursor: zoom-out;
        }}

        #lightbox.active {{
            display: flex;
        }}

        #lightbox img {{
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 8px;
            box-shadow: 0 0 60px rgba(0,0,0,0.8);
        }}

        #lightbox-close {{
            position: fixed;
            top: 20px;
            right: 28px;
            font-size: 2rem;
            color: #fff;
            cursor: pointer;
            line-height: 1;
            z-index: 1001;
        }}

        .hidden {{ display: none !important; }}

        @media (max-width: 768px) {{
            header, .controls, .table-wrapper {{ padding-left: 16px; padding-right: 16px; }}
        }}
    </style>
</head>
<body>

<header>
    <h1>Face Detection <span>Results</span> Comparison</h1>
    <div class="stats">
        <div class="stat-box"><div class="num">{total}</div><div class="label">Total Images</div></div>
        <div class="stat-box"><div class="num">{matched}</div><div class="label">Matched Pairs</div></div>
        <div class="stat-box"><div class="num">{total - matched}</div><div class="label">Unmatched</div></div>
    </div>
</header>

<div class="controls">
    <label for="img-size-slider">Image Size:</label>
    <input type="range" id="img-size-slider" min="80" max="600" value="240" step="10" />
    <span id="size-val">240px</span>

    <div class="search-box">
        <label for="search-input">🔍 Filter:</label>
        <input type="text" id="search-input" placeholder="Search filename..." />
    </div>
</div>

<div class="table-wrapper">
    <table id="main-table">
        <thead>
            <tr>
                <th>#&nbsp;&nbsp;File Name</th>
                <th>Resolution</th>
                <th>Clear Face Output</th>
                <th>Masked Face Output</th>
            </tr>
        </thead>
        <tbody id="table-body">
            {rows}
        </tbody>
    </table>
</div>

<!-- Lightbox -->
<div id="lightbox">
    <span id="lightbox-close">&#x2715;</span>
    <img id="lightbox-img" src="" alt="Enlarged view" />
</div>

<script>
    // Image size slider
    const slider = document.getElementById('img-size-slider');
    const sizeVal = document.getElementById('size-val');
    const root = document.documentElement;

    slider.addEventListener('input', () => {{
        const v = slider.value + 'px';
        sizeVal.textContent = slider.value + 'px';
        root.style.setProperty('--img-size', v);
    }});
    root.style.setProperty('--img-size', '240px');

    // Search / filter
    const searchInput = document.getElementById('search-input');
    const tableBody = document.getElementById('table-body');

    searchInput.addEventListener('input', () => {{
        const q = searchInput.value.toLowerCase();
        const rows = tableBody.querySelectorAll('tr');
        rows.forEach(row => {{
            const text = row.querySelector('.filename span')?.textContent.toLowerCase() || '';
            row.classList.toggle('hidden', q !== '' && !text.includes(q));
        }});
    }});

    // Lightbox
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const lightboxClose = document.getElementById('lightbox-close');

    document.querySelectorAll('td.img-cell img').forEach(img => {{
        img.addEventListener('click', () => {{
            lightboxImg.src = img.src;
            lightbox.classList.add('active');
        }});
    }});

    lightbox.addEventListener('click', (e) => {{
        if (e.target !== lightboxImg) lightbox.classList.remove('active');
    }});
    lightboxClose.addEventListener('click', () => lightbox.classList.remove('active'));
    document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') lightbox.classList.remove('active');
    }});
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report saved to: {output_path}")
    print(f"   Total entries : {total}")
    print(f"   Matched pairs : {matched}")


def main():
    parser = argparse.ArgumentParser(description="Generate face detection comparison HTML report.")
    parser.add_argument("--clear",   required=True, help="Path to the clear face detection folder")
    parser.add_argument("--masked",  required=True, help="Path to the masked face detection folder")
    parser.add_argument("--output",  default="detection_report.html", help="Output HTML file path (default: detection_report.html)")
    parser.add_argument("--embed",   action="store_true",
                        help="Embed images as base64 inside HTML (portable but large file). "
                             "By default, images are referenced by absolute path.")
    args = parser.parse_args()

    clear_folder = Path(args.clear)
    masked_folder = Path(args.masked)

    if not clear_folder.is_dir():
        print(f"❌ Clear folder not found: {clear_folder}")
        sys.exit(1)
    if not masked_folder.is_dir():
        print(f"❌ Masked folder not found: {masked_folder}")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("⚠️  Pillow not installed. Installing...")
        os.system(f"{sys.executable} -m pip install Pillow --break-system-packages -q")
        from PIL import Image

    print(f"🔍 Scanning:\n   Clear  → {clear_folder}\n   Masked → {masked_folder}")
    pairs = find_pairs(clear_folder, masked_folder)

    if not pairs:
        print("❌ No image pairs found. Check your folder paths.")
        sys.exit(1)

    generate_html(pairs, args.output, use_base64=args.embed)


if __name__ == "__main__":
    main()