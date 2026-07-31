from pathlib import Path

MM = 72 / 25.4
PAGE_W, PAGE_H = 210 * MM, 297 * MM
SQUARE = 40 * MM
RADIUS = 12.5 * MM
LINE_W = 0.3 * MM
COLS, ROWS = 4, 6
GAP = 5 * MM
GRID_W = COLS * SQUARE + (COLS - 1) * GAP
GRID_H = ROWS * SQUARE + (ROWS - 1) * GAP
GRID_LEFT = (PAGE_W - GRID_W) / 2
GRID_BOTTOM = (PAGE_H - GRID_H) / 2

# Cubic Bézier approximation to a circle. This keeps every element vector-based.
K = 0.5522847498307936
commands = ["q", "0 G", f"{LINE_W:.6f} w"]
for row in range(ROWS):
    for col in range(COLS):
        left = GRID_LEFT + col * (SQUARE + GAP)
        bottom = GRID_BOTTOM + row * (SQUARE + GAP)
        right, top = left + SQUARE, bottom + SQUARE
        cx, cy = (left + right) / 2, (bottom + top) / 2
        commands.extend([
            f"{left:.6f} {bottom:.6f} {SQUARE:.6f} {SQUARE:.6f} re S",
            f"{cx:.6f} {bottom:.6f} m {cx:.6f} {top:.6f} l S",
            f"{left:.6f} {cy:.6f} m {right:.6f} {cy:.6f} l S",
            f"{cx + RADIUS:.6f} {cy:.6f} m",
            f"{cx + RADIUS:.6f} {cy + K * RADIUS:.6f} {cx + K * RADIUS:.6f} {cy + RADIUS:.6f} {cx:.6f} {cy + RADIUS:.6f} c",
            f"{cx - K * RADIUS:.6f} {cy + RADIUS:.6f} {cx - RADIUS:.6f} {cy + K * RADIUS:.6f} {cx - RADIUS:.6f} {cy:.6f} c",
            f"{cx - RADIUS:.6f} {cy - K * RADIUS:.6f} {cx - K * RADIUS:.6f} {cy - RADIUS:.6f} {cx:.6f} {cy - RADIUS:.6f} c",
            f"{cx + K * RADIUS:.6f} {cy - RADIUS:.6f} {cx + RADIUS:.6f} {cy - K * RADIUS:.6f} {cx + RADIUS:.6f} {cy:.6f} c S",
        ])
commands.append("Q")
content = ("\n".join(commands) + "\n").encode("ascii")

objects = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W:.6f} {PAGE_H:.6f}] /Resources << >> /Contents 4 0 R >>".encode("ascii"),
    b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
]

pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
offsets = [0]
for number, obj in enumerate(objects, start=1):
    offsets.append(len(pdf))
    pdf.extend(f"{number} 0 obj\n".encode("ascii"))
    pdf.extend(obj)
    pdf.extend(b"\nendobj\n")
xref = len(pdf)
pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
for offset in offsets[1:]:
    pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))

Path("同心圆25mm_正方形40mm_田字中线_A4多份.pdf").write_bytes(pdf)
