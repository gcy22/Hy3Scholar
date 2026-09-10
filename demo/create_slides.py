from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1080
BG = "#07111F"
SURFACE = "#0E1D30"
SURFACE_2 = "#132740"
TEXT = "#F4F8FF"
MUTED = "#9CB0C8"
CYAN = "#32E0C4"
BLUE = "#5797FF"
ORANGE = "#FFB45C"
RED = "#FF6B7A"
GREEN = "#53D88B"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = Path(__file__).resolve().parent
BUILD_DIR = DEMO_DIR / "build"
SLIDES_DIR = BUILD_DIR / "slides"
COVER_OUTPUT = DEMO_DIR / "Hy3Scholar_Demo_Cover.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F18 = font(18)
F20 = font(20)
F22 = font(22)
F24 = font(24)
F26 = font(26)
F28 = font(28)
F30 = font(30)
F32 = font(32)
F36 = font(36)
F42 = font(42, True)
F50 = font(50, True)
F64 = font(64, True)
F88 = font(88, True)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(7 + 5 * ratio)
        g = int(17 + 10 * ratio)
        b = int(31 + 18 * ratio)
        draw.line((0, y, WIDTH, y), fill=(r, g, b))
    draw.ellipse((1420, -360, 2180, 400), fill="#0A2740")
    draw.ellipse((-300, 760, 480, 1500), fill="#0A2034")
    return image, draw


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, radius: int = 24,
            outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont,
         max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        candidate = current + char
        if current and draw.textlength(candidate, font=selected_font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
              selected_font: ImageFont.ImageFont, fill: str, max_width: int,
              spacing: int = 12) -> int:
    x, y = xy
    line_height = selected_font.size + spacing
    for line in wrap(draw, text, selected_font, max_width):
        draw.text((x, y), line, font=selected_font, fill=fill)
        y += line_height
    return y


def header(draw: ImageDraw.ImageDraw, index: int, title: str) -> None:
    rounded(draw, (72, 54, 128, 110), CYAN, 18)
    draw.text((88, 61), "H", font=F36, fill=BG)
    draw.text((148, 59), "Hy3Scholar", font=F30, fill=TEXT)
    draw.text((148, 95), "TRUSTWORTHY LITERATURE INTELLIGENCE", font=F18, fill=MUTED)
    draw.text((1410, 62), f"0{index}  /  09", font=F22, fill=MUTED)
    draw.text((72, 146), title, font=F50, fill=TEXT)
    draw.line((72, 220, 1848, 220), fill="#24415F", width=2)


def footer(draw: ImageDraw.ImageDraw, index: int, narration: str) -> None:
    rounded(draw, (72, 900, 1848, 1024), "#07101CEB", 24, outline="#294865", width=2)
    draw.text((104, 924), "旁白", font=F22, fill=CYAN)
    lines = wrap(draw, narration, F30, 1585)
    y = 920
    for line in lines[:2]:
        draw.text((190, y), line, font=F30, fill=TEXT)
        y += 43
    for dot in range(1, 10):
        x = 790 + dot * 35
        color = CYAN if dot == index else "#31516D"
        draw.ellipse((x, 1043, x + 10, 1053), fill=color)


def metric(draw: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, color: str) -> None:
    rounded(draw, (x, y, x + 300, y + 150), SURFACE_2, 24, outline="#264866")
    draw.text((x + 28, y + 23), value, font=F64, fill=color)
    draw.text((x + 30, y + 103), label, font=F26, fill=MUTED)


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: str) -> int:
    width = int(draw.textlength(text, font=F22)) + 38
    rounded(draw, (x, y, x + width, y + 42), color, 18)
    draw.text((x + 19, y + 7), text, font=F22, fill=BG)
    return x + width + 12


def node(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, title: str,
         subtitle: str, color: str, number: str) -> None:
    rounded(draw, (x, y, x + w, y + 150), SURFACE_2, 24, outline=color, width=3)
    rounded(draw, (x + 20, y + 20, x + 64, y + 64), color, 15)
    draw.text((x + 34, y + 24), number, font=F26, fill=BG)
    draw.text((x + 82, y + 23), title, font=F30, fill=TEXT)
    multiline(draw, (x + 22, y + 82), subtitle, F22, MUTED, w - 44, 8)


def save(image: Image.Image, index: int) -> None:
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(SLIDES_DIR / f"{index:02d}.png", quality=95)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slide_1(narration: str) -> None:
    image, draw = canvas()
    rounded(draw, (120, 125, 310, 315), CYAN, 54)
    draw.text((169, 150), "H3", font=F88, fill=BG)
    draw.text((360, 135), "Hy3Scholar", font=F88, fill=TEXT)
    draw.text((365, 245), "可信文献综述生成与 Dataset v0 评测系统", font=F42, fill=CYAN)
    multiline(draw, (365, 330), "从开放论文发现，到证据约束生成、人工审核与可信评测。", F32, MUTED, 1160)
    x = 365
    for text, color in [("Hy3", CYAN), ("Evidence-grounded", BLUE), ("Human-in-the-loop", ORANGE)]:
        x = badge(draw, x, 430, text, color)
    rounded(draw, (365, 530, 1620, 760), SURFACE, 30, outline="#284966", width=2)
    draw.text((415, 580), "FUNCTION DEMO", font=F22, fill=MUTED)
    draw.text((415, 630), "文献发现  →  Dataset 构建  →  人工审核  →  七维评测", font=F36, fill=TEXT)
    draw.text((415, 698), "中文 · 1080P · 真实运行数据", font=F28, fill=CYAN)
    footer(draw, 1, narration)
    save(image, 1)


def slide_2(narration: str) -> None:
    image, draw = canvas()
    header(draw, 2, "为什么需要可信学术评测？")
    draw.text((80, 270), "开放式综述", font=F64, fill=TEXT)
    draw.text((82, 352), "没有唯一标准答案", font=F50, fill=CYAN)
    multiline(draw, (82, 430), "文本相似度无法判断一个结论是否真实，\n也无法确认引用是否真正支持相邻论断。", F30, MUTED, 700, 16)
    cards = [
        (860, "事实错误", "数字、方法或结论偏离原文", RED, "01"),
        (1190, "引用错配", "引用存在，但并不支持论断", ORANGE, "02"),
        (1520, "内容遗漏", "关键方法、结果或局限未覆盖", BLUE, "03"),
    ]
    for x, title, body, color, number in cards:
        rounded(draw, (x, 285, x + 285, 695), SURFACE, 28, outline="#2A4865")
        draw.text((x + 26, 315), number, font=F28, fill=color)
        draw.text((x + 26, 385), title, font=F36, fill=TEXT)
        multiline(draw, (x + 26, 455), body, F26, MUTED, 225, 15)
        draw.line((x + 26, 625, x + 245, 625), fill=color, width=8)
    footer(draw, 2, narration)
    save(image, 2)


def slide_3(narration: str) -> None:
    image, draw = canvas()
    header(draw, 3, "技术原理：证据驱动的双向核验")
    positions = [92, 430, 768, 1106, 1444]
    labels = [
        ("多源发现", "OpenAlex / Crossref / arXiv", BLUE),
        ("证据库", "论文页码 + 稳定证据 ID", CYAN),
        ("Hy3 生成", "摘要 / 问答 / 难例 / 反例", ORANGE),
        ("双向核验", "Claim / Evidence / Review", GREEN),
        ("可信评测", "七维评分与错误定位", RED),
    ]
    for idx, (x, (title, subtitle, color)) in enumerate(zip(positions, labels), 1):
        node(draw, x, 300, 280, title, subtitle, color, str(idx))
        if idx < 5:
            draw.line((x + 282, 375, x + 325, 375), fill="#55718E", width=5)
            draw.polygon([(x + 325, 366), (x + 342, 375), (x + 325, 384)], fill="#55718E")
    rounded(draw, (355, 535, 1565, 790), SURFACE, 26, outline="#2B4B69")
    draw.text((420, 578), "Claim  →  Evidence", font=F42, fill=CYAN)
    draw.text((995, 578), "论断是否被原文支持？", font=F30, fill=TEXT)
    draw.line((420, 650, 1500, 650), fill="#294967", width=2)
    draw.text((420, 685), "Evidence  →  Review", font=F42, fill=ORANGE)
    draw.text((995, 685), "关键证据是否被综述覆盖？", font=F30, fill=TEXT)
    footer(draw, 3, narration)
    save(image, 3)


def slide_4(narration: str, report: dict) -> None:
    image, draw = canvas()
    header(draw, 4, "功能一：多源文献发现与去重")
    rounded(draw, (74, 262, 1846, 410), SURFACE, 24, outline="#2A4B69")
    draw.text((105, 285), "研究主题", font=F22, fill=MUTED)
    draw.text((105, 337), report.get("topic", "LLM agent memory evaluation"), font=F36, fill=TEXT)
    rounded(draw, (1580, 300, 1800, 375), CYAN, 22)
    draw.text((1633, 318), "开始检索", font=F30, fill=BG)
    sources = [("OpenAlex", "结构化 OA 元数据", BLUE), ("Crossref", "DOI 与出版信息", ORANGE), ("arXiv", "预印本与 OA PDF", CYAN)]
    for idx, (title, subtitle, color) in enumerate(sources):
        x = 80 + idx * 400
        rounded(draw, (x, 455, x + 350, 600), SURFACE_2, 22, outline=color, width=2)
        draw.text((x + 25, 478), title, font=F32, fill=TEXT)
        draw.text((x + 25, 535), subtitle, font=F22, fill=MUTED)
    metric(draw, 1280, 455, str(report.get("discovered_records", 22)), "去重候选", BLUE)
    metric(draw, 1580, 455, str(report.get("oa_candidates", 10)), "开放获取候选", CYAN)
    rounded(draw, (80, 655, 1800, 820), SURFACE, 22)
    stages = ["查询规划", "并行检索", "DOI / arXiv ID", "标题 + 第一作者", "相关性排序"]
    for idx, stage in enumerate(stages):
        x = 110 + idx * 330
        draw.ellipse((x, 695, x + 28, 723), fill=CYAN)
        draw.text((x + 45, 688), stage, font=F26, fill=TEXT)
        if idx < len(stages) - 1:
            draw.line((x + 225, 710, x + 315, 710), fill="#385A77", width=3)
    footer(draw, 4, narration)
    save(image, 4)


def slide_5(narration: str, manifests: list[dict]) -> None:
    image, draw = canvas()
    header(draw, 5, "功能二：仅下载开放获取 PDF 并验证")
    rounded(draw, (72, 270, 1848, 650), SURFACE, 26, outline="#2A4B69")
    columns = [(105, "论文"), (1130, "页数"), (1280, "文本字符"), (1500, "状态")]
    for x, title in columns:
        draw.text((x, 300), title, font=F24, fill=MUTED)
    draw.line((100, 350, 1810, 350), fill="#294865", width=2)
    for row, item in enumerate(manifests[:2]):
        y = 380 + row * 120
        title = item.get("title", "Open-access paper")
        if len(title) > 63:
            title = title[:60] + "…"
        draw.text((105, y), f"P00{row + 1}", font=F22, fill=CYAN)
        draw.text((180, y), title, font=F26, fill=TEXT)
        draw.text((1150, y), str(item.get("page_count", "—")), font=F30, fill=TEXT)
        draw.text((1305, y), f"{item.get('verified_text_chars', 0):,}", font=F30, fill=TEXT)
        rounded(draw, (1510, y - 4, 1775, y + 48), "#173D35", 18)
        draw.text((1542, y + 5), "✓ 验证通过", font=F24, fill=GREEN)
    checks = [("%PDF 文件头", CYAN), ("文件大小", BLUE), ("有效页数", ORANGE), ("可提取文本", GREEN), ("SHA-256", RED)]
    x = 95
    for text, color in checks:
        rounded(draw, (x, 700, x + 310, 790), SURFACE_2, 20, outline=color)
        draw.ellipse((x + 22, 728, x + 48, 754), fill=color)
        draw.text((x + 65, 721), text, font=F26, fill=TEXT)
        x += 340
    footer(draw, 5, narration)
    save(image, 5)


def slide_6(narration: str, report: dict, cases: list[dict]) -> None:
    image, draw = canvas()
    header(draw, 6, "功能三：Hy3 自动构建 Dataset v0")
    task_counts = report.get("validation", {}).get("task_counts", {})
    difficulty = report.get("validation", {}).get("difficulty_counts", {})
    metric(draw, 78, 275, str(task_counts.get("structured_summary", 2)), "结构化摘要", BLUE)
    metric(draw, 400, 275, str(task_counts.get("evidence_qa", 2)), "证据问答", CYAN)
    metric(draw, 722, 275, str(task_counts.get("claim_check", 3)), "论断核对", ORANGE)
    rounded(draw, (1068, 275, 1844, 425), SURFACE_2, 24, outline="#284A68")
    draw.text((1100, 298), "难度分布", font=F24, fill=MUTED)
    x = 1100
    for label, key, color in [("标准", "standard", GREEN), ("难例", "hard", ORANGE), ("反例", "counterexample", RED)]:
        draw.text((x, 350), f"{label}  {difficulty.get(key, 0)}", font=F30, fill=color)
        x += 220
    selected = [cases[3] if len(cases) > 3 else {}, cases[6] if len(cases) > 6 else {}]
    labels = [("HARD · 跨论文比较", ORANGE), ("COUNTEREXAMPLE · 矛盾反例", RED)]
    display_instructions = [
        "比较 P001 与 P002 中的‘情景记忆’概念，避免混淆相似术语。",
        "反例：P001 是否声称 FP16、8K 上下文可以容纳 12 个 Agent？",
    ]
    for idx, (case, (label, color)) in enumerate(zip(selected, labels)):
        x = 78 + idx * 885
        rounded(draw, (x, 490, x + 830, 825), SURFACE, 26, outline=color, width=2)
        draw.text((x + 30, 520), case.get("case_id", f"V0-000{idx + 4}"), font=F22, fill=MUTED)
        draw.text((x + 180, 520), label, font=F22, fill=color)
        instruction = display_instructions[idx]
        multiline(draw, (x + 30, 575), instruction, F30, TEXT, 760, 13)
        ev = "  ".join(case.get("gold_evidence_ids", [])[:3])
        rounded(draw, (x + 28, 748, x + 792, 798), SURFACE_2, 15)
        draw.text((x + 45, 758), f"Evidence  {ev}", font=F20, fill=CYAN)
    footer(draw, 6, narration)
    save(image, 6)


def slide_7(narration: str, cases: list[dict], workspace: dict) -> None:
    image, draw = canvas()
    header(draw, 7, "功能四：人工审核与原文证据定位")
    case = cases[-1] if cases else {}
    evidence_id = (case.get("gold_evidence_ids") or ["P001:p1:c1"])[0]
    evidence = next((chunk for chunk in workspace.get("chunks", []) if chunk.get("evidence_id") == evidence_id), {})
    rounded(draw, (75, 270, 1000, 840), SURFACE, 26, outline="#2B4A67")
    draw.text((110, 302), f"{case.get('case_id', 'V0-0007')}  ·  claim_check  ·  counterexample", font=F24, fill=RED)
    draw.text((110, 360), "Instruction", font=F22, fill=MUTED)
    multiline(draw, (110, 405), case.get("instruction", "核对该论断是否成立。"), F30, TEXT, 820, 13)
    draw.text((110, 575), "Claim label", font=F22, fill=MUTED)
    badge(draw, 110, 620, case.get("claim_label", "Contradicted"), RED)
    draw.text((110, 705), "审核结论", font=F22, fill=MUTED)
    x = badge(draw, 110, 748, "PENDING", ORANGE)
    draw.text((x + 5, 754), "→", font=F30, fill=MUTED)
    badge(draw, x + 65, 748, "APPROVED", GREEN)
    rounded(draw, (1045, 270, 1845, 840), SURFACE_2, 26, outline=CYAN, width=2)
    draw.text((1080, 302), "原始证据", font=F30, fill=TEXT)
    draw.text((1080, 355), evidence_id, font=F24, fill=CYAN)
    draw.text((1080, 405), f"论文页码  p.{evidence.get('page', '—')}", font=F22, fill=MUTED)
    excerpt = evidence.get("text", "原文证据在这里按论文、页码和文本块进行定位。")
    if len(excerpt) > 330:
        excerpt = excerpt[:327] + "…"
    multiline(draw, (1080, 465), excerpt, F24, TEXT, 700, 12)
    footer(draw, 7, narration)
    save(image, 7)


def slide_8(narration: str, evaluation: dict) -> None:
    image, draw = canvas()
    header(draw, 8, "功能五：七维可信评测与错误定位")
    score = float(evaluation.get("overall_score", 68.55))
    rounded(draw, (75, 270, 425, 820), SURFACE, 28, outline=CYAN, width=2)
    draw.text((112, 310), "综合得分", font=F26, fill=MUTED)
    draw.text((110, 380), f"{score:.2f}", font=F88, fill=CYAN)
    draw.text((112, 490), "/ 100", font=F30, fill=MUTED)
    claims = evaluation.get("claims", [])
    key_points = evaluation.get("key_evidence_points", [])
    draw.line((110, 560, 390, 560), fill="#294A67", width=2)
    draw.text((112, 600), f"{len(claims)}", font=F50, fill=TEXT)
    draw.text((225, 616), "个原子 Claim", font=F24, fill=MUTED)
    draw.text((112, 690), f"{len(key_points)}", font=F50, fill=TEXT)
    draw.text((225, 706), "个关键证据点", font=F24, fill=MUTED)
    scores = evaluation.get("dimension_scores", {})
    dimensions = [
        ("事实准确性", "factual_accuracy", BLUE),
        ("引用忠实度", "citation_faithfulness", ORANGE),
        ("引用完整性", "citation_completeness", CYAN),
        ("内容覆盖", "coverage", RED),
        ("比较深度", "comparative_depth", GREEN),
        ("逻辑连贯性", "logical_coherence", BLUE),
        ("学术诚信", "academic_integrity", ORANGE),
    ]
    for idx, (label, key, color) in enumerate(dimensions):
        y = 285 + idx * 76
        value = float(scores.get(key, 0))
        draw.text((490, y), label, font=F24, fill=TEXT)
        rounded(draw, (710, y + 4, 1590, y + 34), "#18304A", 12)
        rounded(draw, (710, y + 4, 710 + int(8.8 * value), y + 34), color, 12)
        draw.text((1620, y - 3), f"{value:.1f}", font=F26, fill=color)
    rounded(draw, (1440, 734, 1838, 825), "#351F2B", 20, outline=RED)
    draw.text((1470, 755), "定位具体错误与改进建议", font=F24, fill=TEXT)
    footer(draw, 8, narration)
    save(image, 8)


def slide_9(narration: str) -> None:
    image, draw = canvas()
    header(draw, 9, "Hy3Scholar：让开放式学术任务可验证")
    draw.text((100, 310), "生成", font=F64, fill=BLUE)
    draw.text((340, 310), "+", font=F64, fill=MUTED)
    draw.text((450, 310), "证据核验", font=F64, fill=CYAN)
    draw.text((830, 310), "+", font=F64, fill=MUTED)
    draw.text((940, 310), "人工审核", font=F64, fill=ORANGE)
    draw.text((1320, 310), "+", font=F64, fill=MUTED)
    draw.text((1430, 310), "可信评测", font=F64, fill=GREEN)
    rounded(draw, (100, 480, 1818, 740), SURFACE, 30, outline="#2C4D6B")
    values = [("可靠", "每个结论回到原文证据", CYAN), ("透明", "保留来源、页码与审计记录", BLUE), ("可复现", "结构化 JSONL 与批量评测", GREEN)]
    for idx, (title, subtitle, color) in enumerate(values):
        x = 160 + idx * 550
        draw.text((x, 530), title, font=F50, fill=color)
        draw.text((x, 610), subtitle, font=F26, fill=TEXT)
    draw.text((103, 790), "github.com/gcy22/Hy3Scholar", font=F28, fill=MUTED)
    footer(draw, 9, narration)
    save(image, 9)


def create_cover() -> None:
    """Create a README-friendly cover that links to the rendered MP4."""
    image = Image.open(SLIDES_DIR / "01.png").convert("RGB")
    draw = ImageDraw.Draw(image)
    rounded(draw, (1430, 505, 1795, 690), "#0A1829", 30, outline=CYAN, width=3)
    draw.ellipse((1482, 545, 1582, 645), fill=CYAN)
    draw.polygon(((1521, 568), (1521, 622), (1561, 595)), fill=BG)
    draw.text((1610, 548), "点击播放", font=F32, fill=TEXT)
    draw.text((1610, 605), "1080p · 1:54", font=F22, fill=MUTED)
    image.save(COVER_OUTPUT, optimize=True)


def main() -> None:
    narration = json.loads((DEMO_DIR / "narration.json").read_text(encoding="utf-8"))
    report = load_json(REPO_ROOT / "dataset_v0" / "build_report.json")
    manifests = [json.loads(line) for line in (REPO_ROOT / "dataset_v0" / "download_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [json.loads(line) for line in (REPO_ROOT / "dataset_v0" / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    workspace = load_json(REPO_ROOT / "dataset_v0" / "workspace.json")
    latest_files = sorted((REPO_ROOT / "data").glob("*/results/latest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not latest_files:
        raise FileNotFoundError("没有找到真实评测结果 data/*/results/latest.json")
    evaluation = load_json(latest_files[0])["evaluation"]
    builders = [slide_1, slide_2, slide_3]
    for index, builder in enumerate(builders, 1):
        builder(narration[index - 1]["narration"])
    slide_4(narration[3]["narration"], report)
    slide_5(narration[4]["narration"], manifests)
    slide_6(narration[5]["narration"], report, cases)
    slide_7(narration[6]["narration"], cases, workspace)
    slide_8(narration[7]["narration"], evaluation)
    slide_9(narration[8]["narration"])
    create_cover()
    print(f"Rendered {len(narration)} slides to {SLIDES_DIR}")
    print(f"Rendered README cover to {COVER_OUTPUT}")


if __name__ == "__main__":
    main()
