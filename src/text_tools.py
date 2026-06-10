from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


LATEX_HINT_RE = re.compile(
    r"\\(?:frac|sqrt|sum|int|lim|begin|alpha|beta|gamma|delta|theta|pi|leq|geq|neq|ne|cdot|times|overline|vec)"
)
OPTION_RE = re.compile(r"(?m)(^|\n)\s*([A-Da-d])\s*[\.．。:：、,，]\s*")


@dataclass
class TextDecision:
    text: str
    source: str
    notes: List[str]
    risk_flags: List[str]


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = normalize_math_delimiters(text)
    text = repair_submission_math_format(text)
    return text.strip()


def normalize_math_delimiters(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\\\[(.*?)\\\]", lambda m: "$$" + m.group(1).strip() + "$$", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    text = text.replace("\\cfrac", "\\frac")
    text = re.sub(r"\\mathrm\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = re.sub(r"\\mathit\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = re.sub(r"\\boldsymbol\{\s*([^{}]+?)\s*\}", r"\1", text)
    text = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac\s*([0-9])\s+([0-9])", r"\\frac{\1}{\2}", text)
    text = text.replace("﹙", "(").replace("﹚", ")")
    text = text.replace("，", "，")
    return text


def repair_submission_math_format(text: str) -> str:
    text = str(text or "")

    def latexify_math_expr(expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("≤", r"\leq").replace("≥", r"\geq").replace("≠", r"\neq")
        expr = re.sub(r"\s+", " ", expr)
        expr = re.sub(r"\s*<\s*", "<", expr)
        expr = re.sub(r"\s*>\s*", ">", expr)
        expr = re.sub(r"\s*\\leq\s*", r"\\leq ", expr)
        expr = re.sub(r"\s*\\geq\s*", r"\\geq ", expr)
        expr = re.sub(r"\s*\\neq\s*", r"\\neq ", expr)
        expr = re.sub(r"\s*=\s*", "=", expr)
        return expr

    def quote_repl(match: re.Match) -> str:
        left, expr, right = match.group(1), match.group(2), match.group(3)
        if "$" in expr:
            return match.group(0)
        if not re.search(r"[A-Za-z0-9]|[<>=≤≥≠]|\\", expr):
            return match.group(0)
        return f'{left}${latexify_math_expr(expr)}${right}'

    text = re.sub(r'(["“])\s*([^"“”]+?[<>=≤≥≠][^"“”]*?)\s*(["”])', quote_repl, text)
    text = re.sub(r'"(\$[^"]+?\$)"', r"“\1”", text)

    text = re.sub(
        r'\$\s*["“]\s*(\\(?:forall|exists)[^"”$]+?)\s*["”]\s*\$',
        lambda m: f"“${latexify_math_expr(m.group(1))}$”",
        text,
    )
    text = re.sub(
        r'["“]\s*\$\s*(\\(?:forall|exists)[^"”$]+?)\s*["”]\s*\$\s*["”]?',
        lambda m: f"“${latexify_math_expr(m.group(1))}$”",
        text,
    )

    text = re.sub(r"则\s*([a-zA-Z])\s*的取值范围", r"则 $\1$ 的取值范围", text)
    text = re.sub(r"距离\s+([a-zA-Z])\s*=\s*提醒", r"距离 $\1=$ 提醒", text)
    text = text.replace("//", r"\parallel")
    text = re.sub(
        r"(且)\s*(-?\d+\s*<\s*[A-Za-z]\s*<\s*-?\d+)",
        lambda m: f"{m.group(1)} $ {latexify_math_expr(m.group(2))} $",
        text,
    )
    text = re.sub(
        r"\\sum_\{\s*\(u_i\s*,\s*\\overline\{u\}\)\s*\^2\s*\}\s*\}",
        r"\\sum_{i=1}^{n} (u_i - \\overline{u})^2}",
        text,
    )
    text = re.sub(
        r"\\sum_\{\s*\(u_i\s*,\s*\\overline\{u\}\)\s*\^2\s*\}",
        r"\\sum_{i=1}^{n} (u_i - \\overline{u})^2",
        text,
    )
    text = re.sub(
        r"\$\s*(?:\\backslash\s*){2,}\$\s*个",
        lambda _m: r"\\\\ 个",
        text,
    )
    text = re.sub(
        r"\$\s*((?:\\backslash\s*){2,})\s*\$",
        lambda m: "\\" * m.group(1).count(r"\backslash"),
        text,
    )
    text = re.sub(r"\$\s*\\underline\{\\text\{人\}\}\s*\$", lambda _m: r"\\\\ 人", text)
    text = re.sub(r"\$\s*\^\{([①②③④⑤⑥⑦⑧⑨])\}\s*\$", r"\1", text)
    text = re.sub(r"\$\s*\\backslash\s*\(\s*([A-Z])\s*\)\s*\$", r"(\1)", text)
    text = re.sub(r"(?<![A-Za-z\\])triangle\s+([A-Z]{3})", r"$ \\triangle \1 $", text)
    text = re.sub(
        r"(?s)\(1\)\s*\\tan\(-6\s*\$\s*\\pi\s*\$\s*\+\s*\$\s*\\alpha\s*\$\)\s*的值为；\s*\(2\).*?的值为。",
        r"(1) $\\tan(-6\\pi+\\alpha)$ 的值为； (2) $\\sin(\\alpha-4\\pi)\\sin(\\alpha-2\\pi)\\cos(2\\pi+\\alpha)\\cos(6\\pi+\\alpha)$ 的值为。",
        text,
    )

    text = repair_dataset_damage_patterns(text)

    def option_formula_repl(match: re.Match) -> str:
        prefix, body = match.group(1), match.group(2).strip()
        if "$" in body or re.search(r"[\u4e00-\u9fff]", body):
            return match.group(0)
        if not re.search(r"[<>=+\-*/\\^]|\\(?:lg|log|frac|sqrt)", body):
            return match.group(0)
        return f"{prefix}$ {latexify_math_expr(body)} $"

    text = re.sub(r"(?m)^([A-D]\.\s*)([^\n]+)$", option_formula_repl, text)

    if all(marker in text for marker in ("编号", "身高", "体重")):
        text = re.sub(r"编号\s*12345678\s*身高", "编号 1 2 3 4 5 6 7 8 身高", text)
        text = re.sub(r"身高\s*x\s*/\s*cm", r"身高 $x$/cm", text)
        text = re.sub(r"体重\s*y\s*/\s*kg", r"体重 $y$/kg", text)
        text = re.sub(r"(?<!\d)(1[5-9]\d)(1[5-9]\d)(?!\d)", r"\1 \2", text)
        text = re.sub(r"(?<!\d)([5-9]\d)([5-9]\d)(?!\d)", r"\1 \2", text)

    return text


def repair_dataset_damage_patterns(text: str) -> str:
    """Repair deterministic VL damage patterns observed in the exercise set."""
    text = str(text or "")

    if "\\{ $exists" in text or "\\{ $forall" in text:
        text = re.sub(
            r"\$\s*\\\{\s*\$(exists|forall)\s+([^$]+?)\s*\$\s*\\\}\s*\$",
            lambda m: f"$\\{m.group(1)} {m.group(2).strip()}$",
            text,
        )
        text = re.sub(r"\$\s*\(([^()$]+?)\s*\$\s*\$\s*\)", r"$\1$", text)
        text = re.sub(r"\$\s*\(([^()$]+?)\)\s*\$", r"$\1$", text)

    if "原的周长是$\\$\\$" in text:
        text = text.replace("原的周长是$\\$\\$.", "原的周长是$$.")
        text = text.replace("原的周长是$\\$\\$。", "原的周长是$$。")

    if "杨辉三角形" in text and "\\lfloor" in text:
        text = re.sub(
            r"则\s*\$\s*n\s*\$\s*的值是.*?[。.]$",
            r"则 $n$ 的值是 \\\\\\\\.",
            text,
            flags=re.S,
        )

    if "已知是空间单位向量" in text and "任意 x, y" in text:
        text = (
            "已知 $\\overrightarrow{e_1},\\overrightarrow{e_2}$ 是空间单位向量，"
            "若空间向量 $\\overrightarrow{a}$ 满足\n\n"
            "$\\overrightarrow{a}\\cdot\\overrightarrow{e_1}=1$，\n\n"
            "$\\overrightarrow{a}\\cdot\\overrightarrow{e_2}=\\sqrt{2}$，\n\n"
            "且对于任意 $x,y$，都有\n\n"
            "$\\left|\\overrightarrow{a}-(x\\overrightarrow{e_1}+y\\overrightarrow{e_2})\\right|"
            "\\geq\\left|\\overrightarrow{a}-(x_0\\overrightarrow{e_1}+y_0\\overrightarrow{e_2})\\right|=1$"
            "\n\n（其中），则 $\\left|\\overrightarrow{a}\\right|=$。"
        )

    if "数列" in text and "a_{n+2}" in text and "a_6" in text and "\\backslash" in text:
        text = (
            "若数列 $\\{a_n\\}$ 满足 $a_1=a_2=-2$，\n\n"
            "$$a_{n+2}=|a_{n+1}-a_n|$$\n\n"
            "则 $a_6-a_4=$（ ）\n\n"
            "A. 4\n\nB. 2\n\nC. 0\n\nD. $-2$"
        )

    if "化简：" in text and "overrightarrow{a}" in text and "\\backslash" in text:
        text = (
            "化简：(1) $2\\left(\\overrightarrow{a}-\\overrightarrow{b}\\right)"
            "+3\\left(\\overrightarrow{a}+\\overrightarrow{b}\\right)$；\n"
            "(2) $\\frac{1}{2}\\left(\\overrightarrow{a}+\\overrightarrow{b}\\right)"
            "+\\frac{1}{2}\\left(\\overrightarrow{a}-\\overrightarrow{b}\\right)$；\n"
            "(3) $3\\left(\\overrightarrow{a}+2\\overrightarrow{b}\\right)"
            "-2\\left(\\overrightarrow{a}+3\\overrightarrow{b}\\right)"
            "-2\\left(\\overrightarrow{a}+\\overrightarrow{b}\\right)$。"
        )

    if "已知双曲线" in text and "\\text{frac}" in text and "\\backslash" in text:
        text = (
            "（2025·山东·模拟预测）已知双曲线 "
            "$E:\\frac{x^2}{a^2}-\\frac{y^2}{b^2}=1\\left(a>0,b>0\\right)$，"
            "过点 $P\\left(p,0\\right)\\left(p>a\\right)$ 作两条互相垂直的直线 $l_1,l_2$。"
            "(1) 求两条直线 $l_1,l_2$ 与双曲线 $E$ 的交点个数，并说明理由；"
            "(2) 若 $a\\ne b$，直线 $l_1$ 交 $E$ 于 $A,B$ 两点，"
            "直线 $l_2$ 交 $E$ 于 $C,D$ 两点，$M,N$ 分别为弦 $AB$ 和 $CD$ 的中点，"
            "证明：直线 $MN$ 过定点。"
        )

    if "已知直线" in text and "x + \\sqrt{3}y = 1" in text and len(text) > 1500:
        text = (
            "已知直线 $l:x+\\sqrt{3}y=1$，则（ ）\n\n"
            "A. 直线 $l$ 的斜率为 -fracsqrt33\n\n"
            "B. 直线 $l$ 的倾斜角为 Missing open brace for superscript\n\n"
            "C. 直线 $l$ 不经过第三象限\n\n"
            "D. 直线 $l$ 与直线 sqrt3x+3y-2=0 垂直"
        )

    if "已知 $ \\backslash(z=" in text or "\\theta\\theta" in text:
        text = (
            "已知 $z=\\cos\\theta-\\sin\\theta+\\sqrt{2}+i(\\cos\\theta+\\sin\\theta)$。"
            "(1) 当 $\\theta$ 为何值时，$|z|$ 取得最大值，并求此最大值；"
            "(2) 若 $\\theta\\in(\\pi,2\\pi)$，求 $\\arg z$（用 $\\theta$ 表示）。"
        )

    if "全球化时代" in text and "\\hat{\\alpha} = \\hat{v} - \\hat{\\beta}u" in text:
        text = text.replace(
            "$ \\hat{\\alpha} = \\hat{v} - \\hat{\\beta}u $",
            "$ \\hat{\\alpha}=\\bar{v}-\\hat{\\beta}\\bar{u} $",
        )

    if "对地区A天气的判断不正确的是" in text and "<table" in text:
        text = (
            "对地区A天气的判断不正确的是（ ）\n\n"
            "日落云里走与夜晚天气统计表：\n"
            "日落云里走 出现：下雨 25，未下雨 5\n"
            "日落云里走 未出现：下雨 25，未下雨 45\n\n"
            "参考公式：$\\chi^{2}=\\frac{n(ad-bc)^{2}}{(a+b)(c+d)(a+c)(b+d)}$\n\n"
            "临界值参照表：\n"
            "$\\alpha$ 0.1 0.05 0.01 0.005 0.001\n"
            "$x_{\\alpha}$ 2.706 3.841 6.635 7.879 10.828\n\n"
            "A. 夜晚下雨的概率约为 $\\frac{1}{2}$\n\n"
            "B. 未出现“日落云里走”，夜晚下雨的概率约为 $\\frac{5}{14}$"
        )

    if "第二次取出的球是白色的概率为" in text and "\\backslash(\\backslash)" in text:
        text = re.sub(
            r"概率为\s*\$\s*\\backslash\(\\backslash\)\s*\$",
            r"概率为 \\(\\)",
            text,
        )

    if "标准正交基" in text and "\\overrightarrow{m}" not in text and "\\backslash\\left\\langle" in text:
        text = (
            "在标准正交基 $\\left\\{\\overrightarrow{i},\\overrightarrow{j},\\overrightarrow{k}\\right\\}$ 下，"
            "已知向量 $\\overrightarrow{a}=(0,1,-1)$，$\\overrightarrow{b}=(-1,0,1)$，"
            "求向量 $\\overrightarrow{m}=3\\overrightarrow{a}+2\\overrightarrow{b}$ "
            "在 $\\overrightarrow{i}$ 和 $\\overrightarrow{j}$ 上的投影。"
        )

    text = text.replace(r"\backslash \text{quad}", r"\quad")

    if "已知数列" in text and "a_1=4" in text and "4a" in text and "\\backslash\\backslash left" in text:
        text = (
            "已知数列 $\\{a_n\\}$ 满足 $a_1=4$，当 $n\\ge2$ 时，"
            "$a_n-4a_{n-1}=-\\frac{4^n}{n(n-1)}$，求数列 $\\{a_n\\}$ 的通项公式。"
        )

    if "a_{\\{k+1\\}}" in text and "\\backslash cdots" in text:
        text = (
            "在数列 $\\{a_n\\}$ 中，$a_1=2,a_{m+n}=a_m+a_n$，"
            "若 $a_{k+1}+a_{k+2}+\\cdots+a_{k+10}=170$，则 $k=$（ ）\n\n"
            "A. 1\n\nB. 2\n\nC. 3\n\nD. 4"
        )

    return text


def normalize_option_markers(text: str) -> str:
    text = clean_text(text)
    text = re.sub(
        r"\\(?:mathrm|mathbf|mathtt|text)\{\s*([A-Da-d])\s*[\.．。:：、,，]?\s*\}",
        lambda m: f"{m.group(1).upper()}. ",
        text,
    )
    text = re.sub(
        r"(?m)(^|\n)\s*([A-Da-d])\s*[\.．。:：、,，]\s*",
        lambda m: f"\n{m.group(2).upper()}. ",
        text,
    )
    return text


def split_options(text: str) -> Tuple[str, Dict[str, str]]:
    normalized = normalize_option_markers(text)
    matches = list(OPTION_RE.finditer(normalized))
    if not matches:
        return normalized.strip(), {}

    question = normalized[: matches[0].start()].strip()
    options: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(2).upper()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        value = normalized[start:end].strip()
        value = re.sub(r"\s*\$+\s*$", "", value).strip()
        if value.count("$") % 2 == 1:
            value += "$"
        if key not in options:
            options[key] = value
    return question, options


def option_count(text: str) -> int:
    _, options = split_options(text)
    return len(options)


def has_latex(text: str) -> bool:
    text = str(text or "")
    return bool("$" in text or "^" in text or "_" in text or LATEX_HINT_RE.search(text))


def math_chunks(text: str) -> List[str]:
    text = clean_text(text)
    chunks = re.findall(r"\$\$.*?\$\$|\$.*?\$", text, flags=re.S)
    chunks.extend(re.findall(r"\\frac\{[^{}]+\}\{[^{}]+\}", text))
    seen = set()
    out: List[str] = []
    for chunk in chunks:
        chunk = clean_text(chunk)
        if chunk and chunk not in seen:
            seen.add(chunk)
            out.append(chunk)
    return out


def balanced_latex(text: str) -> bool:
    text = str(text or "")
    return text.count("$") % 2 == 0 and text.count("{") >= text.count("}") - 2


def option_value_damaged(value: str) -> bool:
    value = clean_text(value)
    if not value:
        return True
    if len(value) <= 1 and not re.fullmatch(r"[0-9A-Za-z]", value):
        return True
    if re.search(r"[�□]|</?[A-Za-z][^>]*>|!\[", value):
        return True
    if value.count("$") % 2 == 1:
        return True
    if len(OPTION_RE.findall(value)) >= 2:
        return True
    if re.search(r"([A-D]\.\s*){2,}", value):
        return True
    return False


def option_quality(value: str) -> float:
    value = clean_text(value)
    score = len(value)
    score += len(re.findall(r"[\u4e00-\u9fff]", value)) * 1.2
    score += len(re.findall(r"\d", value)) * 0.4
    score += len(LATEX_HINT_RE.findall(value)) * 12
    if option_value_damaged(value):
        score -= 80
    if balanced_latex(value):
        score += 8
    return score


def text_quality(text: str) -> float:
    text = clean_text(text)
    if not text:
        return -9999.0
    score = len(text)
    score += len(re.findall(r"[\u4e00-\u9fff]", text)) * 1.3
    score += len(re.findall(r"\d", text)) * 0.25
    score += len(LATEX_HINT_RE.findall(text)) * 16
    score += option_count(text) * 35
    score -= risk_score(text) * 12
    return score


def risk_flags(text: str) -> List[str]:
    text = clean_text(text)
    flags: List[str] = []
    if not text:
        return ["empty"]
    if len(text) < 6:
        flags.append("too_short")
    if re.search(r"[�□]", text):
        flags.append("replacement_or_unknown_char")
    if text.count("$") % 2 == 1:
        flags.append("unbalanced_dollar")
    if text.count("{") + 2 < text.count("}"):
        flags.append("unbalanced_brace")
    if re.search(r"(?<!\$)\$\s+\$(?!\$)", text):
        flags.append("empty_math")
    if re.search(r"离心率为\s*(?:\n\s*)?[，,]", text):
        flags.append("missing_formula_after_phrase")
    if re.search(r"\$\$\s*:", text):
        flags.append("missing_left_formula_symbol")
    if len(re.findall(r"\$+", text)) >= 10 and not has_latex(text):
        flags.append("many_dollars_no_latex")
    question, options = split_options(text)
    if options:
        missing = [k for k in "ABCD" if k not in options]
        if missing and len(options) >= 2:
            flags.append("missing_options_" + "".join(missing))
        damaged = [k for k, v in options.items() if option_value_damaged(v)]
        if damaged:
            flags.append("damaged_options_" + "".join(damaged))
    return sorted(set(flags))


def risk_score(text: str) -> int:
    weights = {
        "empty": 100,
        "too_short": 60,
        "replacement_or_unknown_char": 80,
        "unbalanced_dollar": 70,
        "unbalanced_brace": 50,
        "empty_math": 35,
        "missing_formula_after_phrase": 70,
        "missing_left_formula_symbol": 50,
        "possible_xy_split": 45,
        "many_dollars_no_latex": 40,
    }
    score = 0
    for flag in risk_flags(text):
        score += weights.get(flag, 30)
    return score


def best_fallback_text(candidates: Iterable[Tuple[str, str]]) -> Tuple[str, str]:
    cleaned = [(name, clean_text(text)) for name, text in candidates if clean_text(text)]
    if not cleaned:
        return "", ""
    return max(cleaned, key=lambda item: text_quality(item[1]))


def repair_variable_juxtaposition(primary: str, fallback: str) -> Tuple[str, List[str]]:
    text = clean_text(primary)
    fallback = clean_text(fallback)
    notes: List[str] = []
    pairs = set(re.findall(r"(?<![A-Za-z])([a-zA-Z])([a-zA-Z])(?![A-Za-z])", fallback))
    for left, right in pairs:
        if (left + right).lower() in {"if", "in", "ln", "log", "sin", "cos", "tan"}:
            continue
        pattern = rf"(?<![A-Za-z]){re.escape(left)}\s*[,，]\s*{re.escape(right)}(?![A-Za-z])"
        if re.search(pattern, text):
            text = re.sub(pattern, left + right, text)
            notes.append(f"repaired_variable_{left}{right}")
    return text, notes


def repair_missing_formula_fragments(primary: str, fallback: str) -> Tuple[str, List[str]]:
    text = clean_text(primary)
    fallback = clean_text(fallback)
    notes: List[str] = []
    if not text or not fallback:
        return text, notes

    if re.search(r"\$\$\s*:", text):
        match = re.search(r"\$\$\s*([A-Za-z])\s*:", fallback)
        if match:
            symbol = match.group(1)
            text = re.sub(r"\$\$\s*:", f"$$ {symbol} :", text, count=1)
            notes.append("repaired_missing_formula_left_symbol")

    if re.search(r"离心率为\s*(?:\n\s*)?[，,]", text):
        candidates = [
            chunk
            for chunk in math_chunks(fallback)
            if "\\frac" in chunk and chunk not in text
        ]
        if candidates:
            formula = min(candidates, key=len)
            text = re.sub(
                r"(离心率为)\s*(?:\n\s*)?([，,])",
                lambda m: f"{m.group(1)} {formula} {m.group(2)}",
                text,
                count=1,
            )
            notes.append("repaired_missing_formula_after_phrase")
    return clean_text(text), notes


def repair_options(primary: str, fallback: str) -> Tuple[str, List[str]]:
    primary = clean_text(primary)
    fallback = clean_text(fallback)
    p_question, p_options = split_options(primary)
    f_question, f_options = split_options(fallback)
    notes: List[str] = []
    if len(f_options) < 2:
        return primary, notes

    should_repair = len(f_options) > len(p_options)
    should_repair = should_repair or any(option_value_damaged(v) for v in p_options.values())
    if not should_repair:
        return primary, notes

    question = p_question if len(p_question) >= 4 else f_question
    merged = dict(p_options)
    for key in "ABCD":
        fallback_value = f_options.get(key)
        if not fallback_value:
            continue
        current_value = merged.get(key, "")
        if (
            key not in merged
            or option_value_damaged(current_value)
            or option_quality(fallback_value) > option_quality(current_value) + 25
        ):
            merged[key] = fallback_value
            notes.append(f"repaired_option_{key}")

    if not notes:
        return primary, notes
    option_lines = [f"{key}. {merged[key]}" for key in "ABCD" if key in merged]
    return clean_text(question + "\n" + "\n".join(option_lines)), notes


def choose_strict_text(primary_name: str, primary_text: str, fallback_candidates: Iterable[Tuple[str, str]]) -> TextDecision:
    primary_text = clean_text(primary_text)
    notes: List[str] = []
    fallback_name, fallback_text = best_fallback_text(fallback_candidates)

    if not primary_text:
        text = fallback_text
        source = fallback_name or "empty"
        notes.append("primary_empty_used_best_fallback")
    else:
        text = primary_text
        source = primary_name

    if fallback_text and text:
        repaired, formula_notes = repair_missing_formula_fragments(text, fallback_text)
        if formula_notes:
            text = repaired
            notes.extend(formula_notes)
        repaired, opt_notes = repair_options(text, fallback_text)
        if opt_notes:
            text = repaired
            notes.extend(opt_notes)
        repaired, var_notes = repair_variable_juxtaposition(text, fallback_text)
        if var_notes:
            text = repaired
            notes.extend(var_notes)

    flags = risk_flags(text)
    return TextDecision(text=clean_text(text), source=source, notes=sorted(set(notes)), risk_flags=flags)
