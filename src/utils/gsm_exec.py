"""Optional TinyGSM program scoring for trusted generated programs.

Subprocess time and memory limits protect the evaluator from ordinary program
errors. Restricted Python builtins are not a security boundary for hostile code.
"""

import json
import re
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation


_WORKER = """
import contextlib, io, json, math, resource, sys
request = json.load(sys.stdin)
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 ** 2,) * 2)
resource.setrlimit(resource.RLIMIT_CPU, (max(1, math.ceil(request['timeout'])),) * 2)
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
allowed = {
    name: getattr(__builtins__, name)
    for name in ('abs', 'min', 'max', 'sum', 'len', 'range', 'enumerate',
                 'int', 'float', 'str', 'bool', 'round', 'print', 'list',
                 'dict', 'tuple', 'set', 'zip', 'map', 'filter', 'sorted', 'reversed')
}
def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'math' and level == 0:
        return math
    raise ImportError(name)
allowed['__import__'] = limited_import
namespace = {'__builtins__': allowed, 'math': math}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(request['code'], namespace, namespace)
        result = namespace['simple_math_problem']()
    if not isinstance(result, (int, float, str)):
        result = None
    print(json.dumps(result, allow_nan=False))
except BaseException:
    print('null')
"""


def _extract_code(text):
    """Strip Markdown and trailing partial lines from a generated program."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    for stopper in ("<|endoftext|>", "<|eot_id|>", "</s>"):
        text = text.split(stopper, 1)[0]
    start = text.find("def ")
    if start >= 0:
        text = text[start:]
    lines = text.strip().splitlines()
    for trim in range(min(50, len(lines))):
        candidate = "\n".join(lines[:len(lines) - trim])
        try:
            compile(candidate, "<sample>", "exec")
            return candidate
        except (SyntaxError, ValueError):
            continue
    return text


def execute_for_answer(text, timeout_s=1.0):
    """Evaluate `simple_math_problem()` in a fresh, resource-limited process."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    try:
        with tempfile.TemporaryDirectory(prefix="elf-gsm-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _WORKER],
                input=json.dumps({"code": _extract_code(text), "timeout": timeout_s}),
                text=True, capture_output=True, timeout=timeout_s,
                cwd=directory, env={}, check=False,
            )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _number(value):
    if isinstance(value, str):
        matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value.split("####")[-1])
        value = matches[-1].replace(",", "") if matches else None
    if value is None:
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def compute_accuracy(hypotheses, references, timeout_s=1.0):
    """Return answer accuracy in percent; failed programs count as incorrect."""
    if len(hypotheses) != len(references):
        raise ValueError("Each generated answer needs one reference")
    correct = 0
    for hypothesis, reference in zip(hypotheses, references):
        hypothesis, reference = str(hypothesis), str(reference)
        prediction = (execute_for_answer(hypothesis, timeout_s)
                      if "def " in hypothesis else hypothesis)
        target = (execute_for_answer(reference, timeout_s)
                  if "def " in reference else reference)
        prediction, target = _number(prediction), _number(target)
        if prediction is not None and target is not None:
            correct += abs(prediction - target) <= Decimal("0.001")
    return {"accuracy": 100.0 * correct / len(hypotheses) if hypotheses else 0.0}
