"""A small, dependency-free smart calculator.

Run ``python calculator_with_search.py`` for the menu, or add ``--web`` for the
browser app at http://127.0.0.1:8000.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


HOST = "127.0.0.1"
PORT = 8000
MAX_EXPRESSION_LENGTH = 300
MAX_AST_NODES = 100
FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "cos": math.cos,
    "exp": math.exp,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "sqrt": math.sqrt,
    "tan": math.tan,
}
CONSTANTS = {"e": math.e, "pi": math.pi}
BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _parse_expression(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Enter a mathematical expression.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long.")

    try:
        tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ValueError("That expression could not be understood.") from error

    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex.")
    return tree


def _validate_graph_expression(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_graph_expression(node.body)
    elif isinstance(node, ast.Constant):
        _checked_number(node.value)
    elif isinstance(node, ast.Name):
        if node.id != "x" and node.id not in CONSTANTS:
            raise ValueError(f"Unknown variable or constant: {node.id}")
    elif isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        _validate_graph_expression(node.operand)
    elif isinstance(node, ast.BinOp) and (
        type(node.op) in BINARY_OPERATORS or isinstance(node.op, ast.Pow)
    ):
        _validate_graph_expression(node.left)
        _validate_graph_expression(node.right)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in FUNCTIONS or node.keywords or len(node.args) != 1:
            raise ValueError("Use supported one-argument math functions, such as sin(x) or sqrt(x).")
        for argument in node.args:
            _validate_graph_expression(argument)
    else:
        raise ValueError("Expression contains an unsupported operation.")


def _checked_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expression must produce a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("The result is outside the supported numeric range.")
    return result


def _evaluate(node: ast.AST, x_value: float | None = None) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, x_value)
    if isinstance(node, ast.Constant):
        return _checked_number(node.value)
    if isinstance(node, ast.Name):
        if node.id == "x" and x_value is not None:
            return x_value
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise ValueError(f"Unknown variable or constant: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return _checked_number(UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, x_value)))
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, x_value)
        right = _evaluate(node.right, x_value)
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100:
                raise ValueError("Exponent magnitude must be 100 or less.")
            return _checked_number(left**right)
        operation = BINARY_OPERATORS.get(type(node.op))
        if operation is not None:
            return _checked_number(operation(left, right))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = FUNCTIONS.get(node.func.id)
        if function is not None and not node.keywords and len(node.args) == 1:
            return _checked_number(function(_evaluate(node.args[0], x_value)))
    raise ValueError("Use numbers, x, +, -, *, /, //, %, ^, parentheses, or supported functions.")


def calculate(expression: str) -> float:
    return _evaluate(_parse_expression(expression))


def _polynomial(node: ast.AST) -> list[float]:
    """Return coefficients [constant, x, x^2] for a supported polynomial."""
    if isinstance(node, ast.Expression):
        return _polynomial(node.body)
    if isinstance(node, ast.Constant):
        return [_checked_number(node.value), 0.0, 0.0]
    if isinstance(node, ast.Name):
        if node.id == "x":
            return [0.0, 1.0, 0.0]
        if node.id in CONSTANTS:
            return [CONSTANTS[node.id], 0.0, 0.0]
        raise ValueError("Equations may use only x, pi, and e.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        values = _polynomial(node.operand)
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        return [sign * value for value in values]
    if isinstance(node, ast.BinOp):
        left = _polynomial(node.left)
        right = _polynomial(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = -1.0 if isinstance(node.op, ast.Sub) else 1.0
            return [left[i] + sign * right[i] for i in range(3)]
        if isinstance(node.op, ast.Mult):
            result = [0.0, 0.0, 0.0]
            for left_power, left_value in enumerate(left):
                for right_power, right_value in enumerate(right):
                    if left_power + right_power > 2 and left_value * right_value != 0:
                        raise ValueError("Only linear and quadratic equations are supported.")
                    if left_power + right_power <= 2:
                        result[left_power + right_power] += left_value * right_value
            return [_checked_number(value) for value in result]
        if isinstance(node.op, ast.Div) and right[1] == 0 and right[2] == 0:
            if right[0] == 0:
                raise ValueError("Division by zero.")
            return [value / right[0] for value in left]
        if isinstance(node.op, ast.Pow) and right[1] == 0 and right[2] == 0:
            power = right[0]
            if power in (0, 1, 2):
                if power == 0:
                    return [1.0, 0.0, 0.0]
                if power == 1:
                    return left
                if left[1] * left[2] != 0 or left[2] * left[2] != 0:
                    raise ValueError("Only linear and quadratic polynomial equations are supported.")
                return [
                    _checked_number(left[0] * left[0]),
                    _checked_number(2 * left[0] * left[1]),
                    _checked_number(left[1] * left[1] + 2 * left[0] * left[2]),
                ]
    raise ValueError("Only linear and quadratic polynomial equations are supported.")


def solve_equation(equation: str) -> dict[str, Any]:
    if not isinstance(equation, str) or equation.count("=") != 1:
        raise ValueError("Enter an equation with one equals sign, such as 2*x + 3 = 7.")
    left_text, right_text = equation.split("=", maxsplit=1)
    left = _polynomial(_parse_expression(left_text))
    right = _polynomial(_parse_expression(right_text))
    constant, linear, quadratic = (left[i] - right[i] for i in range(3))

    if math.isclose(quadratic, 0.0, abs_tol=1e-12):
        if math.isclose(linear, 0.0, abs_tol=1e-12):
            if math.isclose(constant, 0.0, abs_tol=1e-12):
                return {"kind": "identity", "solutions": []}
            return {"kind": "none", "solutions": []}
        return {"kind": "linear", "solutions": [-constant / linear]}

    discriminant = linear * linear - 4 * quadratic * constant
    if discriminant < -1e-12:
        return {"kind": "none", "solutions": []}
    if math.isclose(discriminant, 0.0, abs_tol=1e-12):
        roots = [-linear / (2 * quadratic)]
    else:
        root = math.sqrt(discriminant)
        roots = sorted(((-linear - root) / (2 * quadratic), (-linear + root) / (2 * quadratic)))
    return {"kind": "quadratic", "solutions": roots}


def graph_expression(
    expression: str,
    x_min: float = -10.0,
    x_max: float = 10.0,
    samples: int = 401,
) -> dict[str, Any]:
    tree = _parse_expression(expression)
    _validate_graph_expression(tree)
    if not isinstance(samples, int) or isinstance(samples, bool) or not 50 <= samples <= 1200:
        raise ValueError("Sample count must be between 50 and 1200.")
    x_min = _checked_number(x_min)
    x_max = _checked_number(x_max)
    if x_min >= x_max or x_max - x_min > 10000:
        raise ValueError("Choose an increasing x range no wider than 10,000.")

    points = []
    for index in range(samples):
        x_value = x_min + (x_max - x_min) * index / (samples - 1)
        try:
            y_value = _evaluate(tree, x_value)
        except (ValueError, ZeroDivisionError, OverflowError):
            y_value = None
        points.append({"x": x_value, "y": y_value})
    return {"expression": expression, "x_min": x_min, "x_max": x_max, "points": points}


def basic_calculate() -> float:
    """Prompt for two numbers and a basic arithmetic operator."""
    first = float(input("Enter the first number: "))
    operation = input("Enter an operator (+, -, *, /): ").strip()
    second = float(input("Enter the second number: "))

    operations = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }
    if operation not in operations:
        raise ValueError("Choose one of: +, -, *, /.")
    if operation == "/" and second == 0:
        raise ValueError("Cannot divide by zero.")
    return _checked_number(operations[operation](first, second))


def equation_solver_menu() -> None:
    equation = input("Enter an equation in x (for example, 2*x + 3 = 7): ")
    result = solve_equation(equation)
    if result["kind"] == "identity":
        print("Every real number is a solution.")
    elif result["kind"] == "none":
        print("There are no real solutions.")
    else:
        print("Solution(s): " + ", ".join(f"x = {value:g}" for value in result["solutions"]))


def visualize_search(equation: str = "x + 5 = 12") -> None:
    """Show a brute-force search through integer guesses from -10 to 10."""
    if not isinstance(equation, str) or equation.count("=") != 1:
        raise ValueError("Enter an equation with one equals sign.")
    left_text, right_text = equation.split("=", maxsplit=1)
    left_tree = _parse_expression(left_text)
    right_tree = _parse_expression(right_text)

    print(f"\nBrute-force search for: {equation}")
    print("Trying integer values of x from -10 to 10:")
    for guess in range(-10, 11):
        left_value = _evaluate(left_tree, guess)
        right_value = _evaluate(right_tree, guess)
        print(f"  x = {guess:>3}: left = {left_value:g}, right = {right_value:g}")
        if math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9):
            print(f"Found a solution: x = {guess}")
            return
    print("No integer solution found in this range.")


def _print_search_explanation() -> None:
    print(
        "\nA search algorithm looks through possible answers to find one that "
        "matches a goal. Brute-force search checks candidates one by one; it "
        "can be slow because the number of checks grows with the search space. "
        "Binary search is a smarter approach for ordered data: it checks the "
        "middle and discards half the remaining candidates each step."
    )


PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Calculator</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; color: #172033; background: #f3f6fb; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header { padding: 2rem 1rem 1.5rem; color: white; background: linear-gradient(120deg, #2246a8, #356cdb); }
    header > div, main { width: min(1050px, calc(100% - 2rem)); margin: auto; }
    h1 { margin: 0 0 .35rem; font-size: clamp(1.7rem, 5vw, 2.4rem); }
    header p { margin: 0; color: #e1eaff; }
    main { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 1rem; padding: 1.3rem 0 2rem; }
    section { min-width: 0; padding: 1.2rem; border: 1px solid #e0e6f0; border-radius: 14px; background: white; box-shadow: 0 5px 18px #1720330b; }
    section.graph { grid-column: 1 / -1; }
    h2 { margin: 0 0 .35rem; font-size: 1.2rem; }
    .hint { margin: 0 0 1rem; color: #59677d; font-size: .9rem; }
    label { display: block; margin: .75rem 0 .35rem; font-weight: 600; font-size: .9rem; }
    input { width: 100%; padding: .7rem .75rem; border: 1px solid #cbd4e2; border-radius: 8px; font: inherit; }
    input:focus { outline: 3px solid #356cdb30; border-color: #356cdb; }
    button { margin-top: .8rem; padding: .65rem 1rem; border: 0; border-radius: 8px; background: #2857bd; color: white; font: inherit; font-weight: 650; cursor: pointer; }
    button:hover { background: #1e4599; }
    .result { min-height: 1.6rem; margin: .9rem 0 0; overflow-wrap: anywhere; font-weight: 650; }
    .error { color: #b42318; }
    .range { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
    canvas { display: block; width: 100%; height: 360px; margin-top: 1rem; border: 1px solid #e5eaf2; border-radius: 8px; }
    @media (max-width: 480px) { canvas { height: 280px; } section.graph { grid-column: auto; } }
  </style>
</head>
<body>
  <header><div><h1>Smart Calculator</h1><p>Calculate, solve equations, and explore function graphs.</p></div></header>
  <main>
    <section>
      <h2>Calculator</h2>
      <p class="hint">Supports +, -, *, /, ^, parentheses, pi, e, and functions like sqrt(9) or sin(pi/2).</p>
      <form id="calculate-form">
        <label for="expression">Expression</label>
        <input id="expression" name="expression" placeholder="(12 + 8) / 4" autocomplete="off" required>
        <button type="submit">Calculate</button>
      </form>
      <p id="calculate-result" class="result" aria-live="polite"></p>
    </section>
    <section>
      <h2>Equation solver</h2>
      <p class="hint">Find real solutions for linear and quadratic equations in x.</p>
      <form id="solve-form">
        <label for="equation">Equation</label>
        <input id="equation" name="equation" placeholder="x^2 - 5*x + 6 = 0" autocomplete="off" required>
        <button type="submit">Solve equation</button>
      </form>
      <p id="solve-result" class="result" aria-live="polite"></p>
    </section>
    <section class="graph">
      <h2>Graph search</h2>
      <p class="hint">Enter a function of x; use ^ for powers, for example x^2 - 4 or sin(x).</p>
      <form id="graph-form">
        <label for="graph-expression">Function y =</label>
        <input id="graph-expression" name="expression" placeholder="x^2 - 4" autocomplete="off" required>
        <div class="range">
          <div><label for="x-min">x minimum</label><input id="x-min" type="number" value="-10" step="any"></div>
          <div><label for="x-max">x maximum</label><input id="x-max" type="number" value="10" step="any"></div>
        </div>
        <button type="submit">Plot function</button>
      </form>
      <p id="graph-result" class="result" aria-live="polite"></p>
      <canvas id="plot" aria-label="Function graph"></canvas>
    </section>
  </main>
  <script>
    const formatNumber = value => Number(value.toPrecision(10)).toString();
    async function sendForm(path, values) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The request could not be completed.");
      return data;
    }
    function showResult(element, text, isError = false) {
      element.textContent = text;
      element.classList.toggle("error", isError);
    }
    function reportError(element, error) { showResult(element, error.message, true); }

    document.getElementById("calculate-form").addEventListener("submit", async event => {
      event.preventDefault();
      const output = document.getElementById("calculate-result");
      try {
        const data = await sendForm("/api/calculate", { expression: document.getElementById("expression").value });
        showResult(output, `Result: ${formatNumber(data.result)}`);
      } catch (error) { reportError(output, error); }
    });

    document.getElementById("solve-form").addEventListener("submit", async event => {
      event.preventDefault();
      const output = document.getElementById("solve-result");
      try {
        const data = await sendForm("/api/solve", { equation: document.getElementById("equation").value });
        if (data.kind === "identity") showResult(output, "Every real number is a solution.");
        else if (data.kind === "none") showResult(output, "There are no real solutions.");
        else showResult(output, `Solution${data.solutions.length === 1 ? "" : "s"}: ${data.solutions.map(value => `x = ${formatNumber(value)}`).join(", ")}`);
      } catch (error) { reportError(output, error); }
    });

    const canvas = document.getElementById("plot");
    const context = canvas.getContext("2d");
    let graphData = null;
    function drawGraph() {
      if (!graphData) return;
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth, height = canvas.clientHeight;
      canvas.width = width * ratio; canvas.height = height * ratio;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      const valid = graphData.points.filter(point => point.y !== null && Number.isFinite(point.y));
      if (!valid.length) return;
      let ymin = Math.min(0, ...valid.map(point => point.y));
      let ymax = Math.max(0, ...valid.map(point => point.y));
      if (ymin === ymax) { ymin -= 1; ymax += 1; }
      const padding = (ymax - ymin) * .08;
      ymin -= padding; ymax += padding;
      const left = 48, right = width - 14, top = 14, bottom = height - 32;
      const px = x => left + (x - graphData.x_min) / (graphData.x_max - graphData.x_min) * (right - left);
      const py = y => bottom - (y - ymin) / (ymax - ymin) * (bottom - top);
      context.font = "12px system-ui"; context.lineWidth = 1;
      context.strokeStyle = "#e7ebf2"; context.fillStyle = "#59677d";
      for (let i = 0; i <= 4; i++) {
        const y = top + (bottom - top) * i / 4;
        const value = ymax - (ymax - ymin) * i / 4;
        context.beginPath(); context.moveTo(left, y); context.lineTo(right, y); context.stroke();
        context.fillText(formatNumber(value), 4, y + 4);
      }
      const zeroX = Math.max(graphData.x_min, Math.min(graphData.x_max, 0));
      const zeroY = Math.max(ymin, Math.min(ymax, 0));
      context.strokeStyle = "#9aa7ba";
      context.beginPath(); context.moveTo(px(zeroX), top); context.lineTo(px(zeroX), bottom);
      context.moveTo(left, py(zeroY)); context.lineTo(right, py(zeroY)); context.stroke();
      context.fillText(formatNumber(graphData.x_min), left, height - 8);
      context.fillText(formatNumber(graphData.x_max), right - 35, height - 8);
      context.strokeStyle = "#2857bd"; context.lineWidth = 2.5; context.beginPath();
      let drawing = false;
      for (const point of graphData.points) {
        const visible = point.y !== null && Number.isFinite(point.y) && point.y >= ymin && point.y <= ymax;
        if (!visible) { drawing = false; continue; }
        if (!drawing) { context.moveTo(px(point.x), py(point.y)); drawing = true; }
        else context.lineTo(px(point.x), py(point.y));
      }
      context.stroke();
    }
    document.getElementById("graph-form").addEventListener("submit", async event => {
      event.preventDefault();
      const output = document.getElementById("graph-result");
      try {
        graphData = await sendForm("/api/graph", {
          expression: document.getElementById("graph-expression").value,
          x_min: Number(document.getElementById("x-min").value),
          x_max: Number(document.getElementById("x-max").value)
        });
        drawGraph();
        showResult(output, `Showing y = ${graphData.expression}`);
      } catch (error) { graphData = null; reportError(output, error); }
    });
    window.addEventListener("resize", drawGraph);
  </script>
</body>
</html>
"""


class CalculatorHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request length.") from error
        if not 0 < length <= 8192:
            raise ValueError("Request body must be between 1 and 8192 bytes.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self._send_json(404, {"error": "Page not found."})
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/calculate", "/api/solve", "/api/graph"}:
            self._send_json(404, {"error": "Endpoint not found."})
            return
        try:
            payload = self._read_json()
            if route == "/api/calculate":
                result = {"result": calculate(payload.get("expression"))}
            elif route == "/api/solve":
                result = solve_equation(payload.get("equation"))
            else:
                result = graph_expression(
                    payload.get("expression"),
                    payload.get("x_min", -10),
                    payload.get("x_max", 10),
                    payload.get("samples", 401),
                )
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, result)

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}")


def run_web_server() -> None:
    server = ThreadingHTTPServer((HOST, PORT), CalculatorHandler)
    print(f"Smart Calculator is running at http://{HOST}:{PORT} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Smart Calculator.")
    finally:
        server.server_close()


def main() -> None:
    while True:
        print(
            "\nSmart Calculator\n"
            "1. Basic calculation\n"
            "2. Equation solver\n"
            "3. Search visualisation\n"
            "4. What is a search algorithm?\n"
            "5. Exit"
        )
        choice = input("Choose an option (1-5): ").strip()
        try:
            if choice == "1":
                print(f"Result: {basic_calculate():g}")
            elif choice == "2":
                equation_solver_menu()
            elif choice == "3":
                visualize_search()
            elif choice == "4":
                _print_search_explanation()
            elif choice == "5":
                print("Goodbye!")
                return
            else:
                print("Please choose an option from 1 to 5.")
        except (ValueError, ZeroDivisionError, OverflowError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--web"]:
        run_web_server()
    elif sys.argv[1:]:
        raise SystemExit(f"Usage: {sys.argv[0]} [--web]")
    else:
        main()