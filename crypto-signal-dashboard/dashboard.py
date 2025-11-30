# dashboard.py
import subprocess
import datetime
from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crypto Signal Dashboard</title>
    <style>
        body { font-family: Arial; padding: 20px; background: #111; color: #eee; }
        h1 { color: #4cc9f0; }
        .section { margin-bottom: 35px; padding: 15px; background: #222; border-radius: 10px; }
        .signal { font-size: 22px; padding: 10px; border-radius: 8px; display: inline-block; }
        .buy { background: #0a0; }
        .sell { background: #a00; }
        .hold { background: #666; }
        pre { background: #000; padding: 15px; overflow-x: auto; color: #0f0; }
    </style>
</head>
<body>

<h1>📊 Crypto AI Signal Dashboard</h1>
<p>Last updated: {{ updated }}</p>

<div class="section">
    <h2>📈 Technical Signals</h2>
    <pre>{{ tech }}</pre>
</div>

<div class="section">
    <h2>⛓ On-Chain Signals</h2>
    <pre>{{ onchain }}</pre>
</div>

<div class="section">
    <h2>🎯 Final Recommendation</h2>
    <span class="signal {{ final_class }}">{{ final_signal }}</span>
</div>

</body>
</html>
"""

def run_script(script):
    """Runs a Python script and returns STDOUT as text."""
    try:
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"  # fixes Windows issues with emojis
        )
        
        output = result.stdout.strip()
        return output if output else "⚠️ No output received."
    except Exception as e:
        return f"Error running {script}: {str(e)}"


@app.route("/")
def dashboard():
    tech_output = run_script("c_signal.py")
    onchain_output = run_script("onchain_analyzer2.py")

    # Escape HTML so ASCII arrows and symbols render correctly
    tech_output = tech_output.replace("<", "&lt;")
    onchain_output = onchain_output.replace("<", "&lt;")

    # Decide final recommendation from combined signals
    final = "HOLD"
    if "SELL" in tech_output.upper():
        final = "SELL"
    if "BUY" in tech_output.upper():
        final = "BUY"

    final_class = {
        "SELL": "sell",
        "BUY": "buy",
        "HOLD": "hold",
    }.get(final, "hold")

    return render_template_string(
        HTML_TEMPLATE,
        tech=tech_output,
        onchain=onchain_output,
        final_signal=final,
        final_class=final_class,
        updated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


if __name__ == "__main__":
    print("🚀 Dashboard running at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
