import os
import json
import re
import time
import pandas as pd
import numpy as np
import cohere
from flask import Flask, render_template, request, jsonify, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ─────────────────────────────────────────────
# COHERE CLIENT
# ─────────────────────────────────────────────
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
co = cohere.ClientV2(COHERE_API_KEY) if COHERE_API_KEY else None

def call_llm(messages, tools=None):
    kwargs = {"model": "command-a-03-2025", "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return co.chat(**kwargs)


# ─────────────────────────────────────────────
# MOCK DATA — 60 days, realistic, with income
# ─────────────────────────────────────────────
_dates = pd.date_range("2026-01-01", periods=74, freq="D")
_txn_data = [
    # ── Income (3 salary entries) ──
    ("2026-01-01", "ESADE Salary", "Income", 2400.00),
    ("2026-02-01", "ESADE Salary", "Income", 2400.00),
    ("2026-03-01", "ESADE Salary", "Income", 2400.00),
    # ── Bizum (~3/month, €55-128 range — incoming transfers) ──
    ("2026-01-08", "Bizum - Maria L.", "Income", 67.00),
    ("2026-01-17", "Bizum - Carlos R.", "Income", 112.50),
    ("2026-01-28", "Bizum - Flat expenses", "Income", 88.00),
    ("2026-02-05", "Bizum - Ana P.", "Income", 55.00),
    ("2026-02-14", "Bizum - Dinner split", "Income", 128.00),
    ("2026-02-22", "Bizum - Flat expenses", "Income", 91.50),
    ("2026-03-03", "Bizum - Ski trip split", "Income", 115.00),
    ("2026-03-09", "Bizum - Carlos R.", "Income", 73.00),
    ("2026-03-14", "Bizum - Concert tickets", "Income", 62.00),
    # ── Rent & Bills ──
    ("2026-01-02", "Rent Transfer - Habitatge BCN", "Rent", -850.00),
    ("2026-02-02", "Rent Transfer - Habitatge BCN", "Rent", -850.00),
    ("2026-03-02", "Rent Transfer - Habitatge BCN", "Rent", -850.00),
    ("2026-01-05", "Endesa Electricity", "Bills", -64.20),
    ("2026-02-05", "Endesa Electricity", "Bills", -71.30),
    ("2026-03-05", "Endesa Electricity", "Bills", -59.80),
    ("2026-01-06", "Vodafone Mobile+Fiber", "Bills", -45.00),
    ("2026-02-06", "Vodafone Mobile+Fiber", "Bills", -45.00),
    ("2026-03-06", "Vodafone Mobile+Fiber", "Bills", -45.00),
    ("2026-01-10", "Naturgy Gas", "Bills", -38.50),
    ("2026-02-10", "Naturgy Gas", "Bills", -42.10),
    ("2026-03-10", "Naturgy Gas", "Bills", -35.60),
    # ── Groceries (~2-3x/week) ──
    ("2026-01-03", "Mercadona Diagonal", "Groceries", -52.30),
    ("2026-01-07", "Bonpreu Gracia", "Groceries", -38.40),
    ("2026-01-11", "Mercadona Diagonal", "Groceries", -61.80),
    ("2026-01-15", "Lidl Eixample", "Groceries", -44.20),
    ("2026-01-19", "Mercadona Diagonal", "Groceries", -57.90),
    ("2026-01-23", "Bonpreu Gracia", "Groceries", -35.60),
    ("2026-01-27", "Mercadona Diagonal", "Groceries", -49.10),
    ("2026-01-31", "Lidl Eixample", "Groceries", -42.70),
    ("2026-02-03", "Mercadona Diagonal", "Groceries", -55.40),
    ("2026-02-07", "Bonpreu Gracia", "Groceries", -41.20),
    ("2026-02-11", "Mercadona Diagonal", "Groceries", -63.50),
    ("2026-02-15", "Lidl Eixample", "Groceries", -39.80),
    ("2026-02-19", "Mercadona Diagonal", "Groceries", -48.60),
    ("2026-02-23", "Bonpreu Gracia", "Groceries", -36.90),
    ("2026-02-27", "Mercadona Diagonal", "Groceries", -54.10),
    ("2026-03-03", "Mercadona Diagonal", "Groceries", -59.20),
    ("2026-03-07", "Bonpreu Gracia", "Groceries", -43.80),
    ("2026-03-11", "Lidl Eixample", "Groceries", -47.50),
    ("2026-03-15", "Mercadona Diagonal", "Groceries", -51.30),
    # ── Food Delivery ──
    ("2026-01-04", "Glovo - Burger King", "Food Delivery", -18.90),
    ("2026-01-09", "Uber Eats - Sushi Shop", "Food Delivery", -26.50),
    ("2026-01-13", "Glovo - Thai Garden", "Food Delivery", -21.40),
    ("2026-01-18", "Glovo - Dominos", "Food Delivery", -19.90),
    ("2026-01-22", "Uber Eats - McDonalds", "Food Delivery", -14.80),
    ("2026-01-26", "Glovo - Wok to Walk", "Food Delivery", -22.30),
    ("2026-02-01", "Uber Eats - Poke House", "Food Delivery", -23.60),
    ("2026-02-06", "Glovo - Burger King", "Food Delivery", -17.40),
    ("2026-02-10", "Glovo - Pizza Hut", "Food Delivery", -20.90),
    ("2026-02-15", "Uber Eats - Five Guys", "Food Delivery", -27.10),
    ("2026-02-20", "Glovo - Thai Garden", "Food Delivery", -19.50),
    ("2026-02-25", "Glovo - Sushi Shop", "Food Delivery", -24.80),
    ("2026-03-02", "Uber Eats - McDonalds", "Food Delivery", -15.90),
    ("2026-03-07", "Glovo - Burger King", "Food Delivery", -18.20),
    ("2026-03-12", "Glovo - Dominos", "Food Delivery", -21.70),
    # ── Clothing ──
    ("2026-01-12", "Zara Online", "Clothing", -89.90),
    ("2026-01-25", "H&M Passeig de Gracia", "Clothing", -45.00),
    ("2026-02-08", "Mango Online", "Clothing", -62.50),
    ("2026-02-18", "Zara Online", "Clothing", -129.00),
    ("2026-03-01", "Pull&Bear Online", "Clothing", -34.95),
    ("2026-03-10", "Zara Online", "Clothing", -79.90),
    # ── Subscriptions ──
    ("2026-01-01", "Netflix", "Subscriptions", -15.99),
    ("2026-01-01", "Spotify Premium", "Subscriptions", -9.99),
    ("2026-01-15", "Apple iCloud+", "Subscriptions", -2.99),
    ("2026-01-20", "ChatGPT Plus", "Subscriptions", -20.00),
    ("2026-02-01", "Netflix", "Subscriptions", -15.99),
    ("2026-02-01", "Spotify Premium", "Subscriptions", -9.99),
    ("2026-02-15", "Apple iCloud+", "Subscriptions", -2.99),
    ("2026-02-20", "ChatGPT Plus", "Subscriptions", -20.00),
    ("2026-03-01", "Netflix", "Subscriptions", -15.99),
    ("2026-03-01", "Spotify Premium", "Subscriptions", -9.99),
    ("2026-03-15", "Apple iCloud+", "Subscriptions", -2.99),
    # ── Transport ──
    ("2026-01-04", "TMB T-casual", "Transport", -11.35),
    ("2026-01-14", "Uber", "Transport", -13.20),
    ("2026-01-24", "Cabify", "Transport", -9.80),
    ("2026-02-04", "TMB T-casual", "Transport", -11.35),
    ("2026-02-13", "Uber", "Transport", -15.60),
    ("2026-02-24", "Cabify", "Transport", -11.40),
    ("2026-03-04", "TMB T-casual", "Transport", -11.35),
    ("2026-03-11", "Uber", "Transport", -12.90),
    # ── Dining Out ──
    ("2026-01-10", "Bar Centric - Born", "Dining Out", -22.50),
    ("2026-01-16", "La Pepita Burger Bar", "Dining Out", -19.80),
    ("2026-01-24", "Flax & Kale", "Dining Out", -31.40),
    ("2026-02-07", "Bar Centric - Born", "Dining Out", -18.60),
    ("2026-02-14", "Can Paixano", "Dining Out", -26.50),
    ("2026-02-21", "La Pepita Burger Bar", "Dining Out", -21.30),
    ("2026-03-01", "Flax & Kale", "Dining Out", -29.80),
    ("2026-03-08", "Bar Centric - Born", "Dining Out", -17.90),
    ("2026-03-14", "Can Paixano", "Dining Out", -24.60),
    # ── Health ──
    ("2026-01-01", "DiR Gym Membership", "Health", -49.90),
    ("2026-02-01", "DiR Gym Membership", "Health", -49.90),
    ("2026-03-01", "DiR Gym Membership", "Health", -49.90),
    ("2026-01-20", "Farmacia Diagonal", "Health", -12.80),
    ("2026-02-18", "Farmacia Diagonal", "Health", -8.50),
    # ── Shopping ──
    ("2026-01-08", "Amazon.es", "Shopping", -67.00),
    ("2026-01-21", "IKEA Online", "Shopping", -149.00),
    ("2026-02-04", "Amazon.es", "Shopping", -42.30),
    ("2026-02-16", "MediaMarkt", "Shopping", -199.00),
    ("2026-03-05", "Amazon.es", "Shopping", -38.90),
    ("2026-03-13", "El Corte Inglés", "Shopping", -85.00),
    # ── Entertainment ──
    ("2026-01-17", "Cinesa Diagonal Mar", "Entertainment", -12.50),
    ("2026-01-30", "Razzmatazz Entry", "Entertainment", -18.00),
    ("2026-02-07", "Sala Apolo Entry", "Entertainment", -15.00),
    ("2026-02-21", "Cinesa Diagonal Mar", "Entertainment", -12.50),
    ("2026-03-07", "Primavera Sound Early", "Entertainment", -85.00),
    ("2026-03-14", "Cinesa Diagonal Mar", "Entertainment", -14.50),
    # ── Coffee & Snacks ──
    ("2026-01-03", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-01-06", "Nomad Coffee", "Coffee", -5.20),
    ("2026-01-10", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-01-14", "Federal Café", "Coffee", -6.80),
    ("2026-01-18", "Nomad Coffee", "Coffee", -5.20),
    ("2026-01-22", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-01-26", "Federal Café", "Coffee", -6.80),
    ("2026-01-30", "Nomad Coffee", "Coffee", -5.20),
    ("2026-02-03", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-02-07", "Federal Café", "Coffee", -6.80),
    ("2026-02-11", "Nomad Coffee", "Coffee", -5.20),
    ("2026-02-15", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-02-19", "Federal Café", "Coffee", -6.80),
    ("2026-02-23", "Nomad Coffee", "Coffee", -5.20),
    ("2026-02-27", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-03-03", "Federal Café", "Coffee", -6.80),
    ("2026-03-07", "Nomad Coffee", "Coffee", -5.20),
    ("2026-03-11", "Satan's Coffee Corner", "Coffee", -4.50),
    ("2026-03-15", "Federal Café", "Coffee", -6.80),
]

MOCK_TRANSACTIONS = pd.DataFrame(_txn_data, columns=["Date", "Description", "Category", "Amount (€)"])
MOCK_TRANSACTIONS["Date"] = pd.to_datetime(MOCK_TRANSACTIONS["Date"])

# Global store for the loaded dataframe (single-user prototype)
app_state = {
    "df": None,
    "chat_history": [],
    "personality": None,
    "top_category": None,
    "top_spend": None,
    "frozen_amount": None,
    "frozen_item": None,
    "monthly_income": None,
    "goals": [
        {"id": 1, "name": "Tokyo Trip", "target": 3000, "saved": 420, "icon": "plane"},
        {"id": 2, "name": "Emergency Fund", "target": 5000, "saved": 1250, "icon": "shield"},
        {"id": 3, "name": "New MacBook", "target": 1800, "saved": 310, "icon": "laptop"},
    ],
    "vault_items": [],
    "impulse_ledger": [],
    "pulse_score": 85,
}


def detect_monthly_income(df):
    """Try to detect monthly income from positive transactions."""
    if "Amount (€)" not in df.columns:
        return None
    income_rows = df[df["Amount (€)"] > 0]
    if len(income_rows) == 0:
        return None
    # Find the largest recurring positive amount (likely salary)
    income_amounts = income_rows.groupby("Description")["Amount (€)"].agg(["mean", "count"])
    income_amounts = income_amounts[income_amounts["count"] >= 1].sort_values("mean", ascending=False)
    if len(income_amounts) > 0:
        return round(income_amounts.iloc[0]["mean"], 2)
    return None


# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────
AUDITOR_TOOLS = [
    {"type": "function", "function": {"name": "sum_by_category", "description": "Calculate total spending for a specific category.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "The spending category to sum."}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "get_largest_expense", "description": "Find the single largest transaction.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_category_breakdown", "description": "Get a full breakdown of total spending per category.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "get_total_spending", "description": "Calculate total amount spent.", "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "count_transactions_by_category", "description": "Count transactions in a category.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "Category to count."}}, "required": ["category"]}}},
    {"type": "function", "function": {"name": "search_transactions", "description": "Search transactions by keyword.", "parameters": {"type": "object", "properties": {"keyword": {"type": "string", "description": "Keyword to search."}}, "required": ["keyword"]}}},
]


def execute_tool(tool_name, arguments, df):
    # Work with expenses only (negative amounts made positive) for spending queries
    expenses = df[df["Amount (€)"] < 0].copy()
    expenses["Amount (€)"] = expenses["Amount (€)"].abs()
    chart_data = None
    if tool_name == "sum_by_category":
        cat = arguments["category"]
        mask = expenses["Category"].str.lower() == cat.lower()
        return json.dumps({"category": cat, "total_eur": round(expenses.loc[mask, "Amount (€)"].sum(), 2), "transaction_count": int(mask.sum())}), None
    elif tool_name == "get_largest_expense":
        row = expenses.loc[expenses["Amount (€)"].idxmax()]
        return json.dumps({"description": row["Description"], "amount_eur": round(row["Amount (€)"], 2), "date": str(row["Date"].date()) if hasattr(row["Date"], "date") else str(row["Date"]), "category": row["Category"]}), None
    elif tool_name == "get_category_breakdown":
        summary = expenses.groupby("Category")["Amount (€)"].sum().sort_values(ascending=False)
        breakdown = [{"category": cat, "total_eur": round(amt, 2)} for cat, amt in summary.items()]
        return json.dumps({"breakdown": breakdown, "grand_total_eur": round(summary.sum(), 2)}), {"type": "breakdown", "data": breakdown}
    elif tool_name == "get_total_spending":
        return json.dumps({"total_eur": round(expenses["Amount (€)"].sum(), 2), "transaction_count": len(expenses)}), None
    elif tool_name == "count_transactions_by_category":
        cat = arguments["category"]
        mask = expenses["Category"].str.lower() == cat.lower()
        return json.dumps({"category": cat, "count": int(mask.sum()), "total_eur": round(expenses.loc[mask, "Amount (€)"].sum(), 2)}), None
    elif tool_name == "search_transactions":
        kw = arguments["keyword"]
        mask = df["Description"].str.contains(kw, case=False, na=False)
        matches = df.loc[mask]
        return json.dumps({"keyword": kw, "matches": [{"description": r["Description"], "amount_eur": round(r["Amount (€)"], 2), "date": str(r["Date"].date()) if hasattr(r["Date"], "date") else str(r["Date"]), "category": r["Category"]} for _, r in matches.iterrows()], "total_eur": round(matches["Amount (€)"].sum(), 2), "count": len(matches)}), None
    return json.dumps({"error": f"Unknown tool: {tool_name}"}), None


def get_ai_response(question, df, chat_history):
    categories = ", ".join(sorted(df["Category"].unique()))
    # Build goal context for the LLM
    goal_names = ", ".join([g["name"] for g in app_state.get("goals", [])])
    messages = [{"role": "system", "content": (
        f"You are Pulse, an empathetic AI financial analyst. Use tools to answer with specific numbers. "
        f"If you spot unhealthy patterns, flag them gently. "
        f"Available categories: {categories}. Statement has {len(df)} transactions.\n\n"
        f"ACTION TAGS — You are a proactive coach. When your analysis reveals an actionable opportunity, "
        f"append ONE of these tags at the very end of your response (after all your text). Never put tags mid-sentence.\n"
        f"- [ACTION:VAULT:<amount>:<item_name>] — suggest locking money in a 48h cooling-off vault. "
        f"Use when you spot a high or repeated discretionary expense (e.g. [ACTION:VAULT:79.90:Zara]).\n"
        f"- [ACTION:GOAL:<amount>:<goal_name>] — suggest moving money toward a savings goal. "
        f"Use when you find unused liquidity or savings potential. The user's goals are: {goal_names}. "
        f"Example: [ACTION:GOAL:100:Emergency Fund]\n"
        f"Only use an action tag when genuinely helpful. Do not force one into every response."
    )}]
    for msg in (chat_history[-6:] if len(chat_history) > 6 else chat_history):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    chart_data = None
    try:
        response = call_llm(messages, tools=AUDITOR_TOOLS)
        if response.message.tool_calls:
            messages.append(response.message)
            for tc in response.message.tool_calls:
                func_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_output, tool_chart = execute_tool(tc.function.name, func_args, df)
                if tool_chart:
                    chart_data = tool_chart
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": [{"type": "document", "document": {"data": tool_output}}]})
            response = call_llm(messages, tools=AUDITOR_TOOLS)
        text = response.message.content[0].text if response.message.content else "Could not generate a response."
        return text, chart_data
    except Exception as e:
        return f"Error: {str(e)}. Try again in a moment (free tier: 5 calls/min).", None


def generate_spending_personality(df):
    total = df["Amount (€)"].sum()
    n_txn = len(df)
    cat_summary = df.groupby("Category")["Amount (€)"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    top_cat = cat_summary.index[0] if len(cat_summary) > 0 else "Unknown"
    top_cat_pct = (cat_summary.iloc[0]["sum"] / total * 100) if total > 0 else 0
    try:
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        avg_per_week = n_txn / max((dates.max() - dates.min()).days / 7, 1) if len(dates) > 1 else n_txn
    except Exception:
        avg_per_week = n_txn

    # Get top merchants for specificity
    top_merchants = df.groupby("Description")["Amount (€)"].sum().sort_values(ascending=False).head(5)
    merchant_lines = "\n".join([f"- {merch}: €{amt:.2f}" for merch, amt in top_merchants.items()])
    cat_breakdown = "\n".join([f"- {cat}: €{row['sum']:.2f} ({row['count']:.0f} transactions)" for cat, row in cat_summary.iterrows()])

    prompt = (
        f"You are Pulse, a sharp fintech app that talks like a smart friend — not a therapist, not a financial advisor. "
        f"Think Revolut's in-app insights or Monzo's spending reports. Direct, slightly cheeky, data-driven.\n\n"
        f"RULES:\n"
        f"- DO NOT flatter the user. Do not frame high spending on non-essentials as 'self-expression', 'lifestyle', or 'treating yourself'. Call it what it is.\n"
        f"- Always use € symbol (never EUR or Euro)\n"
        f"- Reference specific merchant names (Glovo, Zara, Netflix, etc.) not just categories\n"
        f"- Short punchy sentences. No corporate jargon.\n"
        f"- Be specific with numbers. '€106 on 5 Glovo orders' not 'significant food delivery spending'\n"
        f"- The archetype should be blunt and behavioural, 2-4 words max (e.g. 'The Convenience Spender', 'The Scroll-to-Cart Shopper', 'The Subscription Stacker', 'The Comfort Orderer')\n"
        f"- Strength and blind spot should be ONE sentence each, max 15 words\n"
        f"- Tip should be concrete and actionable, not generic advice\n\n"
        f"USER'S DATA:\n"
        f"- Total: €{total:.2f} across {n_txn} transactions ({avg_per_week:.1f}/week)\n"
        f"- Top category: {top_cat} ({top_cat_pct:.1f}% of total)\n"
        f"- Category breakdown:\n{cat_breakdown}\n"
        f"- Top merchants:\n{merchant_lines}\n\n"
        f"Respond EXACTLY in this format (no extra text):\n"
        f"ARCHETYPE: [2-4 word name]\n"
        f"DESCRIPTION: [2-3 punchy sentences using € amounts and merchant names]\n"
        f"STRENGTH: [one sentence, max 15 words]\n"
        f"BLIND_SPOT: [one sentence, max 15 words]\n"
        f"TIP: [one concrete actionable tip with a specific number]"
    )
    try:
        return call_llm([{"role": "user", "content": prompt}]).message.content[0].text
    except Exception:
        return f"ARCHETYPE: The Impulse Explorer\nDESCRIPTION: You dropped €{total:.2f} across {n_txn} transactions. {top_cat} took the biggest hit at {top_cat_pct:.1f}% of your total.\nSTRENGTH: You keep your groceries spending disciplined.\nBLIND_SPOT: {top_cat} is quietly eating your budget.\nTIP: Cap your {top_cat} at €{cat_summary.iloc[0]['sum']*0.7:.0f}/month — that's a 30% cut."


def parse_personality(text):
    result = {}
    for key in ["ARCHETYPE", "DESCRIPTION", "STRENGTH", "BLIND_SPOT", "TIP"]:
        match = re.search(rf"{key}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
        if match:
            result[key.lower()] = match.group(1).strip()
    return result


def extract_impulse_details(text):
    prompt = (
        f"Extract purchase details from: \"{text}\"\n\n"
        f"Respond ONLY with JSON: {{\"item\": \"...\", \"amount\": 0.0, \"store\": \"unknown\"}}"
    )
    try:
        raw = re.sub(r"```json\s*|```\s*", "", call_llm([{"role": "user", "content": prompt}]).message.content[0].text.strip())
        data = json.loads(raw)
        return {"item": data.get("item", "item"), "amount": float(data.get("amount", 0)), "store": data.get("store", "unknown")}
    except Exception:
        price_match = re.search(r"[€]\s*(\d+(?:[.,]\d{1,2})?)", text)
        return {"item": "your item", "amount": float(price_match.group(1).replace(",", ".")) if price_match else 0.0, "store": "unknown"}


def get_eli5(principal, years, risk_label, final_value, gain, item_name, top_cat=None, monthly_spend=None):
    rates = {"Conservative (4%)": 4, "Balanced (7%)": 7, "Aggressive (12%)": 12}
    rate_pct = rates.get(risk_label, 7)
    context = f"\nContext: Their top discretionary spending category is {top_cat} at €{monthly_spend:.2f}.\n" if top_cat and monthly_spend else ""
    prompt = (
        f"You are Pulse, a sharp fintech app advisor. Not a children's storyteller.\n\n"
        f"The user is considering investing €{principal:.2f}"
        f"{' (instead of buying ' + item_name + ')' if item_name and item_name != 'your purchase' else ''}.\n\n"
        f"Scenario: €{principal:.2f} at {rate_pct}%/year for {years} years = €{final_value:,.2f} (gain: €{gain:,.2f}, +{(gain/principal)*100:.1f}%)\n"
        f"{context}\n"
        f"RULES:\n"
        f"- Use € symbol, never EUR\n"
        f"- NO fairy tales, seeds, gardens, snowballs, or magic analogies\n"
        f"- Talk like a sharp fintech app — direct, specific, numbers-first\n"
        f"- Compare the gain to something concrete (months of rent, number of dinners out, a specific purchase they could afford)\n"
        f"- 2-3 sentences max. Punchy. End with one direct line about why this matters.\n"
        f"- No asterisks or markdown formatting."
    )
    try:
        r = call_llm([{"role": "user", "content": prompt}])
        return r.message.content[0].text if r.message.content else "Your money grows while you sleep. That's the whole point."
    except Exception:
        return f"€{principal:.2f} at {rate_pct}% annual return becomes €{final_value:,.2f} in {years} years. That's €{gain:,.2f} you didn't have to work for. Time is the only ingredient you can't buy — start now."


def scan_impulse_purchases(df):
    """Use the LLM to identify 2-4 past purchases that look impulsive."""
    expenses = df[df["Amount (€)"] < 0].copy()
    expenses["Amount (€)"] = expenses["Amount (€)"].abs()
    # Filter to discretionary only
    ESSENTIAL = {"Income", "Rent", "Bills"}
    discretionary = expenses[~expenses["Category"].isin(ESSENTIAL)]
    if len(discretionary) == 0:
        return []

    # Build a summary for the LLM
    txn_lines = []
    for _, r in discretionary.sort_values("Amount (€)", ascending=False).head(30).iterrows():
        txn_lines.append(f"- {r['Date'].strftime('%Y-%m-%d') if hasattr(r['Date'], 'strftime') else r['Date']}: {r['Description']} ({r['Category']}) €{r['Amount (€)']:.2f}")
    txn_text = "\n".join(txn_lines)

    prompt = (
        f"You are Pulse, a fintech AI. Scan these transactions and identify 2-4 that look like impulse or unnecessary purchases — "
        f"things the user might regret or could have skipped. Focus on one-off high amounts, repeated convenience spending, "
        f"or luxury items.\n\n{txn_text}\n\n"
        f"Respond ONLY with a JSON array. Each object: {{\"description\": \"...\", \"amount\": 0.00, \"date\": \"YYYY-MM-DD\", \"category\": \"...\", \"reason\": \"short 8-word reason\"}}. "
        f"Return 2-4 items. No extra text."
    )
    try:
        raw = call_llm([{"role": "user", "content": prompt}]).message.content[0].text.strip()
        cleaned = re.sub(r"```json\s*|```\s*", "", raw)
        items = json.loads(cleaned)
        return items if isinstance(items, list) else []
    except Exception:
        # Fallback: pick top 3 discretionary by amount
        top = discretionary.sort_values("Amount (€)", ascending=False).head(3)
        return [
            {"description": r["Description"], "amount": round(r["Amount (€)"], 2),
             "date": r["Date"].strftime("%Y-%m-%d") if hasattr(r["Date"], "strftime") else str(r["Date"]),
             "category": r["Category"], "reason": "High discretionary spend flagged"}
            for _, r in top.iterrows()
        ]


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-demo", methods=["POST"])
def load_demo():
    df = MOCK_TRANSACTIONS.copy()
    app_state["df"] = df
    app_state["chat_history"] = []

    # Detect monthly income
    app_state["monthly_income"] = detect_monthly_income(df)

    # Compute top category (discretionary expenses only)
    ESSENTIAL_CATEGORIES = {"Income", "Rent", "Bills"}
    expenses = df[df["Amount (€)"] < 0].copy()
    expenses["Amount (€)"] = expenses["Amount (€)"].abs()
    discretionary = expenses[~expenses["Category"].isin(ESSENTIAL_CATEGORIES)]
    cat_totals = discretionary.groupby("Category")["Amount (€)"].sum().sort_values(ascending=False)
    app_state["top_category"] = cat_totals.index[0] if len(cat_totals) > 0 else None
    app_state["top_spend"] = round(cat_totals.iloc[0], 2) if len(cat_totals) > 0 else None

    # Generate personality (on discretionary expenses only)
    raw = generate_spending_personality(discretionary)
    app_state["personality"] = parse_personality(raw)

    # Compute insights (discretionary only)
    total_expenses = round(discretionary["Amount (€)"].sum(), 2)
    top_3 = cat_totals.head(3)
    insights = []
    for cat, amt in top_3.items():
        pct = round(amt / total_expenses * 100, 1)
        count = len(expenses[expenses["Category"] == cat])
        insights.append({"category": cat, "amount": round(amt, 2), "pct": pct, "count": count})

    # Scan for impulse purchases (LLM-powered)
    flagged_impulses = scan_impulse_purchases(df)

    # Return data
    transactions = []
    for _, row in df.iterrows():
        transactions.append({
            "date": row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"]),
            "description": row["Description"],
            "category": row["Category"],
            "amount": round(row["Amount (€)"], 2),
        })

    return jsonify({
        "success": True,
        "transactions": transactions,
        "personality": app_state["personality"],
        "top_category": app_state["top_category"],
        "top_spend": app_state["top_spend"],
        "monthly_income": app_state["monthly_income"],
        "total_expenses": total_expenses,
        "insights": insights,
        "flagged_impulses": flagged_impulses,
    })


@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    try:
        df = pd.read_csv(file)
        # Basic column check
        required = {"Date", "Description", "Category", "Amount (€)"}
        if not required.issubset(set(df.columns)):
            # Try LLM mapping
            try:
                prompt = (
                    f"Map these CSV columns: {list(df.columns)} to: Date, Description, Category, Amount (€). "
                    f"Respond ONLY with JSON mapping. Null if no match."
                )
                raw = re.sub(r"```json\s*|```\s*", "", call_llm([{"role": "user", "content": prompt}]).message.content[0].text.strip())
                mapping = json.loads(raw)
                df = df.rename(columns={k: v for k, v in mapping.items() if v and v != "null"})
            except Exception:
                pass

        if "Amount (€)" in df.columns:
            df["Amount (€)"] = pd.to_numeric(df["Amount (€)"], errors="coerce").abs()

        app_state["df"] = df
        app_state["chat_history"] = []

        if "Category" in df.columns and "Amount (€)" in df.columns:
            cat_totals = df.groupby("Category")["Amount (€)"].sum().sort_values(ascending=False)
            app_state["top_category"] = cat_totals.index[0] if len(cat_totals) > 0 else None
            app_state["top_spend"] = round(cat_totals.iloc[0], 2) if len(cat_totals) > 0 else None

        raw = generate_spending_personality(df)
        app_state["personality"] = parse_personality(raw)

        transactions = []
        for _, row in df.head(50).iterrows():
            transactions.append({
                "date": str(row.get("Date", "")),
                "description": str(row.get("Description", "")),
                "category": str(row.get("Category", "")),
                "amount": round(float(row.get("Amount (€)", 0)), 2),
            })

        return jsonify({
            "success": True,
            "transactions": transactions,
            "personality": app_state["personality"],
            "top_category": app_state["top_category"],
            "top_spend": app_state["top_spend"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("message", "")
    if not app_state["df"] is not None:
        return jsonify({"error": "No statement loaded"}), 400

    app_state["chat_history"].append({"role": "user", "content": question})
    response_text, chart_data = get_ai_response(question, app_state["df"], app_state["chat_history"][:-1])
    app_state["chat_history"].append({"role": "assistant", "content": response_text})

    return jsonify({
        "response": response_text,
        "chart_data": chart_data,
    })


@app.route("/api/extract-impulse", methods=["POST"])
def extract_impulse():
    data = request.json
    text = data.get("text", "")
    result = extract_impulse_details(text)
    return jsonify(result)


@app.route("/api/freeze", methods=["POST"])
def freeze():
    data = request.json
    app_state["frozen_amount"] = data.get("amount", 0)
    app_state["frozen_item"] = data.get("item", "")
    return jsonify({"success": True})


@app.route("/api/get-frozen", methods=["GET"])
def get_frozen():
    return jsonify({
        "amount": app_state["frozen_amount"],
        "item": app_state["frozen_item"],
        "top_category": app_state["top_category"],
        "top_spend": app_state["top_spend"],
    })


@app.route("/api/clear-frozen", methods=["POST"])
def clear_frozen():
    app_state["frozen_amount"] = None
    app_state["frozen_item"] = None
    return jsonify({"success": True})


@app.route("/api/impulse/succumb", methods=["POST"])
def impulse_succumb():
    data = request.json
    item = data.get("item", "unknown")
    amount = data.get("amount", 0)
    app_state["impulse_ledger"].append({
        "item": item,
        "amount": amount,
        "status": "succumbed",
        "goal": None,
        "timestamp": datetime.now().isoformat(),
    })
    app_state["pulse_score"] = max(0, app_state["pulse_score"] - 5)
    return jsonify({"success": True, "pulse_score": app_state["pulse_score"]})


@app.route("/api/vault/remove", methods=["POST"])
def remove_vault_item():
    data = request.json
    index = data.get("index", -1)
    if 0 <= index < len(app_state["vault_items"]):
        removed = app_state["vault_items"].pop(index)
        # Subtract from the goal it was allocated to
        for g in app_state["goals"]:
            if g["id"] == removed.get("goal_id"):
                g["saved"] = max(0, round(g["saved"] - removed["amount"], 2))
                break
        # Reverse the pulse score bump
        app_state["pulse_score"] = max(0, app_state["pulse_score"] - 5)
    return jsonify({"success": True, "goals": app_state["goals"], "vault": app_state["vault_items"], "pulse_score": app_state["pulse_score"]})


@app.route("/api/eli5", methods=["POST"])
def eli5():
    data = request.json
    principal = float(data.get("principal", 150))
    years = int(data.get("years", 10))
    risk_label = data.get("risk", "Balanced (7%)")
    rates = {"Conservative (4%)": 0.04, "Balanced (7%)": 0.07, "Aggressive (12%)": 0.12}
    rate = rates.get(risk_label, 0.07)
    final_value = principal * (1 + rate) ** years
    gain = final_value - principal

    # Projection data for all profiles
    projections = {}
    for label, r in rates.items():
        projections[label] = [round(principal * (1 + r) ** y, 2) for y in range(years + 1)]

    text = get_eli5(
        principal, years, risk_label, final_value, gain,
        app_state.get("frozen_item") or "your purchase",
        app_state.get("top_category"),
        app_state.get("top_spend"),
    )

    return jsonify({
        "text": text,
        "final_value": round(final_value, 2),
        "gain": round(gain, 2),
        "gain_pct": round((gain / principal) * 100, 1),
        "projections": projections,
    })


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify({
        "has_data": app_state["df"] is not None,
        "personality": app_state["personality"],
        "top_category": app_state["top_category"],
        "top_spend": app_state["top_spend"],
        "frozen_amount": app_state["frozen_amount"],
        "frozen_item": app_state["frozen_item"],
        "monthly_income": app_state["monthly_income"],
        "pulse_score": app_state["pulse_score"],
        "impulse_ledger": app_state["impulse_ledger"],
    })


@app.route("/api/goals", methods=["GET"])
def get_goals():
    return jsonify({"goals": app_state["goals"], "vault": app_state["vault_items"]})


@app.route("/api/goals/allocate", methods=["POST"])
def allocate_to_goal():
    data = request.json
    goal_id = data.get("goal_id")
    amount = data.get("amount", 0)
    item = data.get("item", "")

    goal_name = ""
    for g in app_state["goals"]:
        if g["id"] == goal_id:
            g["saved"] = round(g["saved"] + amount, 2)
            goal_name = g["name"]
            break

    # Add to vault with 48h expiry concept
    app_state["vault_items"].append({
        "item": item,
        "amount": amount,
        "goal_id": goal_id,
        "timestamp": datetime.now().isoformat(),
    })

    # Log to impulse ledger as intercepted
    app_state["impulse_ledger"].append({
        "item": item,
        "amount": amount,
        "status": "intercepted",
        "goal": goal_name,
        "timestamp": datetime.now().isoformat(),
    })

    # Bump pulse score
    app_state["pulse_score"] = min(100, app_state["pulse_score"] + 5)

    # Clear frozen
    app_state["frozen_amount"] = None
    app_state["frozen_item"] = None

    return jsonify({"success": True, "goals": app_state["goals"], "pulse_score": app_state["pulse_score"]})


@app.route("/api/goals/update", methods=["POST"])
def update_goal():
    data = request.json
    goal_id = data.get("id")
    for g in app_state["goals"]:
        if g["id"] == goal_id:
            if "name" in data:
                g["name"] = data["name"]
            if "target" in data:
                g["target"] = data["target"]
            break
    return jsonify({"success": True, "goals": app_state["goals"]})


if __name__ == "__main__":
    if not COHERE_API_KEY:
        print("WARNING: Set COHERE_API_KEY environment variable!")
        print("Run: export COHERE_API_KEY='your-key-here'")
    app.run(debug=True, port=5000)
