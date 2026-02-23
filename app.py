import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Pulse", page_icon="💸", layout="centered")

st.title("💸 Pulse: The Mindful Money Hub")
st.write("Welcome to your financial command center.")
st.divider()

# ─────────────────────────────────────────────
# MODULE 0: WIZARD OF OZ ACCOUNT LINKING
# ─────────────────────────────────────────────
st.subheader("🔗 Linked Accounts")

if "accounts_linked" not in st.session_state:
    st.session_state.accounts_linked = False

if not st.session_state.accounts_linked:
    st.info("Securely connect your bank accounts to enable AI analysis.")
    if st.button("Connect CaixaBank & Revolut", type="primary"):
        with st.spinner("Authenticating via secure Open Banking portal..."):
            time.sleep(2.5)
        st.session_state.accounts_linked = True
        st.rerun()

if st.session_state.accounts_linked:
    st.success("✅ Accounts securely linked!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="CaixaBank (Primary)", value="€2,450.00", delta="-€45.00")
    with col2:
        st.metric(label="Revolut (Daily)", value="€340.50", delta="+€12.00")
    with col3:
        st.metric(label="Total Liquidity", value="€2,790.50", delta="Normal")

    if st.button("Disconnect Accounts"):
        st.session_state.accounts_linked = False
        st.rerun()

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🧾 AI Statement Auditor",
    "🎙️ Impulse Confessional",
    "📈 Sliding Doors Projector",
])


# ══════════════════════════════════════════════
# TAB 1 — AI STATEMENT AUDITOR
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🧾 AI Statement Auditor")
    st.write(
        "Upload your bank statement CSV and chat with your AI financial analyst. "
        "Ask anything about your spending habits."
    )

    # ── Prediction / response logic (separated from UI) ──────────────────────
    MOCK_TRANSACTIONS = pd.DataFrame({
        "Date": pd.date_range("2025-05-01", periods=20, freq="3D"),
        "Description": [
            "Glovo Food Delivery", "Netflix", "Zara Online", "Mercadona",
            "Glovo Food Delivery", "Spotify", "H&M", "Gym Membership",
            "Glovo Food Delivery", "Amazon", "Mercadona", "Uber",
            "Glovo Food Delivery", "Apple App Store", "Zara Online",
            "Mercadona", "Cafetería", "Uber", "IKEA", "Glovo Food Delivery",
        ],
        "Category": [
            "Food Delivery", "Subscriptions", "Clothing", "Groceries",
            "Food Delivery", "Subscriptions", "Clothing", "Health",
            "Food Delivery", "Shopping", "Groceries", "Transport",
            "Food Delivery", "Subscriptions", "Clothing",
            "Groceries", "Dining Out", "Transport", "Home", "Food Delivery",
        ],
        "Amount (€)": [
            24.50, 15.99, 79.90, 62.30,
            18.75, 9.99, 45.00, 30.00,
            21.00, 55.00, 58.20, 12.40,
            19.90, 4.99, 120.00,
            47.80, 8.60, 9.80, 149.00, 22.10,
        ],
    })

    def get_ai_response(question: str, df: pd.DataFrame) -> str:
        """Rule-based mock RAG response."""
        q = question.lower()

        if any(w in q for w in ["food delivery", "glovo", "delivery"]):
            total = df[df["Category"] == "Food Delivery"]["Amount (€)"].sum()
            count = df[df["Category"] == "Food Delivery"].shape[0]
            return (
                f"📦 You spent **€{total:.2f}** on food delivery across **{count} orders** this period. "
                f"That averages **€{total/count:.2f} per order**. "
                "Food delivery is your single largest discretionary category — worth reviewing! 🤔"
            )

        if any(w in q for w in ["biggest", "largest", "most expensive", "highest"]):
            row = df.loc[df["Amount (€)"].idxmax()]
            return (
                f"💸 Your biggest single expense was **{row['Description']}** "
                f"(€{row['Amount (€)']:.2f}) on {row['Date'].strftime('%B %d')}. "
                "Was that planned?"
            )

        if any(w in q for w in ["subscription", "netflix", "spotify"]):
            total = df[df["Category"] == "Subscriptions"]["Amount (€)"].sum()
            return (
                f"📺 You're paying **€{total:.2f}/month** in subscriptions. "
                "A quick audit: are you actively using all of them?"
            )

        if any(w in q for w in ["groceries", "supermarket", "mercadona"]):
            total = df[df["Category"] == "Groceries"]["Amount (€)"].sum()
            return (
                f"🛒 Groceries cost you **€{total:.2f}** this period. "
                "That's actually healthy — meal-prepping at home beats food delivery every time!"
            )

        if any(w in q for w in ["clothing", "clothes", "zara", "h&m"]):
            total = df[df["Category"] == "Clothing"]["Amount (€)"].sum()
            return (
                f"👗 You spent **€{total:.2f}** on clothing this period. "
                "Consider a 24-hour rule before any fashion purchase!"
            )

        if any(w in q for w in ["total", "overall", "all", "sum"]):
            total = df["Amount (€)"].sum()
            return (
                f"📊 Your total spending this period was **€{total:.2f}** "
                f"across {len(df)} transactions. "
                "Would you like a breakdown by category?"
            )

        if any(w in q for w in ["category", "breakdown", "split"]):
            summary = df.groupby("Category")["Amount (€)"].sum().sort_values(ascending=False)
            lines = "\n".join([f"- **{cat}**: €{amt:.2f}" for cat, amt in summary.items()])
            return f"📂 Here's your spending by category:\n\n{lines}"

        return (
            "🤖 Great question! Based on your statement, I can see patterns around food delivery, "
            "subscriptions, and clothing. Try asking me something like: "
            "*'What did I spend on food delivery?'* or *'What was my biggest expense?'*"
        )

    # ── UI ────────────────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "statement_loaded" not in st.session_state:
        st.session_state.statement_loaded = False

    uploaded_file = st.file_uploader(
        "Upload your bank statement (.csv)",
        type=["csv"],
        help="Don't have one? Click 'Use Demo Data' below.",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        use_demo = st.button("⚡ Use Demo Data Instead")

    if uploaded_file or use_demo:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = MOCK_TRANSACTIONS

        st.session_state.statement_loaded = True
        st.session_state.statement_df = df

        with st.expander("📄 Preview Statement", expanded=False):
            st.dataframe(df, use_container_width=True)

        if not st.session_state.chat_history:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": (
                    "👋 Statement loaded! I've analysed your transactions. "
                    "Ask me anything — *'What was my biggest expense?'*, "
                    "*'How much on food delivery?'*, or *'Show me a category breakdown.'*"
                ),
            })

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if st.session_state.statement_loaded:
        if prompt := st.chat_input("Ask about your spending…"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analysing your statement…"):
                    time.sleep(1.2)
                response = get_ai_response(prompt, st.session_state.statement_df)
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.info("⬆️ Upload a CSV or use demo data to start chatting with your AI analyst.")


# ══════════════════════════════════════════════
# TAB 2 — VOICE-ACTIVATED IMPULSE CONFESSIONAL
# ══════════════════════════════════════════════
with tab2:
    st.subheader("🎙️ The Impulse Confessional")
    st.write(
        "Feeling the urge to splurge? Record yourself describing the purchase. "
        "Your AI coach will calculate its *real* cost before you tap 'Buy'."
    )

    # ── Prediction logic (separated) ─────────────────────────────────────────
    MOCK_TRANSCRIPTIONS = [
        ("I want to buy a €150 jacket at Zara", 150.0, "jacket", "Zara"),
        ("I'm thinking of getting new AirPods for €249", 249.0, "AirPods", "Apple"),
        ("There's a €89 pair of sneakers I really like", 89.0, "sneakers", "the store"),
        ("I found a €320 handbag at Mango", 320.0, "handbag", "Mango"),
    ]

    def simulate_transcription(audio_bytes) -> dict:
        """Simulate Whisper STT + LLM entity extraction."""
        idx = len(audio_bytes) % len(MOCK_TRANSCRIPTIONS)
        text, amount, item, store = MOCK_TRANSCRIPTIONS[idx]
        return {"text": text, "amount": amount, "item": item, "store": store}

    def calculate_real_cost(amount: float, hourly_wage: float) -> dict:
        hours = amount / hourly_wage
        days  = hours / 8
        return {"hours": round(hours, 1), "days": round(days, 1)}

    # ── UI ────────────────────────────────────────────────────────────────────
    hourly_wage = st.number_input(
        "Your approximate hourly wage (€)",
        min_value=5.0, max_value=500.0, value=15.0, step=1.0,
        help="Used to calculate the 'real' cost of a purchase in working hours.",
    )

    st.markdown("#### 🎤 Record your impulse purchase")
    audio_value = st.audio_input(
        "Speak your desire (e.g. *'I want to buy a €150 jacket'*)"
    )

    if "frozen_amount" not in st.session_state:
        st.session_state.frozen_amount = None
    if "frozen_item" not in st.session_state:
        st.session_state.frozen_item = None

    if audio_value is not None:
        with st.spinner("🔊 Transcribing audio…"):
            time.sleep(1.5)

        result = simulate_transcription(audio_value.getvalue())

        st.success("✅ Transcription complete!")
        st.markdown(f"> *\"{result['text']}\"*")

        with st.spinner("🤖 Calculating true cost…"):
            time.sleep(0.8)

        cost = calculate_real_cost(result["amount"], hourly_wage)
        amount = result["amount"]
        item   = result["item"]

        st.divider()
        st.markdown(f"### 💡 Reality Check: The **{item}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Purchase Price", f"€{amount:.2f}")
        c2.metric("Working Hours", f"{cost['hours']} hrs")
        c3.metric("Working Days", f"{cost['days']} days")

        st.warning(
            f"⏰ That {item} costs you **{cost['hours']} hours** of your life at €{hourly_wage}/hr. "
            "Is it worth it?"
        )

        col_freeze, col_buy = st.columns(2)
        with col_freeze:
            if st.button("🧊 Freeze This Purchase", type="primary", use_container_width=True):
                st.session_state.frozen_amount = amount
                st.session_state.frozen_item   = item
                st.balloons()
                st.success(
                    f"✅ **€{amount:.2f}** frozen! Head to the **Sliding Doors Projector** "
                    "to see how this money could grow. 🚀"
                )
        with col_buy:
            if st.button("🛒 Buy Anyway", use_container_width=True):
                st.error(
                    "💸 Purchase recorded. No judgement — but next time, try freezing it first!"
                )


# ══════════════════════════════════════════════
# TAB 3 — ELI5 "SLIDING DOORS" PROJECTOR
# ══════════════════════════════════════════════
with tab3:
    st.subheader("📈 The 'Sliding Doors' Projector")
    st.write(
        "What if you *invested* that impulse purchase instead? "
        "See the power of compound interest — explained simply."
    )

    # ── Prediction logic (separated) ─────────────────────────────────────────
    RISK_PROFILES = {
        "🐢 Conservative (4% / year)":   {"rate": 0.04, "color": "#2196F3"},
        "⚖️  Balanced (7% / year)":       {"rate": 0.07, "color": "#4CAF50"},
        "🚀 Aggressive (12% / year)":    {"rate": 0.12, "color": "#FF5722"},
    }

    ELI5_EXPLANATIONS = {
        "🐢 Conservative (4% / year)": (
            "Imagine planting an apple seed in good, steady soil. 🌱 "
            "It grows slowly but reliably. You won't become rich overnight, "
            "but in 20 years you'll have a solid apple tree that keeps giving you fruit. "
            "Perfect if you hate surprises."
        ),
        "⚖️  Balanced (7% / year)": (
            "Think of this like a pizza shop. 🍕 Some months are great, some are slow, "
            "but over the years the business steadily grows. "
            "You accept a little uncertainty in exchange for meaningfully more money "
            "at the end. This is the 'just right' porridge of investing."
        ),
        "🚀 Aggressive (12% / year)": (
            "Imagine riding a rollercoaster that, on average, goes UP. 🎢 "
            "Some years your money might shrink — scary! — but over 10-20 years, "
            "it tends to shoot way up. You need a strong stomach and patience, "
            "but the rewards can be enormous."
        ),
    }

    def project_growth(principal: float, annual_rate: float, years: int) -> pd.Series:
        return pd.Series(
            [principal * (1 + annual_rate) ** y for y in range(years + 1)],
            index=range(years + 1),
        )

    # ── UI ────────────────────────────────────────────────────────────────────
    frozen = st.session_state.get("frozen_amount")
    frozen_item = st.session_state.get("frozen_item", "your purchase")

    if frozen is None:
        st.info("💡 No amount frozen yet. Go to **🎙️ Impulse Confessional** and freeze a purchase first!")
        principal = st.number_input(
            "Or enter an amount manually (€)",
            min_value=1.0, max_value=100000.0, value=150.0, step=10.0
        )
    else:
        st.success(f"🧊 Using your frozen amount: **€{frozen:.2f}** ({frozen_item})")
        principal = frozen
        if st.button("🔄 Clear frozen amount"):
            st.session_state.frozen_amount = None
            st.session_state.frozen_item   = None
            st.rerun()

    years = st.slider("⏳ Years to invest", min_value=1, max_value=30, value=10)
    risk_label = st.select_slider(
        "📊 Risk Tolerance",
        options=list(RISK_PROFILES.keys()),
        value="⚖️  Balanced (7% / year)",
    )

    profile = RISK_PROFILES[risk_label]
    rate    = profile["rate"]

    # Build chart with all three profiles for comparison
    fig = go.Figure()
    for label, cfg in RISK_PROFILES.items():
        series = project_growth(principal, cfg["rate"], years)
        dash   = "dot" if label != risk_label else "solid"
        width  = 3    if label == risk_label else 1.5
        fig.add_trace(go.Scatter(
            x=list(series.index),
            y=series.values,
            mode="lines",
            name=label,
            line=dict(color=cfg["color"], width=width, dash=dash),
        ))

    final_value = principal * (1 + rate) ** years
    gain        = final_value - principal

    fig.update_layout(
        title=f"Growth of €{principal:.2f} over {years} years",
        xaxis_title="Year",
        yaxis_title="Portfolio Value (€)",
        yaxis_tickprefix="€",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Key metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("💰 Initial Investment", f"€{principal:.2f}")
    m2.metric(f"📈 Value after {years} yrs", f"€{final_value:,.2f}")
    m3.metric("🎁 Total Gain", f"€{gain:,.2f}", delta=f"+{(gain/principal)*100:.1f}%")

    # ELI5 box
    st.divider()
    st.markdown("#### 🧒 ELI5 — What does this risk level actually mean?")
    st.info(ELI5_EXPLANATIONS[risk_label])

    st.caption(
        "⚠️ This is a simplified projection using compound interest. "
        "Real investments involve risk, fees, and market volatility. Not financial advice."
    )