from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import align_features, load_data, load_model, risk_label, apply_dark_theme


def latest_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    latest = df.sort_values("month").groupby("country").tail(1).copy()
    latest_aligned = align_features(latest)
    probs = model.predict_proba(latest_aligned)[:, 1]
    latest["crisis_prob"] = probs
    latest["risk_level"] = [risk_label(p) for p in probs]
    return latest[["country", "month", "crisis_prob", "risk_level"]].sort_values(
        "crisis_prob", ascending=False
    )


def compute_alerts(df: pd.DataFrame, model):
    records = []
    for country, g in df.sort_values("month").groupby("country"):
        tail = g.tail(2)
        if tail.empty:
            continue
        aligned = align_features(tail)
        probs = model.predict_proba(aligned)[:, 1]
        if len(probs) == 1:
            latest_prob, prev_prob = probs[0], None
        else:
            prev_prob, latest_prob = probs[-2], probs[-1]
        change = None if prev_prob is None else latest_prob - prev_prob
        records.append(
            {
                "country": country,
                "latest_prob": latest_prob,
                "prev_prob": prev_prob,
                "change": change,
            }
        )
    alert_df = pd.DataFrame(records)
    if alert_df.empty:
        return None, None, None
    top_risk = alert_df.sort_values("latest_prob", ascending=False).iloc[0]
    inc_df = alert_df.dropna(subset=["change"]).sort_values("change", ascending=False)
    top_increase = inc_df.iloc[0] if not inc_df.empty else None
    return alert_df, top_risk, top_increase


def main():
    st.title("Global Risk")
    df = load_data()
    model = load_model()

    pred_df = latest_predictions(df, model)
    alert_df, top_risk, top_increase = compute_alerts(df, model)

    if top_risk is not None:
        st.warning(
            f"Top Risk: **{top_risk['country']}** at {top_risk['latest_prob']:.1%}",
            icon="⚠️",
        )
    if top_increase is not None and top_increase["change"] > 0:
        st.error(
            f"Rising Risk: **{top_increase['country']}** increased by {top_increase['change']:.1%} since last period",
            icon="⬆️",
        )

    st.subheader("Latest Risk by Country")
    st.dataframe(
        pred_df[["country", "crisis_prob", "risk_level"]].sort_values("crisis_prob", ascending=False).style.format({"crisis_prob": "{:.2%}"}),
        use_container_width=True,
    )

    st.subheader("Top 5 Most Risky Countries")
    top5 = pred_df.head(5)
    fig = px.bar(
        top5,
        x="crisis_prob",
        y="country",
        orientation="h",
        text=top5["crisis_prob"].map(lambda x: f"{x:.1%}"),
        labels={"crisis_prob": "Crisis Probability"},
        color="crisis_prob",
        color_continuous_scale=["green", "orange", "red"],
        title="Top 5 Crisis Probabilities",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        title_font=dict(size=18, color="white"),
        yaxis_title="Country",
        xaxis_title="Crisis Probability",
        xaxis_tickformat=".0%",
        coloraxis_showscale=False,
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
    )
    fig = apply_dark_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Heatmap")
    fig_map = px.choropleth(
        pred_df,
        locations="country",
        locationmode="ISO-3",
        color="crisis_prob",
        color_continuous_scale=[
            [0, "green"],
            [0.5, "yellow"],
            [1, "red"],
        ],
        labels={"crisis_prob": "Crisis Prob"},
        title="Global Crisis Probability",
    )
    fig_map.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        title_font=dict(size=18, color="white"),
        coloraxis_colorbar=dict(tickformat=".0%", title="Crisis Probability"),
        hoverlabel=dict(bgcolor="white"),
        xaxis=dict(showgrid=True, gridcolor="gray"),
        yaxis=dict(showgrid=True, gridcolor="gray"),
        geo=dict(
            bgcolor="#0e1117",
            lakecolor="#0e1117",
            landcolor="#1a1a1a",
        ),
    )
    fig_map = apply_dark_theme(fig_map)
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Full Country Table (sortable)")
    st.dataframe(
        pred_df.sort_values("crisis_prob", ascending=False).style.format({"crisis_prob": "{:.2%}"}),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
