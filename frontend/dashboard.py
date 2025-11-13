"""Streamlit dashboard for the table tennis weakness analyzer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_DEFAULT = "http://127.0.0.1:8000"


st.set_page_config(page_title="Table Tennis Analyzer", layout="wide")
st.title("🏓 桌球影片弱點分析儀表板")

backend_url = st.sidebar.text_input("FastAPI 伺服器位址", API_DEFAULT)
st.sidebar.markdown("---")
st.sidebar.info("上傳影片後將執行整套分析流程，並生成九宮格、雷達圖與建議。")


@st.cache_data(show_spinner=False)
def _load_report(path: str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _post_file(url: str, file_bytes: bytes, filename: str) -> Dict[str, object]:
    files = {"file": (filename, file_bytes, "video/mp4")}
    response = requests.post(url, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def _post(url: str) -> Dict[str, object]:
    response = requests.post(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _get(url: str) -> Dict[str, object]:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


uploaded_file = st.file_uploader("上傳 10-15 秒的桌球影片 (.mp4)", type=["mp4"])

if uploaded_file and st.button("上傳並取得 video_id"):
    with st.spinner("上傳中..."):
        upload_resp = _post_file(f"{backend_url}/api/videos", uploaded_file.read(), uploaded_file.name)
    st.session_state["video_id"] = upload_resp["video_id"]
    st.success(f"上傳成功！video_id = {upload_resp['video_id']}")

video_id = st.session_state.get("video_id")
if video_id:
    st.subheader(f"目前影片 video_id：{video_id}")
    if st.button("執行分析"):
        with st.spinner("分析中，請稍候..."):
            analyze_resp = _post(f"{backend_url}/api/videos/{video_id}/analyze")
        st.session_state["report_path"] = analyze_resp["report_path"]
        st.success("分析完成，報告已生成！")

report_path = st.session_state.get("report_path")
report_data: Dict[str, object] | None = None
if report_path and Path(report_path).exists():
    report_data = _load_report(report_path)
elif video_id:
    try:
        report_data = _get(f"{backend_url}/api/videos/{video_id}/report")
        st.session_state["report_path"] = None
    except Exception as exc:  # pragma: no cover - network errors
        st.warning(f"無法直接從 API 取得報告：{exc}")

if report_data:
    st.success("已載入分析報告！")
    col_summary, col_importance = st.columns([2, 1])
    with col_summary:
        st.markdown("### 摘要")
        st.json(report_data["summary"])
        st.markdown("### 雙方比較")
        st.write(report_data["aggregates"])
    with col_importance:
        st.markdown("### 特徵重要度")
        fi_df = pd.DataFrame(report_data["feature_importances"])
        fig_importance = alt.Chart(fi_df).mark_bar().encode(
            x=alt.X("importance", title="重要度"),
            y=alt.Y("feature", sort="-x", title="特徵"),
        )
        st.altair_chart(fig_importance, use_container_width=True)

    heatmap = np.array(report_data["heatmap"])
    heat_df = pd.DataFrame(
        [
            {"x": x + 1, "y": y + 1, "value": heatmap[y, x]}
            for y in range(heatmap.shape[0])
            for x in range(heatmap.shape[1])
        ]
    )
    heat_chart = alt.Chart(heat_df).mark_rect().encode(
        x="x:O",
        y="y:O",
        color=alt.Color("value:Q", scale=alt.Scale(scheme="orangered"), title="失誤密度"),
        tooltip=["x", "y", alt.Tooltip("value", format=".2f")],
    )
    st.markdown("### 九宮格失誤熱度圖")
    st.altair_chart(heat_chart, use_container_width=True)

    radar_entries = report_data["radar"]
    if radar_entries:
        categories = [entry["metric"] for entry in radar_entries]
        player_a = [entry["player_a"] for entry in radar_entries]
        player_b = [entry["player_b"] for entry in radar_entries]
        categories += categories[:1]
        player_a += player_a[:1]
        player_b += player_b[:1]
        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(r=player_a, theta=categories, fill="toself", name="Player A")
        )
        radar_fig.add_trace(
            go.Scatterpolar(r=player_b, theta=categories, fill="toself", name="Player B")
        )
        radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.markdown("### 雙方差距雷達圖")
        st.plotly_chart(radar_fig, use_container_width=True)

    if report_data.get("stroke_predictions"):
        st.markdown("### 揮拍事件預測")
        st.dataframe(pd.DataFrame(report_data["stroke_predictions"]))

    st.markdown("### 建議")
    for rec in report_data.get("recommendations", []):
        st.info(rec)

else:
    st.info("請先上傳並分析影片以產生報告。")
