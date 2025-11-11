# Table Tennis AI Cloud

## 驗收條件
1. 上傳 10–15 秒 `sample.mp4` 至 `/api/videos` 並呼叫 `/api/videos/{video_id}/analyze` 後會在 `data/reports/` 生成報告 JSON。
2. Streamlit 儀表板可顯示九宮格熱圖、雙方差距雷達圖，且建議至少 3 條。
3. `pytest` 與 `flake8` 測試全部通過。

---

## 專案簡介
本專案提供桌球影片弱點分析的 MVP，後端使用 FastAPI 串接 OpenCV/MediaPipe 與 scikit-learn 進行關鍵點擷取、特徵工程與決策樹分類，前端以 Streamlit 呈現九宮格熱圖、雷達圖與訓練建議。

## 專案結構
```
.
├─ backend/
│  ├─ app.py                 # FastAPI：上傳/分析/回傳報告
│  ├─ config.py              # 路徑、參數設定
│  ├─ db.py                  # CSV/JSON 儲存
│  ├─ schemas.py             # Pydantic models
│  ├─ pipeline/
│  │  ├─ extract.py          # 影片→關鍵點/事件擷取
│  │  ├─ features.py         # 特徵工程與手別推論
│  │  ├─ classify.py         # 決策樹訓練/推論
│  │  ├─ metrics.py          # 指標彙整與建議
│  │  └─ utils.py            # 通用工具
│  └─ models/
│     └─ (runtime generated `.pkl` models)
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ reports/
├─ frontend/
│  └─ dashboard.py           # Streamlit 儀表板
├─ tests/
│  ├─ test_features.py
│  ├─ test_classify.py
│  └─ test_metrics.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ .github/workflows/ci.yml
```

## 安裝與執行
```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload
streamlit run frontend/dashboard.py
```

## 範例 API 流程
```bash
curl -F "file=@sample.mp4" http://127.0.0.1:8000/api/videos
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/analyze
curl http://127.0.0.1:8000/api/videos/<video_id>/report
```

## Pipeline 說明
1. **/api/videos**：接收 mp4 影片並儲存於 `data/raw/`。
2. **/api/videos/{video_id}/analyze**：
   - `extract.py` 以 MediaPipe（若無則使用啟發式）擷取肩肘腕與球的軌跡並切割成揮拍事件。
   - `features.py` 計算幾何角度、揮拍方向、九宮格落點、發/接球旗標等特徵。
   - `classify.py` 使用決策樹訓練/載入模型預測正反手與好球率，同時輸出特徵重要度。
   - `metrics.py` 彙整雙方比重、失誤熱圖、雷達指標並生成 3–5 條訓練建議。
   - 報告以 JSON 儲存於 `data/reports/`。
3. **/api/videos/{video_id}/report**：回傳已生成的 JSON 報告。

## Streamlit 儀表板
- 上傳影片並取得 `video_id`。
- 呼叫分析 API，顯示報告摘要。
- 提供九宮格熱圖、雷達圖、特徵重要度條形圖與建議卡片。

## 測試
```bash
pytest
flake8
```

## 標註與微調流程
1. 將人工標註結果追加於 `data/processed/labels.csv`，格式：`video_id,t_ms,stroke_good`。
2. 重新呼叫 `/api/videos/{video_id}/analyze` 時會自動合併標註並重新訓練決策樹。
3. 完成分析後的報告 JSON 可作為教練回饋或持續調整模型的依據。

## 注意事項
- 首次分析時若 `backend/models/` 下沒有模型，系統會依據當前特徵自動訓練並於該資料夾產生 `.pkl` 檔案。
- 若需 GPU/加速，可將 MediaPipe 切換為對應硬體版本或使用更進階姿態估計器。
