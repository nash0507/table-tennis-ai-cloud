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

# 指定雲端模型儲存位置（例：S3/R2）
export TTAI_MODEL_REPOSITORY_URI="s3://your-model-bucket/table-tennis-ai"

# 指定 Cloudflare R2 影片資料集（可讀寫）
export TTAI_DATASET_REPOSITORY_URI="s3://table-tennis-dataset"
export TTAI_DATASET_STORAGE_OPTIONS='{"key":"<R2_ACCESS_KEY>","secret":"<R2_SECRET>","client_kwargs":{"endpoint_url":"https://<accountid>.r2.cloudflarestorage.com"}}'

uvicorn backend.app:app --reload
streamlit run frontend/dashboard.py
```

## 範例 API 流程
```bash
curl -F "file=@sample.mp4" http://127.0.0.1:8000/api/videos
curl -X POST http://127.0.0.1:8000/api/videos/<video_id>/analyze
curl http://127.0.0.1:8000/api/videos/<video_id>/report
```

## 雲端模型與 Cloudflare R2 自動學習流程
1. **建立雲端儲存**：
   - 模型：於 Cloudflare R2 或任何 fsspec 支援的物件儲存建立桶位（例：`table-tennis-ai-models`）。
   - 資料集：在 R2 建立第二個桶位（例：`table-tennis-dataset`）儲存原始影片與特徵 CSV。
2. **設定權限與認證**：在 Cloudflare Dashboard → R2 建立 `Access Keys`，取得 `Access Key ID` 與 `Secret Access Key`。
3. **設定環境變數**：於部署環境或本地終端設定：
   ```bash
   export TTAI_MODEL_REPOSITORY_URI="s3://table-tennis-ai-models"
   export TTAI_DATASET_REPOSITORY_URI="s3://table-tennis-dataset"
   export TTAI_DATASET_STORAGE_OPTIONS='{"key":"<R2_ACCESS_KEY>","secret":"<R2_SECRET>","client_kwargs":{"endpoint_url":"https://<accountid>.r2.cloudflarestorage.com"}}'
   ```
   - `endpoint_url` 需替換為 Cloudflare 提供的 `https://<accountid>.r2.cloudflarestorage.com`。
   - 若模型與資料集都放在 R2，可共用同組金鑰；亦可各自配置。
4. **自動訓練流程**：
   - `/api/videos` 上傳的 mp4 會同步寫入本地 `data/raw/` 與 R2 `videos/<video_id>.mp4`。
   - `/api/videos/{video_id}/analyze` 產出的特徵 CSV 會同步寫入 R2 `features/features_<video_id>.csv`。
   - 訓練時會合併本地與 R2 的歷史特徵，再將最新決策樹模型發佈到 `TTAI_MODEL_REPOSITORY_URI`。
   - 連續上傳影片並追加（可選的）`labels.csv` 時，模型會在每次分析後自動學習並提高精度。
5. **版本管理建議**：可在 R2 Bucket 啟用 Object Versioning，或自行以日期區分路徑（例：`models/2024-05-01/decision_tree.pkl`）。

## 操作步驟：自動訓練與部署
1. **安裝依賴與設定環境變數**：在乾淨的虛擬環境中執行 `pip install -r requirements.txt`，並設定：
   - `TTAI_MODEL_REPOSITORY_URI` 指向模型雲端儲存位置（若未設定則使用 `data/model_store`）。
   - `TTAI_DATASET_REPOSITORY_URI` 指向 Cloudflare R2 資料集（若未設定則使用 `data/cloudflare_dataset`）。
   - `TTAI_DATASET_STORAGE_OPTIONS` 為 JSON 字串，內含 `key`、`secret` 與 `client_kwargs.endpoint_url`（本地測試可省略）。
2. **啟動服務**：分別啟動 FastAPI 與 Streamlit（`uvicorn backend.app:app --reload` 與 `streamlit run frontend/dashboard.py`）。
3. **上傳第一支影片並分析**：
   - 以 API 或前端上傳 10–15 秒的 `sample.mp4`。
   - 後端會在儲存本地檔案後，將影片同步到 R2 `videos/<video_id>.mp4`；若同步失敗會回傳 502 以提醒重新上傳或檢查權限。
   - 呼叫 `/api/videos/{video_id}/analyze` 觸發整個 pipeline，並於完成時將 `features_<video_id>.csv` 同步到 R2。
4. **累積更多影片促進自動學習**：
   - 針對第二支、第三支……影片重複步驟 3，R2 `features/` 目錄會累積所有特徵 CSV。
   - 每次分析時 `train_with_history` 會合併本地與 R2 的歷史特徵，以及（可選的）`labels.csv`，重新訓練模型並上傳最新權重。
   - Streamlit 儀表板載入最新報告即可看到模型更新後的指標與建議。
5. **加入人工標註提升精準度**：
   - 在 `data/processed/labels.csv` 追加格式為 `video_id,t_ms,stroke_good` 的標註。
   - 重新呼叫 `/api/videos/{video_id}/analyze`（任一影片）即會觸發重新訓練，並把新的模型上傳至雲端。
6. **驗證雲端模型與資料集是否更新**：
   - 使用 `aws s3 ls s3://table-tennis-ai-models/` 或 `rclone ls r2:table-tennis-dataset/features/` 確認 `.pkl` 與 `features_*.csv` 的時間戳。
   - 若設定為本地測試，可直接檢查 `data/model_store/` 與 `data/cloudflare_dataset/` 內檔案的修改時間。
7. **執行測試**：`pytest` 與 `flake8` 可驗證特徵、分類與指標邏輯皆運作正常。

## Pipeline 說明
1. **/api/videos**：接收 mp4 影片，先寫入 `data/raw/`，再同步到 Cloudflare R2 `videos/` 目錄。
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
2. 每次分析都會重新整理所有 `features_*.csv` 與標註，訓練後再同步至雲端模型儲存。
3. 完成分析後的報告 JSON 可作為教練回饋或持續調整模型的依據。

## 注意事項
- 雲端模型儲存預設由環境變數 `TTAI_MODEL_REPOSITORY_URI` 控制，若未指定則會寫入本機 `data/model_store` 目錄。
- Cloudflare R2 影片與特徵同步由 `TTAI_DATASET_REPOSITORY_URI` 與 `TTAI_DATASET_STORAGE_OPTIONS` 控制，若未設定則落在本機 `data/cloudflare_dataset`。
- 若需 GPU/加速，可將 MediaPipe 切換為對應硬體版本或使用更進階姿態估計器。

## 模型用途與限制
- **主要用途**：本系統的決策樹模型以 `features.py` 生成的揮拍事件特徵為基礎，評估每次出手的手別（正手/反手/發球）與品質（good/bad），並提供 `metrics.py` 聚合後的技術指標與建議。決策樹能給出可解釋的分支規則與 `feature_importances_`，讓教練理解哪些角度或速度特徵對判斷影響最大。
- **資料需求**：需要影片能清楚呈現上肢關節與球拍/球，且建議提供少量人工標註 (`labels.csv`) 以持續微調模型，否則僅依賴預訓練模型可能無法涵蓋所有球風與拍型。
- **精度限制**：
  - MediaPipe/影像啟發式在光線不足、遮擋或低解析度時可能產生錯誤關鍵點，導致角度與落點計算偏差。
  - 九宮格落點與擊球速度僅為幾何估計，無法代表真實球速，建議搭配實際測量資料調整映射。
  - 決策樹對於資料偏態敏感，若標註資料不平衡，預測結果可能偏向常見類別，需透過再抽樣或調參改善。
- **部署與性能**：目前為同步處理流程，分析長時間影片時會造成阻塞，可透過背景工作（例如 Celery）或分段處理提升效能；同時記得清理 `data/raw` 與中間產物以節省磁碟空間。
- **適用範圍**：MVP 主要針對單鏡頭、標準視角之比賽影片，若為多鏡頭或不同角度需擴充 `extract.py` 與特徵工程，以維持判定一致性。
