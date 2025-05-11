# app.py  ― 3D 地層プロット & NN ダッシュボード（カード UI 版）
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import seaborn as sns

# ------------------------- ページ設定 & 共通 CSS -------------------------
st.set_page_config(page_title="3D 地層プロット & NN", layout="wide")

# Google Fonts とカード / KPI のスタイル
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }

    /* --- 基本色 --- */
    :root {
      --bg-card: #ffffff;
      --shadow-card: 0 2px 6px rgba(0,0,0,0.08);
      --radius-card: 10px;
      --gap: 1.5rem;
    }

    /* --- セクションカード --- */
    .card {
      background: var(--bg-card);
      padding: 1.5rem;
      border-radius: var(--radius-card);
      box-shadow: var(--shadow-card);
      margin-bottom: var(--gap);
    }

    /* --- KPI (st.metric) の文字サイズ上書き --- */
    div[data-testid="metric-container"] {
      width: 100%;
      margin: 0.5rem 0 1.0rem 0;
    }
    div[data-testid="metric-container"] > label {
      font-size: 1.0rem;
    }
    div[data-testid="metric-container"] .metric-value > div {
      font-size: 2.2rem;          /* ← ここを調整するとさらに大きくできる */
      font-weight: 700;
      color: #ff7b00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------- アプリ本体 -------------------------
def main():
    st.markdown("<h1 style='margin-bottom:0.2rem'>3D 地層プロット＆ニューラルネットワーク</h1>", unsafe_allow_html=True)
    st.markdown("### ファイルをアップロードして学習・予測を実行します")

    # ========== アップローダー (重複 key 回避) ==========
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        f_train = st.file_uploader("トレーニング用 Excel (xlsx/xls)", type=["xlsx", "xls"], key="train_file")
    with col_up2:
        f_test  = st.file_uploader("テスト用 Excel (xlsx/xls)",     type=["xlsx", "xls"], key="test_file")

    if f_train is None or f_test is None:
        st.info("トレーニング用とテスト用の Excel ファイルを両方アップロードしてください。")
        st.stop()

    # ========== Excel 読み込み ==========
    train_excel = pd.ExcelFile(f_train)
    test_excel  = pd.ExcelFile(f_test)

    col_sht1, col_sht2 = st.columns(2)
    with col_sht1:
        sheet_tr = st.selectbox("トレーニングシートを選択", train_excel.sheet_names, key="train_sheet")
    with col_sht2:
        sheet_te = st.selectbox("テストシートを選択",      test_excel.sheet_names,  key="test_sheet")

    df_tr = pd.read_excel(f_train, sheet_name=sheet_tr)
    df_te = pd.read_excel(f_test,  sheet_name=sheet_te)

    # ===== 緯度・経度・標高列の自動検出 =====
    def detect_geo(df):
        lat = next((c for c in df.columns if c == "緯度"), None)
        lon = next((c for c in df.columns if c == "経度"), None)
        z   = next((c for c in df.columns if c == "標高"), None)
        return lat, lon, z

    lat_col, lon_col, elev_col = detect_geo(df_tr)
    if None in (lat_col, lon_col, elev_col):
        st.error("緯度・経度・標高の列が見つかりません（列名は '緯度','経度','標高'）")
        st.stop()

    # ===== タスク定義 =====
    task1 = dict(
        name="土質情報",
        explan=["土質区分"],
        target="地層区分（土質情報）",
        color="red"
    )
    st.markdown("#### 堆積年代タスクの説明変数を選択")
    opt = st.radio("", ["N値, 深度, 色調", "N値, 深度, R, G, B"], horizontal=True, key="explan_radio")
    explan2 = ["N値", "深度", "色調"] if opt.startswith("N値, 深度, 色調") else ["N値", "深度", "R", "G", "B"]
    task2 = dict(
        name="堆積年代",
        explan=explan2,
        target="地層区分（堆積年代）",
        color="blue"
    )
    tasks = [task1, task2]

    # ===== 各タスク処理 =====
    for task in tasks:
        missing = [c for c in task["explan"] + [task["target"]] if c not in df_tr.columns]
        if missing:
            st.warning(f"**{task['name']}** : 必須列がありません → {missing}")
            continue

        # ---------- カード開始 ----------
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)

        st.markdown(f"### <span style='color:{task['color']}'>{task['name']}</span>", unsafe_allow_html=True)

        # ----- 前処理 -----
        y      = df_tr[task["target"]]
        le     = LabelEncoder().fit(y)
        y_enc  = le.transform(y)
        classes= le.classes_

        preproc = ColumnTransformer([("num", MinMaxScaler(), task["explan"])], remainder="drop")
        X_tr_all = preproc.fit_transform(df_tr)
        X_te     = preproc.transform(df_te)

        # ----- NN -----
        model = Sequential([
            Dense(64, input_dim=X_tr_all.shape[1], activation="relu"),
            Dense(32, activation="relu"),
            Dense(len(classes), activation="softmax")
        ])
        model.compile(optimizer=Adam(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        X_tr, X_val, y_tr, y_val = train_test_split(X_tr_all, y_enc, test_size=0.2, random_state=42)

        model.fit(
            X_tr, y_tr,
            epochs=500,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
            verbose=0
        )

        # ----- KPI (トレーニング正解率) -----
        y_pred_tr = np.argmax(model.predict(X_tr), axis=1)
        train_acc = (y_pred_tr == y_tr).mean()
        st.metric(label="トレーニング正解率", value=f"{train_acc:.2%}")

        # ----- 2 カラムでグラフ配置 -----
        col_left, col_right = st.columns([1, 1.1])

        # == 左側: 学習曲線 + 混同行列 ==
        with col_left:
            # 学習曲線
            history = model.history.history
            fig_lc, ax = plt.subplots(1, 2, figsize=(8, 3.5))
            ax[0].plot(history["loss"], label="訓練"); ax[0].plot(history["val_loss"], label="検証")
            ax[0].set_title("損失"); ax[0].legend()
            ax[1].plot(history["accuracy"], label="訓練"); ax[1].plot(history["val_accuracy"], label="検証")
            ax[1].set_title("精度"); ax[1].legend()
            st.pyplot(fig_lc, use_container_width=True)

            # 混同行列
            cm = confusion_matrix(y_tr, y_pred_tr)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=classes, yticklabels=classes, ax=ax_cm)
            ax_cm.set_xlabel("予測"); ax_cm.set_ylabel("実測")
            ax_cm.set_title("混同行列")
            st.pyplot(fig_cm, use_container_width=True)

        # == 右側: 3D Plotly ==
        with col_right:
            # テスト予測
            y_pred_test = np.argmax(model.predict(X_te), axis=1)
            df_te[f"予測_{task['target']}"] = le.inverse_transform(y_pred_test)

            fig3d = px.scatter_3d(
                df_te, x=lon_col, y=lat_col, z=elev_col,
                color=f"予測_{task['target']}",
                title="3D 地層プロット"
            )
            fig3d.update_layout(
                margin=dict(l=0, r=0, t=35, b=0),
                font_family="Noto Sans JP, IPAexGothic, sans-serif"
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # ---------- カード終了 ----------
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== 統合結果 =====
    if all(f"予測_{t['target']}" in df_te for t in tasks):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 予測結果の統合")

        def combine(row):
            age  = row["予測_地層区分（堆積年代）"]
            soil = row["予測_地層区分（土質情報）"]
            return "G" if age == "G" else f"{age}{soil}"

        df_te["予測された地層区分"] = df_te.apply(combine, axis=1)
        st.dataframe(df_te[[lon_col, lat_col, elev_col,
                            "予測_地層区分（土質情報）",
                            "予測_地層区分（堆積年代）",
                            "予測された地層区分"]])

        csv = df_te.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("結果を CSV でダウンロード", csv, "prediction_results.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------- 実行 -------------------------
if __name__ == "__main__":
    main()
