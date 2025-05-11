# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import japanize_matplotlib   # ★ 追加：これだけで IPAexGothic が使える
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ----------------- 共通レイアウト -----------------
st.set_page_config(page_title="3D 地層プロット & NN", layout="wide")

# Google Fonts で Noto Sans JP を読み込んで Plotly / スタイルに適用
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Noto Sans JP', sans-serif; }
    .title      { font-size: 36px; font-weight: 700; }
    .subheader  { font-size: 24px; font-weight: 700; }
    .text       { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- アプリ本体 -----------------
def main():
    st.markdown("<h1 class='title'>3D 地層プロット＆ニューラルネットワーク アプリ</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 class='subheader'>設定</h2>", unsafe_allow_html=True)

    # ---------- ファイルアップロード ----------
    st.sidebar.markdown("### トレーニングデータファイルをアップロード")
    uploaded_train_file = st.sidebar.file_uploader("トレーニング用 Excelファイルをアップロード", type=["xlsx", "xls"])

    st.sidebar.markdown("### テストデータファイルをアップロード")
    uploaded_test_file = st.sidebar.file_uploader("テスト用 Excelファイルをアップロード", type=["xlsx", "xls"])

    if uploaded_train_file is None or uploaded_test_file is None:
        st.markdown("<p class='text'>トレーニング用およびテスト用のExcelファイルをアップロードしてください。</p>", unsafe_allow_html=True)
        st.stop()

    try:
        # ---------- Excel 読み込み ----------
        train_excel = pd.ExcelFile(uploaded_train_file)
        test_excel  = pd.ExcelFile(uploaded_test_file)

        selected_train_sheet = st.sidebar.selectbox("トレーニングシート選択", train_excel.sheet_names)
        selected_test_sheet  = st.sidebar.selectbox("テストシート選択",    test_excel.sheet_names)

        train_df = pd.read_excel(uploaded_train_file, sheet_name=selected_train_sheet)
        test_df  = pd.read_excel(uploaded_test_file,  sheet_name=selected_test_sheet)

        # ---------- 緯度・経度・標高列の自動検出 ----------
        def detect_geo(df):
            lat = next((c for c in df.columns if c == "緯度"), None)
            lon = next((c for c in df.columns if c == "経度"), None)
            elev= next((c for c in df.columns if c == "標高"), None)
            return lat, lon, elev

        lat_col, lon_col, elev_col = detect_geo(train_df)
        if None in (lat_col, lon_col, elev_col):
            st.error("緯度・経度・標高の列が見つかりません（列名は '緯度','経度','標高'）")
            st.stop()

        st.info(f"検出された列 ➜ 緯度: **{lat_col}**, 経度: **{lon_col}**, 標高: **{elev_col}**")

        # ------------------ タスク定義 ------------------
        task1 = dict(
            name  = "土質情報による地層区分の予測",
            explan= ["土質区分"],
            target= "地層区分（土質情報）"
        )
        st.sidebar.markdown("### 堆積年代タスクの説明変数を選択")
        option = st.sidebar.radio("", ["N値, 深度, 色調", "N値, 深度, R, G, B"])

        explan2 = ["N値", "深度", "色調"] if option.startswith("N値, 深度, 色調") else ["N値", "深度", "R", "G", "B"]
        task2 = dict(
            name  = "堆積年代による地層区分の予測",
            explan= explan2,
            target= "地層区分（堆積年代）"
        )
        tasks = [task1, task2]

        # ------------------ 学習ループ ------------------
        prediction_results = {}
        for task in tasks:
            missing = [c for c in task["explan"] + [task["target"]] if c not in train_df.columns]
            if missing:
                st.warning(f"タスク **{task['name']}**: 必須列が見つかりません → {missing}")
                continue

            # ----------- 見出し ----------
            color = "red" if task["name"].startswith("土質") else "blue"
            st.markdown(f"<h2 style='color:{color};'>{task['name']}</h2>", unsafe_allow_html=True)

            # ----------- 前処理 ----------
            y      = train_df[task["target"]]
            y_enc  = LabelEncoder().fit_transform(y)
            classes= np.unique(y)

            transformers = []
            if task["explan"]:  # 連続 or 離散混在に対応
                transformers.append(("num", MinMaxScaler(), task["explan"]))
            preproc = ColumnTransformer(transformers, remainder="drop")
            X_train = preproc.fit_transform(train_df)
            X_test  = preproc.transform(test_df)

            # ----------- NN モデル ----------
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation="relu"),
                Dense(32, activation="relu"),
                Dense(len(classes), activation="softmax")
            ])
            model.compile(optimizer=Adam(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_enc, test_size=0.2, random_state=42)
            hist = model.fit(X_tr, y_tr, epochs=500, validation_data=(X_val, y_val),
                             callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
                             verbose=0)
            st.success("学習完了")

            # ----------- 学習曲線 ----------
            fig_lc, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(hist.history["loss"], label="訓練")
            ax[0].plot(hist.history["val_loss"], label="検証")
            ax[0].set_title("損失の推移"); ax[0].legend()

            ax[1].plot(hist.history["accuracy"], label="訓練")
            ax[1].plot(hist.history["val_accuracy"], label="検証")
            ax[1].set_title("精度の推移"); ax[1].legend()
            st.pyplot(fig_lc)

            # ----------- 混同行列 ----------
            y_pred_train = np.argmax(model.predict(X_tr), axis=1)
            cm = confusion_matrix(y_tr, y_pred_train)
            fig_cm, ax_cm = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax_cm)
            ax_cm.set_xlabel("予測"); ax_cm.set_ylabel("実測")
            st.pyplot(fig_cm)

            # ----------- テストデータ予測 ----------
            y_pred = np.argmax(model.predict(X_test), axis=1)
            test_df[f"予測_{task['target']}"] = LabelEncoder().fit(classes).inverse_transform(y_pred)
            prediction_results[task['target']] = test_df[f"予測_{task['target']}"]

            # ----------- 3D Plotly ----------
            fig3d = px.scatter_3d(
                test_df, x=lon_col, y=lat_col, z=elev_col,
                color=f"予測_{task['target']}",
                title=f"3D 地層プロット（{task['target']}）"
            )
            fig3d.update_layout(font_family="Noto Sans JP, IPAexGothic, sans-serif")
            st.plotly_chart(fig3d, use_container_width=True)

        # ------------ 統合結果 ------------
        if all(f"予測_{t['target']}" in test_df for t in tasks):
            st.markdown("<h2 class='subheader'>予測結果の統合</h2>", unsafe_allow_html=True)

            def comb(row):
                age, soil = row["予測_地層区分（堆積年代）"], row["予測_地層区分（土質情報）"]
                return "G" if age == "G" else f"{age}{soil}"

            test_df["予測された地層区分"] = test_df.apply(comb, axis=1)
            st.dataframe(test_df[[lon_col, lat_col, elev_col,
                                  "予測_地層区分（土質情報）",
                                  "予測_地層区分（堆積年代）",
                                  "予測された地層区分"]])

            csv = test_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("予測結果CSVをダウンロード", csv, "prediction_results.csv", "text/csv")

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# ----------------- 実行 -----------------
if __name__ == "__main__":
    main()
