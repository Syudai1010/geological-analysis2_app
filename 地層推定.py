# app.py ― 3D 地層プロット & NN ダッシュボード（統合 3D 付き）
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

# ---------- ページ設定 & 共通 CSS ----------
st.set_page_config(page_title="3D 地層プロット & NN", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }
    :root { --bg-card:#fff; --radius:10px; --shadow:0 2px 6px rgba(0,0,0,.08); --gap:1.5rem; }
    .card { background:var(--bg-card); border-radius:var(--radius); box-shadow:var(--shadow);
            padding:1.5rem; margin-bottom:var(--gap); }
    /* KPI サイズ */
    div[data-testid="metric-container"] .metric-value > div { font-size:2.2rem; font-weight:700; color:#ff7b00; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- アプリ本体 ----------
def main():
    st.markdown("<h1 style='margin-bottom:0.2rem'>3D 地層プロット＆ニューラルネットワーク</h1>", unsafe_allow_html=True)

    # アップロード
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        f_train = st.file_uploader("トレーニング用 Excel (xlsx/xls)", type=["xlsx", "xls"], key="train_file")
    with col_up2:
        f_test  = st.file_uploader("テスト用 Excel (xlsx/xls)",     type=["xlsx", "xls"], key="test_file")
    if f_train is None or f_test is None:
        st.info("両方のファイルをアップロードしてください")
        st.stop()

    # Excel 読込
    train_excel, test_excel = pd.ExcelFile(f_train), pd.ExcelFile(f_test)
    sheet_tr = st.selectbox("トレーニングシート", train_excel.sheet_names, key="sheet_tr")
    sheet_te = st.selectbox("テストシート",      test_excel.sheet_names,  key="sheet_te")
    df_tr, df_te = pd.read_excel(f_train, sheet_tr), pd.read_excel(f_test, sheet_te)

    # 緯度・経度・標高列
    def detect_geo(df):
        return (next((c for c in df.columns if c == k), None) for k in ("緯度","経度","標高"))
    lat_col, lon_col, elev_col = detect_geo(df_tr)
    if None in (lat_col, lon_col, elev_col):
        st.error("緯度・経度・標高の列が見つかりません（列名は '緯度','経度','標高'）")
        st.stop()

    # タスク定義
    task1 = dict(name="土質情報",   explan=["土質区分"],           target="地層区分（土質情報）", color="red")
    st.markdown("#### 堆積年代タスクの説明変数を選択")
    opt = st.radio("", ["N値, 深度, 色調", "N値, 深度, R, G, B"], horizontal=True, key="radio_explan")
    explan2 = ["N値","深度","色調"] if opt.startswith("N値, 深度, 色調") else ["N値","深度","R","G","B"]
    task2 = dict(name="堆積年代",   explan=explan2,               target="地層区分（堆積年代）", color="blue")
    tasks = [task1, task2]

    # ------------- 各タスク（カード表示） -------------
    for task in tasks:
        if any(col not in df_tr.columns for col in task["explan"] + [task["target"]]):
            st.warning(f"{task['name']} : 必須列が不足しています"); continue
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"### <span style='color:{task['color']}'>{task['name']}</span>", unsafe_allow_html=True)

        # 前処理
        y, le = df_tr[task["target"]], LabelEncoder().fit(df_tr[task["target"]])
        y_enc, classes = le.transform(y), le.classes_
        pre = ColumnTransformer([("num", MinMaxScaler(), task["explan"])], remainder="drop")
        X_all, X_te = pre.fit_transform(df_tr), pre.transform(df_te)

        # NN
        model = Sequential([Dense(64, input_dim=X_all.shape[1], activation="relu"),
                            Dense(32, activation="relu"),
                            Dense(len(classes), activation="softmax")])
        model.compile(optimizer=Adam(0.01), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_enc, test_size=0.2, random_state=42)
        model.fit(X_tr, y_tr, epochs=500, validation_data=(X_val,y_val),
                  callbacks=[EarlyStopping("val_loss", patience=20, restore_best_weights=True)], verbose=0)

        # KPI
        y_pred_tr = np.argmax(model.predict(X_tr),1)
        st.metric("トレーニング正解率", f"{(y_pred_tr==y_tr).mean():.2%}")

        # 2 カラム
        col_l, col_r = st.columns([1,1.1])

        # 左: 学習曲線 & 混同行列
        with col_l:
            hist = model.history.history
            fig_l, ax = plt.subplots(1,2, figsize=(8,3.4))
            ax[0].plot(hist["loss"]); ax[0].plot(hist["val_loss"]); ax[0].set_title("損失")
            ax[1].plot(hist["accuracy"]); ax[1].plot(hist["val_accuracy"]); ax[1].set_title("精度")
            st.pyplot(fig_l, use_container_width=True)

            cm = confusion_matrix(y_tr, y_pred_tr)
            fig_c, ax_c = plt.subplots(figsize=(4,3.4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, ax=ax_c, cbar=False)
            ax_c.set_xlabel("予測"); ax_c.set_ylabel("実測"); ax_c.set_title("混同行列")
            st.pyplot(fig_c, use_container_width=True)

        # 右: 新しいデータの予測 3D
        with col_r:
            df_te[f"予測_{task['target']}"] = le.inverse_transform(np.argmax(model.predict(X_te),1))
            fig3d = px.scatter_3d(
                df_te, x=lon_col, y=lat_col, z=elev_col,
                color=f"予測_{task['target']}",
                title="新しいデータの 3D 予測プロット"
            )
            fig3d.update_layout(font_family="Noto Sans JP, IPAexGothic, sans-serif",
                                legend_title_text="予測ラベル")
            st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------- 統合結果カード -------------
    if all(f"予測_{t['target']}" in df_te for t in tasks):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 統合結果 (土質 + 堆積年代)")

        def comb(r):
            return "G" if r["予測_地層区分（堆積年代）"]=="G" else f"{r['予測_地層区分（堆積年代）']}{r['予測_地層区分（土質情報）']}"
        df_te["予測された地層区分"] = df_te.apply(comb, axis=1)

        # --- 3D プロット (統合) ---
        fig_comb = px.scatter_3d(
            df_te, x=lon_col, y=lat_col, z=elev_col,
            color="予測された地層区分",
            title="統合結果の 3D 地層プロット"
        )
        fig_comb.update_layout(font_family="Noto Sans JP, IPAexGothic, sans-serif",
                               legend_title_text="統合ラベル")
        st.plotly_chart(fig_comb, use_container_width=True)

        st.dataframe(df_te[[lon_col, lat_col, elev_col,
                            "予測_地層区分（土質情報）",
                            "予測_地層区分（堆積年代）",
                            "予測された地層区分"]])

        csv = df_te.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("結果を CSV でダウンロード", csv, "prediction_results.csv", "text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- 実行 ----------
if __name__ == "__main__":
    main()
