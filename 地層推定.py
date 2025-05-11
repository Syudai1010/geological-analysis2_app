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
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os

# IPAゴシックフォントのパスを明示的に設定 (Streamlit Cloud環境のフォントパス)
font_dirs = ['/usr/share/fonts/opentype/ipafont-gothic']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'IPAexGothic'
plt.rcParams['axes.unicode_minus'] = False


# ▼▼▼ 日本語フォント設定（IPAexGothicを使用） ▼▼▼
plt.rcParams["font.sans-serif"] = ["IPAexGothic"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # マイナス記号の文字化け対策
# ▲▲▲


def main():
    # スタイル設定
    st.markdown(
        """
        <style>
        .title {
            color: black;
            font-size: 36px;
            font-weight: bold;
        }
        .subheader {
            color: black;
            font-size: 24px;
            font-weight: bold;
        }
        .text {
            color: black;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # タイトル表示
    st.markdown("<h1 class='title'>3D 地層プロット＆ニューラルネットワーク アプリ</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 class='subheader'>設定</h2>", unsafe_allow_html=True)

    # ファイルアップロード
    st.sidebar.markdown("### トレーニングデータファイルをアップロード")
    uploaded_train_file = st.sidebar.file_uploader("トレーニング用 Excelファイルをアップロード", type=["xlsx", "xls"])

    st.sidebar.markdown("### テストデータファイルをアップロード")
    uploaded_test_file = st.sidebar.file_uploader("テスト用 Excelファイルをアップロード", type=["xlsx", "xls"])

    if uploaded_train_file is not None and uploaded_test_file is not None:
        try:
            # トレーニングデータのExcelファイルを読み込む
            train_excel = pd.ExcelFile(uploaded_train_file)
            train_sheet_names = train_excel.sheet_names

            # テストデータのExcelファイルを読み込む
            test_excel = pd.ExcelFile(uploaded_test_file)
            test_sheet_names = test_excel.sheet_names

            # シート選択のためのセレクトボックスをサイドバーに追加
            st.sidebar.markdown("### トレーニングデータのシートを選択")
            selected_train_sheet = st.sidebar.selectbox("トレーニングデータのシートを選択", train_sheet_names)

            st.sidebar.markdown("### テストデータのシートを選択")
            selected_test_sheet = st.sidebar.selectbox("テストデータのシートを選択", test_sheet_names)

            # 選択されたシートを読み込む
            train_df = pd.read_excel(uploaded_train_file, sheet_name=selected_train_sheet)
            test_df = pd.read_excel(uploaded_test_file, sheet_name=selected_test_sheet)

            # 自動で緯度、経度、標高の列を検出
            def detect_geospatial_columns(df):
                lat_col = None
                lon_col = None
                elev_col = None
                for col in df.columns:
                    if col == '緯度':
                        lat_col = col
                    if col == '経度':
                        lon_col = col
                    if col == '標高':
                        elev_col = col
                return lat_col, lon_col, elev_col

            lat_col, lon_col, elev_col = detect_geospatial_columns(train_df)

            if not all([lat_col, lon_col, elev_col]):
                st.error("緯度、経度、標高の列がデータに存在しません。列名が '緯度'、'経度'、'標高' であることを確認してください。")
                return

            st.markdown(f"### 自動検出された地理情報列:\n- 緯度: {lat_col}\n- 経度: {lon_col}\n- 標高: {elev_col}")

            # タスク1の設定
            task1 = {
                'name': '土質情報による地層区分の予測',
                'explanatory': ['土質区分'],  # 説明変数
                'target': '地層区分（土質情報）'  # 目的変数
            }

            # タスク2の説明変数選択オプションの追加
            st.sidebar.markdown("### 堆積年代による地層区分の予測 説明変数の選択")
            task2_options = ['N値, 深度, 色調', 'N値, 深度, R, G, B']
            selected_task2_option = st.sidebar.radio(
                "堆積年代による地層区分の予測に使用する説明変数を選択してください",
                task2_options
            )

            if selected_task2_option == 'N値, 深度, 色調':
                task2_explanatory = ['N値', '深度', '色調']
            elif selected_task2_option == 'N値, 深度, R, G, B':
                task2_explanatory = ['N値', '深度', 'R', 'G', 'B']
            else:
                task2_explanatory = []

            task2 = {
                'name': '堆積年代による地層区分の予測',
                'explanatory': task2_explanatory,
                'target': '地層区分（堆積年代）'
            }

            tasks = [task1, task2]
            prediction_results = {}

            for task in tasks:
                if task['name'] == '堆積年代による地層区分の予測' and not task['explanatory']:
                    st.warning(f"タスク '{task['name']}' に使用する説明変数が選択されていません。")
                    continue

                # ★★ 見出しを追加（カラー指定） ★★
                if task['name'] == '土質情報による地層区分の予測':
                    st.markdown("<h2 style='color:red;'>土質情報</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='color:blue;'>堆積年代</h2>", unsafe_allow_html=True)

                st.markdown(f"<h3 class='subheader'>{task['name']}</h3>", unsafe_allow_html=True)

                explanatory = task['explanatory']
                target = task['target']

                missing_cols = [col for col in explanatory + [target] if col not in train_df.columns]
                if missing_cols:
                    st.error(f"タスク '{task['name']}' に必要な列がデータに存在しません: {missing_cols}")
                    continue

                # 土質情報タスク → 離散特徴量、堆積年代タスク → 連続特徴量として扱う例
                if task['name'] == '土質情報による地層区分の予測':
                    explanatory_discrete = explanatory
                    explanatory_continuous = []
                else:
                    explanatory_continuous = explanatory
                    explanatory_discrete = []

                ml_df = train_df.copy()
                X_continuous = ml_df[explanatory_continuous] if explanatory_continuous else pd.DataFrame()
                X_discrete = ml_df[explanatory_discrete] if explanatory_discrete else pd.DataFrame()
                y = ml_df[target]

                # ラベルエンコーディング
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                num_classes = len(label_encoder.classes_)

                # 前処理パイプライン
                transformers = []
                if not X_continuous.empty:
                    transformers.append(('minmax', MinMaxScaler(), explanatory_continuous))
                if not X_discrete.empty:
                    transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore'), explanatory_discrete))

                preprocessor = ColumnTransformer(transformers=transformers)
                X_train = preprocessor.fit_transform(ml_df)
                y_train = y_encoded

                # テストデータにも同じ前処理を適用
                X_test = preprocessor.transform(test_df)
                y_test = label_encoder.transform(test_df[target])

                # ニューラルネットワークの構築
                input_dim = X_train.shape[1]
                model = Sequential()
                model.add(Dense(64, input_dim=input_dim, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))

                optimizer = Adam(learning_rate=0.01)
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

                # トレーニング＆検証用に分割
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                # トレーニング
                with st.spinner(f"{task['name']} をトレーニング中..."):
                    history = model.fit(
                        X_train_split, y_train_split,
                        epochs=500,
                        validation_data=(X_val_split, y_val_split),
                        callbacks=[early_stop],
                        verbose=0
                    )
                st.success(f"{task['name']} のトレーニングが完了しました！")

                # 学習曲線の可視化（Matplotlib）
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].plot(history.history['loss'], label='損失')
                if 'val_loss' in history.history:
                    ax[0].plot(history.history['val_loss'], label='検証損失')
                ax[0].set_title('損失の推移')
                ax[0].legend()
                ax[1].plot(history.history['accuracy'], label='精度')
                if 'val_accuracy' in history.history:
                    ax[1].plot(history.history['val_accuracy'], label='検証精度')
                ax[1].set_title('精度の推移')
                ax[1].legend()
                st.pyplot(fig)

                # トレーニングデータでの正解率
                train_predictions = model.predict(X_train_split)
                train_predicted_classes = np.argmax(train_predictions, axis=1)
                train_accuracy = np.mean(train_predicted_classes == y_train_split)

                st.markdown("## モデルの性能", unsafe_allow_html=True)
                st.markdown(
                    f"<p style='font-size:20px;'>トレーニングデータに対する正解率: {train_accuracy:.2%}</p>",
                    unsafe_allow_html=True
                )

                # 混同行列（Matplotlib）
                cm = confusion_matrix(y_train_split, train_predicted_classes)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, ax=ax_cm)
                ax_cm.set_xticklabels(label_encoder.classes_)
                ax_cm.set_yticklabels(label_encoder.classes_)
                ax_cm.set_xlabel('予測ラベル')
                ax_cm.set_ylabel('実際のラベル')
                ax_cm.set_title(f'混同行列（{task["name"]} - トレーニングデータ）')
                st.pyplot(fig_cm)

                # テストデータでの予測
                predictions = model.predict(X_test)
                predicted_classes = np.argmax(predictions, axis=1)
                predicted_labels = label_encoder.inverse_transform(predicted_classes)

                test_df[f'予測_{target}'] = predicted_labels
                prediction_results[target] = predicted_labels

                # 新しい地点に対する予測結果
                st.markdown("## 新しい地点に対する予測結果", unsafe_allow_html=True)

                # Plotly 3D 散布図の作成（軸タイトルや凡例に Meiryo を指定）
                st.markdown(f"#### 予測結果の3D 地層プロット ({target})")
                marker_size_pred = st.sidebar.slider(
                    f"{target} 予測結果マーカーの大きさ", 1, 20, 10, key=f"{task['name']}_pred"
                )
                fig_3d_pred = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color=f'予測_{target}',
                    title=f"3D 地層プロット（予測: {target}）"
                )
                fig_3d_pred.update_layout(
                    title={
                        "text": f"3D 地層プロット（予測: {target}）",
                        "font": {
                            "family": "Meiryo",
                            "size": 18,
                            "color": "black"
                        }
                    },
                    scene=dict(
                        xaxis=dict(
                            title=dict(
                                text='経度',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                text='緯度',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        ),
                        zaxis=dict(
                            title=dict(
                                text='標高',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        )
                    ),
                    legend=dict(
                        font=dict(
                            family='Meiryo',
                            size=12,
                            color='black'
                        )
                    )
                )
                st.plotly_chart(fig_3d_pred, use_container_width=True)

            # タスク1,2 の予測結果がそろっていれば統合結果の表示
            if all([f'予測_{task["target"]}' in test_df.columns for task in tasks]):
                st.markdown("### [統合結果]")
                st.markdown("<h3 class='subheader'>予測結果の統合</h3>", unsafe_allow_html=True)

                def combine_predictions(row):
                    depositional_age = row['予測_地層区分（堆積年代）']
                    soil_info = row['予測_地層区分（土質情報）']
                    if depositional_age == 'G':
                        return 'G'
                    else:
                        return f"{depositional_age}{soil_info}"

                test_df['予測された地層区分'] = test_df.apply(combine_predictions, axis=1)

                st.markdown("#### 予測された地層区分の3D 地層プロット")
                marker_size_new = st.sidebar.slider(
                    "予測された地層区分マーカーの大きさ", 1, 20, 10, key="new_combined"
                )
                fig_3d_new_combined = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color='予測された地層区分',
                    title="3D 地層プロット（予測された地層区分）"
                )
                fig_3d_new_combined.update_layout(
                    title={
                        "text": "3D 地層プロット（予測された地層区分）",
                        "font": {
                            "family": "Meiryo",
                            "size": 18,
                            "color": "black"
                        }
                    },
                    scene=dict(
                        xaxis=dict(
                            title=dict(
                                text='経度',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        ),
                        yaxis=dict(
                            title=dict(
                                text='緯度',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        ),
                        zaxis=dict(
                            title=dict(
                                text='標高',
                                font=dict(
                                    family='Meiryo',
                                    size=12,
                                    color='black'
                                )
                            )
                        )
                    ),
                    legend=dict(
                        font=dict(
                            family='Meiryo',
                            size=12,
                            color='black'
                        )
                    )
                )
                st.plotly_chart(fig_3d_new_combined, use_container_width=True)

                st.markdown("<h3 class='subheader'>予測結果</h3>", unsafe_allow_html=True)
                display_columns = [f'予測_{task["target"]}' for task in tasks] + ['予測された地層区分']
                st.dataframe(test_df[display_columns])

                download_df = test_df[display_columns].copy()
                csv = download_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="予測結果をダウンロード",
                    data=csv,
                    file_name='prediction_results_combined.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.markdown("<p class='text'>トレーニング用およびテスト用のExcelファイルをアップロードしてください。</p>", unsafe_allow_html=True)

# Streamlitアプリの実行
if __name__ == "__main__":
    main()
