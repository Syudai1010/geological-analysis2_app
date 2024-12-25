# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:30:04 2024

@author: 鈴木脩大
"""

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

# 日本語フォント設定
plt.rcParams["font.family"] = "Meiryo"

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

            # タスク2の説明変数を設定
            if selected_task2_option == 'N値, 深度, 色調':
                task2_explanatory = ['N値', '深度', '色調']
            elif selected_task2_option == 'N値, 深度, R, G, B':
                task2_explanatory = ['N値', '深度', 'R', 'G', 'B']
            else:
                task2_explanatory = []

            # タスク2の設定
            task2 = {
                'name': '堆積年代による地層区分の予測',
                'explanatory': task2_explanatory,
                'target': '地層区分（堆積年代）'  # 目的変数
            }

            # タスクのリスト
            tasks = [task1, task2]

            prediction_results = {}

            for task in tasks:
                # タスク2の説明変数が選択されていない場合はスキップ
                if task['name'] == '堆積年代による地層区分の予測' and not task['explanatory']:
                    st.warning(f"タスク '{task['name']}' に使用する説明変数が選択されていません。")
                    continue

                st.markdown(f"<h3 class='subheader'>{task['name']}</h3>", unsafe_allow_html=True)

                explanatory = task['explanatory']
                target = task['target']

                # チェック: 説明変数と目的変数がデータに存在するか
                missing_cols = [col for col in explanatory + [target] if col not in train_df.columns]
                if missing_cols:
                    st.error(f"タスク '{task['name']}' に必要な列がデータに存在しません: {missing_cols}")
                    continue

                # 説明変数の種類を判定
                # ここでは、'土質区分'は離散値、その他は連続値と仮定
                if task['name'] == '土質情報による地層区分の予測':
                    explanatory_discrete = explanatory
                    explanatory_continuous = []
                else:
                    explanatory_continuous = explanatory
                    explanatory_discrete = []

                # 前処理
                ml_df = train_df.copy()
                X_continuous = ml_df[explanatory_continuous] if explanatory_continuous else pd.DataFrame()
                X_discrete = ml_df[explanatory_discrete] if explanatory_discrete else pd.DataFrame()
                y = ml_df[target]

                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                num_classes = len(label_encoder.classes_)

                transformers = []
                if not X_continuous.empty:
                    transformers.append(('minmax', MinMaxScaler(), explanatory_continuous))
                if not X_discrete.empty:
                    transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore'), explanatory_discrete))

                preprocessor = ColumnTransformer(transformers=transformers)
                X_train = preprocessor.fit_transform(ml_df)
                y_train = y_encoded

                # テストデータの前処理
                X_test = preprocessor.transform(test_df)
                y_test = label_encoder.transform(test_df[target])

                input_dim = X_train.shape[1]

                # モデル構築
                model = Sequential()
                model.add(Dense(64, input_dim=input_dim, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))

                optimizer = Adam(learning_rate=0.01)
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                # 早期停止の設定
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

                # データ分割（トレーニングとバリデーション）
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                # モデルのトレーニング
                with st.spinner(f"{task['name']} をトレーニング中..."):
                    history = model.fit(
                        X_train_split, y_train_split,
                        epochs=500,
                        validation_data=(X_val_split, y_val_split),
                        callbacks=[early_stop],
                        verbose=0
                    )
                st.success(f"{task['name']} のトレーニングが完了しました！")

                # トレーニングの進行状況をプロット
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

                # テストデータでの予測
                predictions = model.predict(X_test)
                predicted_classes = np.argmax(predictions, axis=1)

                accuracy = np.mean(predicted_classes == y_test)
                st.markdown(f"**正解率:** {accuracy:.2%}")

                # 予測結果のデコード
                predicted_labels = label_encoder.inverse_transform(predicted_classes)

                # テストデータに予測結果を追加
                test_df[f'予測_{target}'] = predicted_labels
                prediction_results[target] = predicted_labels

                # 3D プロット - 実際の地層区分
                st.markdown(f"#### 実際の地層区分の3D 地層プロット ({target})")
                # '地層区分'列の 'Alt' を 'G' に置換
                if '地層区分' in test_df.columns:
                    test_df['地層区分'] = test_df['地層区分'].replace({'Alt': 'G'})
                else:
                    st.warning("実際の地層区分の列 '地層区分' がデータに存在しません。")

                # 共通のラベルを取得
                actual_labels = test_df[target].unique()
                predicted_labels_unique = np.unique(predicted_labels)  # 修正箇所
                all_labels = np.unique(np.concatenate((actual_labels, predicted_labels_unique)))

                # 色マッピングの作成
                # Plotlyの定義済み色から必要な数だけ選択
                available_colors = px.colors.qualitative.Plotly
                if len(all_labels) > len(available_colors):
                    # 色が不足する場合は拡張
                    available_colors = px.colors.qualitative.Alphabet + available_colors

                color_discrete_map = {label: available_colors[i % len(available_colors)] for i, label in enumerate(all_labels)}

                marker_size_actual = st.sidebar.slider(f"{target} 実際の地層区分マーカーの大きさ", 1, 20, 10, key=f"{task['name']}_actual")
                fig_3d_actual = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color=target,
                    category_orders={target: all_labels},
                    color_discrete_map=color_discrete_map,
                    size_max=marker_size_actual,
                    title=f"3D 地層プロット（実際の地層区分: {target}）"
                )
                fig_3d_actual.update_layout(scene=dict(
                    xaxis=dict(
                        title='経度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        title='緯度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        title='標高',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_actual, use_container_width=True)

                # 3D プロット - 予測結果
                st.markdown(f"#### 予測結果の3D 地層プロット ({target})")
                marker_size_pred = st.sidebar.slider(f"{target} 予測結果マーカーの大きさ", 1, 20, 10, key=f"{task['name']}_pred")
                fig_3d_pred = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color=f'予測_{target}',
                    category_orders={f'予測_{target}': all_labels},
                    color_discrete_map=color_discrete_map,
                    size_max=marker_size_pred,
                    title=f"3D 地層プロット（予測: {target}）"
                )
                fig_3d_pred.update_layout(scene=dict(
                    xaxis=dict(
                        title='経度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        title='緯度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        title='標高',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_pred, use_container_width=True)

            # 結果の統合はタスク処理の外で行う
            if all([f'予測_{task["target"]}' in test_df.columns for task in tasks]):
                st.markdown("<h3 class='subheader'>予測結果の統合</h3>", unsafe_allow_html=True)
                # 新たな地層区分を作成（名称を '予測された地層区分' に変更）
                def combine_predictions(row):
                    depositional_age = row['予測_地層区分（堆積年代）']
                    soil_info = row['予測_地層区分（土質情報）']
                    if depositional_age == 'G':
                        return 'G'
                    else:
                        return f"{depositional_age}{soil_info}"

                test_df['予測された地層区分'] = test_df.apply(combine_predictions, axis=1)

                # 新たな地層区分の正解率を計算
                # '地層区分'が実際の正解値として存在することを前提とします
                if '地層区分' in test_df.columns:
                    # '地層区分'列の 'Alt' を 'G' に置換（再確認）
                    test_df['地層区分'] = test_df['地層区分'].replace({'Alt': 'G'})
                    combined_accuracy = np.mean(test_df['予測された地層区分'] == test_df['地層区分'])
                    st.markdown(f"**予測された地層区分の正解率:** {combined_accuracy:.2%}")
                else:
                    st.warning("実際の地層区分の列 '地層区分' がデータに存在しません。正解率を計算できません。")

                # 新たな地層区分の3Dプロットは削除

                # 新たな地層区分の3Dプロットを '予測された地層区分' に変更
                st.markdown("#### 予測された地層区分の3D 地層プロット")
                marker_size_new = st.sidebar.slider("予測された地層区分マーカーの大きさ", 1, 20, 10, key="new_combined")
                
                # 共通のラベルを取得
                predicted_labels_combined = test_df['予測された地層区分'].unique()
                actual_labels_combined = test_df['地層区分'].unique()
                all_labels_combined = np.unique(np.concatenate((predicted_labels_combined, actual_labels_combined)))

                # 色マッピングの作成
                available_colors_combined = px.colors.qualitative.Plotly
                if len(all_labels_combined) > len(available_colors_combined):
                    # 色が不足する場合は拡張
                    available_colors_combined = px.colors.qualitative.Alphabet + available_colors_combined

                color_discrete_map_combined = {label: available_colors_combined[i % len(available_colors_combined)] for i, label in enumerate(all_labels_combined)}

                fig_3d_new_combined = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color='予測された地層区分',
                    category_orders={'予測された地層区分': all_labels_combined},
                    color_discrete_map=color_discrete_map_combined,
                    size_max=marker_size_new,
                    title="3D 地層プロット（予測された地層区分）"
                )
                fig_3d_new_combined.update_layout(scene=dict(
                    xaxis=dict(
                        title='経度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        title='緯度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        title='標高',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_new_combined, use_container_width=True)

                # 実際の地層区分の3Dプロットを作成（予測との比較のため）
                st.markdown("#### 実際の地層区分の3D 地層プロット")
                marker_size_actual_comparison = st.sidebar.slider("実際の地層区分マーカーの大きさ", 1, 20, 10, key="actual_comparison")
                fig_3d_actual_comparison = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color='地層区分',
                    category_orders={'地層区分': all_labels_combined},
                    color_discrete_map=color_discrete_map_combined,
                    size_max=marker_size_actual_comparison,
                    title="3D 地層プロット（実際の地層区分）"
                )
                fig_3d_actual_comparison.update_layout(scene=dict(
                    xaxis=dict(
                        title='経度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        title='緯度',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        title='標高',
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_actual_comparison, use_container_width=True)

                # 結果の表示とダウンロード
                st.markdown("<h3 class='subheader'>予測結果</h3>", unsafe_allow_html=True)
                # 必要な列のみ表示
                display_columns = [f'予測_{task["target"]}' for task in tasks] + ['予測された地層区分']
                st.dataframe(test_df[display_columns])

                # ダウンロード用CSVの作成
                download_df = test_df[display_columns].copy()
                # 地層区分（土質情報）と堆積年代の順番に並べ替え
                download_df = download_df.rename(columns={
                    '予測_地層区分（土質情報）': '予測_地層区分（土質情報）',
                    '予測_地層区分（堆積年代）': '予測_地層区分（堆積年代）',
                    '予測された地層区分': '予測された地層区分'
                })
                csv = download_df.to_csv(index=False).encode('utf-8-sig')
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
