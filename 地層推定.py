# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 18:16:04 2024

@author: 鈴木脩大
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import random

# 日本語フォント設定（Matplotlib用）
plt.rcParams["font.family"] = "Meiryo"

def plot_stratigraphic_cross_section_svm(df, lon_col, lat_col, elev_col, strat_col, svm_model, color_map):
    """
    SVMモデルを用いて予測された地層区分を基に、3D地層縦断図を作成します。
    各地層区分ごとにサーフェスを描画し、ボーリング柱も色分けしてプロットします。
    """
    strata = df[strat_col].unique()

    # 緯度と経度のスケーリング（Min-Maxスケーリング）
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[[lon_col, lat_col]] = scaler.fit_transform(df[[lon_col, lat_col]])

    # 緯度と経度の分散を確認
    lon_std = df_scaled[lon_col].std()
    lat_std = df_scaled[lat_col].std()

    if lon_std < 1e-3 or lat_std < 1e-3:
        st.warning("緯度と経度の分散が非常に小さいため、3Dプロットが正常に作成できません。データを確認してください。")
        return

    # ジャギング（微小なノイズの追加）を適用
    jitter_amount = 1e-5
    df_scaled[f"{lon_col}_jittered"] = df_scaled[lon_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_scaled))
    df_scaled[f"{lat_col}_jittered"] = df_scaled[lat_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_scaled))

    # 空間グリッドの作成
    lon_min, lon_max = df_scaled[f"{lon_col}_jittered"].min(), df_scaled[f"{lon_col}_jittered"].max()
    lat_min, lat_max = df_scaled[f"{lat_col}_jittered"].min(), df_scaled[f"{lat_col}_jittered"].max()
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(lon_min, lon_max, 50),
        np.linspace(lat_min, lat_max, 50)
    )

    # 3D散布図の作成
    fig = go.Figure()

    for idx, strat in enumerate(strata):
        # 該当地層のデータを取得
        strat_points = df_scaled[df_scaled[strat_col] == strat]

        if strat_points.empty:
            continue

        # データポイント数を確認
        num_points = len(strat_points)
        if num_points < 4:
            st.warning(f"地層区分 '{strat}' に属するデータポイントが不足しています（必要:4, 実際:{num_points}）。ポイントとしてプロットします。")
            # ポイントを散布図としてプロット
            fig.add_trace(go.Scatter3d(
                x=strat_points[f"{lon_col}_jittered"],
                y=strat_points[f"{lat_col}_jittered"],
                z=strat_points[elev_col],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_map.get(strat, 'gray'),
                    symbol='circle'
                ),
                name=strat
            ))
            continue

        # 標高データの補間
        try:
            elev_grid = griddata(
                (strat_points[f"{lon_col}_jittered"], strat_points[f"{lat_col}_jittered"]),
                strat_points[elev_col],
                (lon_grid, lat_grid),
                method='linear'
            )
        except Exception as e:
            st.error(f"地層区分 '{strat}' に対する標高データの補間中にエラーが発生しました: {e}")
            continue

        # 補間結果が全てNaNの場合はスキップ
        if np.all(np.isnan(elev_grid)):
            st.warning(f"地層区分 '{strat}' に対する補間結果が全てNaNのため、プロットをスキップします。")
            continue

        # NaNを補間方法に応じて適切に処理（ここでは線形補間後のNaNを平均値で補完）
        elev_grid = np.where(np.isnan(elev_grid), strat_points[elev_col].mean(), elev_grid)

        # Surfaceの追加（透明度を1に設定）
        fig.add_trace(go.Surface(
            x=lon_grid,
            y=lat_grid,
            z=elev_grid,
            colorscale=[[0, color_map[strat]], [1, color_map[strat]]],
            showscale=False,
            name=strat,
            opacity=1  # 透明度を1に設定
        ))

    # ボーリング柱の描画（地層区分ごとに色を割り当て）
    fig.add_trace(go.Scatter3d(
        x=df_scaled[f"{lon_col}_jittered"],
        y=df_scaled[f"{lat_col}_jittered"],
        z=df_scaled[elev_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df_scaled[strat_col].map(color_map),
            symbol='circle',
            line=dict(color='black', width=0.5)
        ),
        name='Borehole',
        hovertemplate=
            f'地層区分: %{{marker.color}}<br>'
            f'経度: %{{x}}<br>'
            f'緯度: %{{y}}<br>'
            f'標高: %{{z}}'
    ))

    # 凡例のカスタマイズ（地層区分ごとの色を表示）
    for strat in strata:
        if len(df_scaled[df_scaled[strat_col] == strat]) < 4:
            # ポイントとしてプロットされた場合は既に凡例に表示されている
            continue
        fig.add_trace(go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode='markers',
            marker=dict(
                size=10,
                color=color_map[strat]
            ),
            name=strat
        ))

    # レイアウト設定
    fig.update_layout(
        scene=dict(
            xaxis_title='経度',
            yaxis_title='緯度',
            zaxis_title='標高'
        ),
        title='SVMによる3D地層縦断図',
        width=800,
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

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

            # タスク1の設定（堆積年代による地層区分の予測を先に）
            task1 = {
                'name': '堆積年代による地層区分の予測',
                'explanatory': ['N値', '深度', '色調', 'R', 'G', 'B'],  # 説明変数は選択に基づき動的に設定
                'target': '地層区分（堆積年代）'  # 目的変数
            }

            # タスク2の設定（土質情報による地層区分の予測）
            task2 = {
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

            # タスク1の説明変数を設定（タスク2から分離）
            if selected_task2_option == 'N値, 深度, 色調':
                task1_explanatory = ['N値', '深度', '色調']
            elif selected_task2_option == 'N値, 深度, R, G, B':
                task1_explanatory = ['N値', '深度', 'R', 'G', 'B']
            else:
                task1_explanatory = []

            # 更新されたタスク1の説明変数
            task1['explanatory'] = task1_explanatory

            # タスクのリスト（堆積年代を先に、土質情報を後に）
            tasks = [task1, task2]

            prediction_results = {}
            svm_models = {}  # 各タスクごとのSVMモデルを保持する辞書

            # 一時的なリストに予測された地層区分を格納
            # これにより、すべてのタスクが完了した後に色マップを再構築可能
            combined_predictions = []

            for task in tasks:
                # タスク1の説明変数が選択されていない場合はスキップ
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
                # '土質区分'は離散値、その他は連続値と仮定
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

                # SVMモデルの構築（One-vs-Rest）
                svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
                svm.fit(X_train, y_train)
                svm_models[target] = svm  # 各タスクごとにSVMモデルを保持

                # Neural Networkモデル構築
                model = Sequential()
                model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
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

                # テストデータでの予測（ニューラルネットワーク）
                predictions = model.predict(X_test)
                predicted_classes = np.argmax(predictions, axis=1)

                accuracy = np.mean(predicted_classes == y_test)
                st.markdown(f"**正解率:** {accuracy:.2%}")

                # 予測結果のデコード
                predicted_labels = label_encoder.inverse_transform(predicted_classes)

                # テストデータに予測結果を追加
                test_df[f'予測_{target}'] = predicted_labels
                prediction_results[target] = predicted_labels

                # 一時的なリストに予測された地層区分を追加
                combined_predictions.extend(predicted_labels)

                # 3D プロット - 実際の地層区分
                st.markdown(f"#### 実際の地層区分の3D 地層プロット ({target})")
                # '地層区分'列の 'Alt' を 'G' に置換
                if '地層区分' in test_df.columns:
                    test_df['地層区分'] = test_df['地層区分'].replace({'Alt': 'G'})
                else:
                    st.warning("実際の地層区分の列 '地層区分' がデータに存在しません。")

                # 共通のラベルを取得
                actual_labels = test_df[target].unique()
                predicted_labels_unique = np.unique(predicted_labels)
                all_labels = np.unique(np.concatenate((actual_labels, predicted_labels_unique)))

                # マーカーサイズを小さく設定
                marker_size_actual = st.sidebar.slider(f"{target} 実際の地層区分マーカーの大きさ", 1, 10, 5, key=f"{task['name']}_actual")
                fig_3d_actual = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color=target,
                    category_orders={target: all_labels},
                    color_discrete_map=None,  # 後で再定義
                    size_max=marker_size_actual,
                    title=f"3D 地層プロット（実際の地層区分: {target}）",
                    opacity=1  # 透明度を1に設定
                )

                # 凡例に対応する色マッピングを適用
                fig_3d_actual.update_traces(marker=dict(colorscale=None))
                fig_3d_actual.update_layout(coloraxis_colorbar=dict(title=target))

                st.plotly_chart(fig_3d_actual, use_container_width=True)

                # 3D プロット - 予測結果
                st.markdown(f"#### 予測結果の3D 地層プロット ({target})")
                marker_size_pred = st.sidebar.slider(f"{target} 予測結果マーカーの大きさ", 1, 10, 5, key=f"{task['name']}_pred")
                fig_3d_pred = px.scatter_3d(
                    test_df,
                    x=lon_col,
                    y=lat_col,
                    z=elev_col,
                    color=f'予測_{target}',
                    category_orders={f'予測_{target}': all_labels},
                    color_discrete_map=None,  # 後で再定義
                    size_max=marker_size_pred,
                    title=f"3D 地層プロット（予測: {target}）",
                    opacity=1  # 透明度を1に設定
                )

                # 凡例に対応する色マッピングを適用
                fig_3d_pred.update_traces(marker=dict(colorscale=None))
                fig_3d_pred.update_layout(coloraxis_colorbar=dict(title=f'予測_{target}'))

                st.plotly_chart(fig_3d_pred, use_container_width=True)

            # 結果の統合はタスク処理の外で行う
            # '予測された地層区分' を生成（堆積年代を先に、土質情報を後に）
            st.markdown("<h3 class='subheader'>予測結果の統合</h3>", unsafe_allow_html=True)
            def combine_predictions(row):
                # すべての予測結果を結合（堆積年代を先に、土質情報を後に）
                preds = []
                for task in tasks:
                    target = task['target']
                    preds.append(str(row.get(f'予測_{target}', '')))
                combined = ''.join(preds)
                return combined if combined else '未分類'

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

            # color_mapの再構築（予測された地層区分に基づく）
            unique_combined_strata = test_df['予測された地層区分'].unique()
            color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24  # 十分な色数を確保
            color_map_combined = {strat: color_palette[i % len(color_palette)] for i, strat in enumerate(unique_combined_strata)}

            # タスク処理で得たcolor_map_combinedを使用して再プロット
            # ここでは既にプロットされているため、必要に応じて修正

            # 3D 地層縦断図の描画（SVMによる分類）
            st.markdown("<h3 class='subheader'>3D 地層縦断図（SVMによる分類）</h3>", unsafe_allow_html=True)
            # '予測された地層区分'が存在することを確認
            if '予測された地層区分' in test_df.columns:
                # 使用するSVMモデルを選択（堆積年代のモデルを使用）
                if task1['target'] in svm_models:
                    # color_map_combinedを使用
                    plot_stratigraphic_cross_section_svm(
                        test_df,
                        lon_col,
                        lat_col,
                        elev_col,
                        '予測された地層区分',
                        svm_models[task1['target']],
                        color_map_combined
                    )
                else:
                    st.error("堆積年代による地層区分のSVMモデルが見つかりません。")
            else:
                st.warning("予測された地層区分の列 '予測された地層区分' がデータに存在しません。")

            # 凡例の追加（color_map_combinedを使用）
            # 既に plot_stratigraphic_cross_section_svm 内で凡例が追加されているため、ここでは不要

            # 結果の表示とダウンロード
            st.markdown("<h3 class='subheader'>予測結果</h3>", unsafe_allow_html=True)
            # 必要な列のみ表示
            display_columns = [f'予測_{task["target"]}' for task in tasks] + ['予測された地層区分']
            st.dataframe(test_df[display_columns])

            # ダウンロード用CSVの作成
            download_df = test_df[display_columns].copy()
            # 地層区分（土質情報）と堆積年代の順番に並べ替え（既に順序はタスクリストに基づく）
            # ここでは列名を保持するためリネームは不要ですが、必要に応じて調整
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





