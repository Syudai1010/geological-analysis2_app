# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:32:55 2024

@author: 鈴木脩大
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Meiryo"

def main():
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

    st.markdown("<h1 class='title'>3D 地層プロット＆ニューラルネットワーク アプリ</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 class='subheader'>設定</h2>", unsafe_allow_html=True)

    st.sidebar.markdown("### トレーニングデータファイルをアップロード")
    uploaded_train_file = st.sidebar.file_uploader("トレーニング用 Excelファイルをアップロード", type=["xlsx", "xls"])

    st.sidebar.markdown("### テストデータファイルをアップロード")
    uploaded_test_file = st.sidebar.file_uploader("テスト用 Excelファイルをアップロード", type=["xlsx", "xls"])

    if uploaded_train_file is not None and uploaded_test_file is not None:
        try:
            # トレーニングデータ選択
            train_excel = pd.ExcelFile(uploaded_train_file)
            train_sheet_names = train_excel.sheet_names
            selected_train_sheet = st.sidebar.selectbox("トレーニングデータのシートを選択", train_sheet_names)
            train_df = pd.read_excel(uploaded_train_file, sheet_name=selected_train_sheet)

            # テストデータ選択
            test_excel = pd.ExcelFile(uploaded_test_file)
            test_sheet_names = test_excel.sheet_names
            selected_test_sheet = st.sidebar.selectbox("テストデータのシートを選択", test_sheet_names)
            test_df = pd.read_excel(uploaded_test_file, sheet_name=selected_test_sheet)

            columns = train_df.columns.tolist()

            st.sidebar.markdown("<h3 class='subheader'>機械学習の設定</h3>", unsafe_allow_html=True)

            latitude = st.sidebar.selectbox("緯度を選択", columns)
            longitude = st.sidebar.selectbox("経度を選択", columns)
            elevation = st.sidebar.selectbox("標高を選択", columns)

            explanatory_continuous = st.sidebar.multiselect("説明変数（連続値）を選択", columns, default=[])
            explanatory_discrete = st.sidebar.multiselect("説明変数（離散値）を選択", columns, default=[])
            target_variable = st.sidebar.selectbox("目的変数を選択", columns)

            # 以降の処理は、分割なしでtrain_df, test_dfを使用する
            explanatory_continuous = [col for col in explanatory_continuous if col != target_variable]
            explanatory_discrete = [col for col in explanatory_discrete if col != target_variable]

            if explanatory_continuous or explanatory_discrete:
                ml_df = train_df.copy()
                X_continuous = ml_df[explanatory_continuous] if explanatory_continuous else pd.DataFrame()
                X_discrete = ml_df[explanatory_discrete] if explanatory_discrete else pd.DataFrame()
                y = ml_df[target_variable]

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

                X_test = preprocessor.transform(test_df)
                y_test = label_encoder.transform(test_df[target_variable])

                input_dim = X_train.shape[1]

                model = Sequential()
                model.add(Dense(64, input_dim=input_dim, activation='relu'))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(num_classes, activation='softmax'))

                optimizer = Adam(learning_rate=0.01)
                model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                epochs = 500
                history = model.fit(X_train, y_train, epochs=epochs, verbose=0)

                st.markdown("<h3 class='subheader'>学習の進行状況</h3>", unsafe_allow_html=True)
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].plot(history.history['loss'], label='損失')
                ax[0].set_title('損失の推移')
                ax[0].legend()
                ax[1].plot(history.history['accuracy'], label='精度')
                ax[1].set_title('精度の推移')
                ax[1].legend()
                st.pyplot(fig)

                predictions = model.predict(X_test)
                predicted_classes = np.argmax(predictions, axis=1)

                accuracy = np.mean(predicted_classes == y_test)
                st.markdown(f"<h3 class='subheader'>正解率: {accuracy:.2%}</h3>", unsafe_allow_html=True)

                # ラベル名を統一してマーカーの色を一致
                all_labels = np.union1d(label_encoder.inverse_transform(np.arange(num_classes)),
                                        test_df[target_variable].unique())

                # 目的変数の3Dプロット（テストデータ）
                st.markdown("<h3 class='subheader'>目的変数の3D 地層プロット（テストデータ）</h3>", unsafe_allow_html=True)
                marker_size = st.sidebar.slider("目的変数マーカーの大きさ", 1, 20, 10)
                fig_3d_target = px.scatter_3d(test_df, x=longitude, y=latitude, z=elevation,
                                              color=target_variable, category_orders={target_variable: all_labels},
                                              size_max=marker_size, title="3D 地層プロット（目的変数）")
                fig_3d_target.update_layout(scene=dict(
                    xaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_target, use_container_width=True)

                # 予測値の3Dプロット
                st.markdown("<h3 class='subheader'>予測値の3D 地層プロット</h3>", unsafe_allow_html=True)
                plot_df = test_df.copy()
                plot_df['予測値'] = label_encoder.inverse_transform(predicted_classes)
                marker_size_pred = st.sidebar.slider("予測値マーカーの大きさ", 1, 20, 10)
                fig_3d_pred = px.scatter_3d(plot_df, x=longitude, y=latitude, z=elevation,
                                            color='予測値', category_orders={'予測値': all_labels},
                                            size_max=marker_size_pred, title="3D 地層プロット（予測値）")
                fig_3d_pred.update_layout(scene=dict(
                    xaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    yaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    ),
                    zaxis=dict(
                        titlefont=dict(color='black'),
                        tickfont=dict(color='black')
                    )
                ))
                st.plotly_chart(fig_3d_pred, use_container_width=True)

                result_df = test_df.copy()
                result_df['予測値'] = label_encoder.inverse_transform(predicted_classes)
                result_df['予測確率'] = np.max(predictions, axis=1)
                st.markdown("<h3 class='subheader'>予測結果</h3>", unsafe_allow_html=True)
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="予測結果をダウンロード", data=csv, file_name='prediction_results.csv', mime='text/csv')

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.markdown("<p class='text'>トレーニング用およびテスト用のExcelファイルをアップロードしてください。</p>", unsafe_allow_html=True)

# Streamlitアプリの実行
if __name__ == "__main__":
    main()
