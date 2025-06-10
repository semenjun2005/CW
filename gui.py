import streamlit as st
import pandas as pd
from joblib import load
from pipeline_class import BloodPressureCleaner, FeatureGenerator

features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

model_pipe = load('model/model.pkl')
st.title("Оценка риска заболевания")

user_data = {}
for f in features:
    user_data[f] = st.number_input(f, step=1.0 if f in ['age', 'height', 'weight', 'ap_hi', 'ap_lo'] else 1)

if st.button("Рассчитать риск"):
    user_df = pd.DataFrame([user_data])
    proba = model_pipe.predict_proba(user_df)[0, 1]
    st.write(f"Вероятность наличия заболевания: {proba:.2%}")
    if proba >= 0.5:
        st.warning("Внимание! Риск выше 50% — рекомендуется обратиться к врачу.")
    else:
        st.success("Риск низкий, но всегда следите за здоровьем.")

# Для использования написать в терминале:
# streamlit run gui.py
# Для выхода нажмите CNTRL+C
