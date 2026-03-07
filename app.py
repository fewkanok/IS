import streamlit as st
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# ==========================================
# 1. SETTINGS & CONFIG
# ==========================================
st.set_page_config(page_title="Project IS 2568 - AI Analysis", layout="wide")

MUSHROOM_MAP = {
    'odor': {'a': 'Almond (อัลมอนด์)', 'l': 'Anise (โป๊ยกั๊ก)', 'p': 'Pungent (กลิ่นฉุน)', 'n': 'None (ไม่มีกลิ่น)', 'f': 'Foul (กลิ่นเหม็น)', 'm': 'Musty (กลิ่นอับ)', 's': 'Spicy (กลิ่นเครื่องเทศ)', 'y': 'Fishy (กลิ่นคาว)', 'c': 'Creosote (กลิ่นน้ำมันดิน)'},
    'gill-color': {'k': 'Black (ดำ)', 'n': 'Brown (น้ำตาล)', 'b': 'Buff (น้ำตาลอ่อน)', 'h': 'Chocolate (ช็อกโกแลต)', 'g': 'Gray (เทา)', 'r': 'Green (เขียว)', 'o': 'Orange (ส้ม)', 'p': 'Pink (ชมพู)', 'u': 'Purple (ม่วง)', 'e': 'Red (แดง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'spore-print-color': {'k': 'Black (ดำ)', 'n': 'Brown (น้ำตาล)', 'b': 'Buff (น้ำตาลอ่อน)', 'h': 'Chocolate (ช็อกโกแลต)', 'r': 'Green (เขียว)', 'o': 'Orange (ส้ม)', 'u': 'Purple (ม่วง)', 'w': 'White (ขาว)', 'y': 'Yellow (เหลือง)'},
    'population': {'a': 'Abundant (หนาแน่นมาก)', 'c': 'Clustered (รวมกลุ่ม)', 'n': 'Numerous (จำนวนมาก)', 's': 'Scattered (กระจาย)', 'v': 'Several (ปานกลาง)', 'y': 'Solitary (ขึ้นโดดเดี่ยว)'}
}

@st.cache_resource
def load_all_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    age_path = os.path.join(base_dir, "models", "age_model_best.h5")
    mush_path = os.path.join(base_dir, "models", "mushroom_model.pkl")
    mush_enc_path = os.path.join(base_dir, "models", "mushroom_encoders.pkl")
    mush_target_path = os.path.join(base_dir, "models", "mushroom_target_encoder.pkl")

    if not all(os.path.exists(p) for p in [age_path, mush_path, mush_enc_path, mush_target_path]):
        return None

    nn_model = tf.keras.models.load_model(age_path, compile=False)
    ml_model = joblib.load(mush_path)
    ml_encoders = joblib.load(mush_enc_path)
    ml_target = joblib.load(mush_target_path)
    return nn_model, ml_model, ml_encoders, ml_target

models = load_all_models()
if models:
    age_model, mush_model, mush_encoders, mush_target = models
    model_ready = True
else:
    model_ready = False

# ==========================================
# 2. SIDEBAR NAVIGATION (เหลือ 4 หน้าตามข้อกำหนด 4.a และ 4.b)
# ==========================================
st.sidebar.title("📌 IS 2568 Navigation")
page = st.sidebar.radio("เลือกหัวข้อ:", 
    ["📘 อธิบาย Neural Network", "📸 ทดสอบทายอายุ (NN)", 
     "📙 อธิบาย Machine Learning", "🍄 ทดสอบจำแนกเห็ด (ML)"])

# ==========================================
# 📘 หน้าอธิบาย NN (เพิ่มเนื้อหาเชิงลึก)
# ==========================================
if page == "📘 อธิบาย Neural Network":
    st.title("📘 รายละเอียดการพัฒนาโมเดล Neural Network")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. ข้อมูลที่ใช้พัฒนา (Dataset)")
        st.write("- **ที่มา:** https://www.kaggle.com/datasets/jangedoo/utkface-new/data")
        st.write("- **ลักษณะข้อมูล:** รูปภาพใบหน้ามนุษย์แบบ Unstructured Data จำนวนกว่า 20,000 ภาพ")
        st.write("- **ฟีเจอร์หลัก (Feature Extract):** พิกเซลของรูปภาพที่ระบุอายุ (Age) ซึ่งมีช่วงตั้งแต่อายุ 0 ถึง 116 ปี ")
        
        st.subheader("2. ขั้นตอนการเตรียมข้อมูล (Preprocessing)")
        st.write("- **ปัญหาความไม่สมบูรณ์:** ข้อมูลช่วงอายุเด็กและคนชรามีปริมาณน้อยกว่าช่วงวัยทำงานอย่างมาก (Data Imbalance)")
        st.write("- **การทำความสะอาด:** กรองรูปภาพที่เบลอหรือมุมกล้องผิดเพี้ยน และเลือกช่วงอายุ 10-60 ปีเพื่อให้โมเดลเรียนรู้ได้แม่นยำขึ้น")
        st.write("- **Data Augmentation:** การทำ Random Rotation, Flip และ Zoom เพื่อเพิ่มความหลากหลายให้กับข้อมูลฝึกสอน")

    with col2:
        st.subheader("3. ทฤษฎีและโครงสร้างโมเดล")
        st.write("- **สถาปัตยกรรม:** MobileNetV2 ซึ่งถูกออกแบบมาให้ทำงานได้รวดเร็วบนอุปกรณ์พกพา")
        
        st.write("- **Transfer Learning:** ใช้ Weights จาก ImageNet เพื่อลดระยะเวลาในการฝึกสอนและเพิ่มประสิทธิภาพในการดึงฟีเจอร์ใบหน้า")
        st.write("- **Regression Head:** ปรับแต่งเลเยอร์สุดท้ายด้วย Dense 256 unit (Activation: Swish) และ Output 1 unit เพื่อทายค่าต่อเนื่อง (อายุ)")


elif page == "📸 ทดสอบทายอายุ (NN)":
    st.title("📸 ทดสอบระบบทำนายอายุจากใบหน้า")
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and model_ready:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, use_column_width=True)
            if st.button("เริ่มประมวลผล", use_container_width=True):
                prep = np.array(img.resize((128, 128))) / 255.0
                pred = age_model.predict(np.expand_dims(prep, axis=0))
                st.success(f"🎯 อายุที่คาดการณ์คือ: {int(pred[0][0])} ปี")


# ==========================================
# 📙 หน้าอธิบาย ML (เพิ่มเนื้อหาเชิงลึก)
# ==========================================
elif page == "📙 อธิบาย Machine Learning":
    st.title("📙 รายละเอียดการพัฒนาโมเดล Machine Learning")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. ข้อมูลที่ใช้พัฒนา (Dataset)")
        st.write("- **ที่มา:** https://www.kaggle.com/datasets/uciml/mushroom-classification")
        st.write("- **ลักษณะข้อมูล:** ข้อมูลแบบตาราง (Structured Data) จำนวน 8,124 รายการ ")
        st.write("- **คุณลักษณะ (Features):** 22 รายการ เช่น กลิ่น (Odor), สีสปอร์ (Spore-print-color), และรูปร่างของหมวกเห็ด ")

        st.subheader("2. การเตรียมข้อมูล (Preprocessing)")
        st.write("- **Missing Values:** ตรวจสอบข้อมูลที่สูญหายในคอลัมน์ 'stalk-root'")
        st.write("- **Label Encoding:** แปลงข้อมูลจากตัวอักษรเป็นตัวเลขเพื่อให้โมเดลสามารถคำนวณได้")
        st.write("- **Data Splitting:** แบ่งข้อมูลออกเป็น 80% สำหรับการฝึกสอน และ 20% สำหรับการทดสอบ")

    with col2:
        st.subheader("3. อัลกอริทึม Ensemble (Random Forest)")
        st.write("- **โมเดล:** ใช้ Random Forest Classifier ที่ประกอบด้วย Decision Trees 100 ต้น")
        st.write("- **หลักการทำงาน:** ใช้เทคนิค Bootstrap Aggregating (Bagging) เพื่อลดความเสี่ยงจากการ Overfitting")
        st.write("- **ประสิทธิภาพ:** ความแม่นยำสูงถึง 100% บนชุดข้อมูลทดสอบ เนื่องจากข้อมูลมีความสัมพันธ์ระหว่างฟีเจอร์ที่ชัดเจน")
    

# ==========================================
# 🍄 หน้าทดสอบ ML (ข้อกำหนด 4.b, 18)
# ==========================================
elif page == "🍄 ทดสอบจำแนกเห็ด (ML)":
    st.title("🍄 ทดสอบระบบจำแนกเห็ดพิษ")
    if model_ready:
        col1, col2 = st.columns(2)
        with col1:
            o_c = st.selectbox("กลิ่น (Odor):", list(MUSHROOM_MAP['odor'].values()))
            g_c = st.selectbox("สีครีบเห็ด (Gill Color):", list(MUSHROOM_MAP['gill-color'].values()))
        with col2:
            s_c = st.selectbox("สีสปอร์ (Spore Color):", list(MUSHROOM_MAP['spore-print-color'].values()))
            p_c = st.selectbox("ลักษณะประชากร (Population):", list(MUSHROOM_MAP['population'].values()))

        if st.button("วิเคราะห์ผล", use_container_width=True):
            def get_k(v, m): return next(k for k, val in m.items() if v == val)
            vec = np.zeros((1, 22))
            vec[0, 4] = mush_encoders['odor'].transform([get_k(o_c, MUSHROOM_MAP['odor'])])[0]
            vec[0, 9] = mush_encoders['gill-color'].transform([get_k(g_c, MUSHROOM_MAP['gill-color'])])[0]
            vec[0, 20] = mush_encoders['spore-print-color'].transform([get_k(s_c, MUSHROOM_MAP['spore-print-color'])])[0]
            vec[0, 21] = mush_encoders['population'].transform([get_k(p_c, MUSHROOM_MAP['population'])])[0]
            
            res = mush_model.predict(vec)
            if res[0] == 0: st.success("✅ เห็ดนี้กินได้ (Edible)")
            else: st.error("⚠️ เห็ดนี้มีพิษ (Poisonous)")