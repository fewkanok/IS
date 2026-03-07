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
st.set_page_config(page_title="Project IS 2568 - AI Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #7B68EE;
        --success-color: #50C878;
        --danger-color: #FF6B6B;
        --warning-color: #FFB84D;
    }
    
    /* Header styling */
    h1 {
        color: #2C3E50;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #4A90E2;
        margin-bottom: 20px;
    }
    
    h2, h3 {
        color: #34495E;
        font-weight: 600;
    }
    
    /* Card-like sections */
    .css-1r6slb0 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #4A90E2;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #5D6D7E;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4A90E2;
        border-radius: 10px;
        padding: 20px;
        background: #F8F9FA;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4A90E2, transparent);
        margin: 30px 0;
    }
    
    /* Radio buttons in sidebar */
    .stRadio > label {
        color: white !important;
        font-weight: 600;
    }
    
    /* Navigation items */
    .stRadio > div {
        background: rgba(255,255,255,0.1);
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
# 2. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown("# 📌 IS 2568 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("**เลือกหัวข้อ:**", 
    ["📘 อธิบาย Neural Network", "📸 ทดสอบทายอายุ (NN)", 
     "📙 อธิบาย Machine Learning", "🍄 ทดสอบจำแนกเห็ด (ML)"],
    index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px; color: white;'>
    <h4>🎓 Project IS 2568</h4>
    <p style='font-size: 0.9em; opacity: 0.8;'>AI Analysis & Classification</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 📘 หน้าอธิบาย NN
# ==========================================
if page == "📘 อธิบาย Neural Network":
    # Header with icon
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4A90E2; border: none;'>🧠 Neural Network Model</h1>
        <p style='font-size: 1.2em; color: #7F8C8D;'>Age Prediction from Facial Images</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics Dashboard with enhanced styling
    st.markdown("### 📊 Model Performance Metrics")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
            <h4 style='margin: 0; color: white;'>Model Accuracy</h4>
            <h1 style='margin: 10px 0; font-size: 3em; color: white; border: none;'>60.5%</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Test Set Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metrics2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);'>
            <h4 style='margin: 0; color: white;'>Mean Absolute Error</h4>
            <h1 style='margin: 10px 0; font-size: 3em; color: white; border: none;'>7.2</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Years (Average)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metrics3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);'>
            <h4 style='margin: 0; color: white;'>Architecture</h4>
            <h1 style='margin: 10px 0; font-size: 2.5em; color: white; border: none;'>MobileNetV2</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Transfer Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content with enhanced cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;'>
            <h3 style='color: #4A90E2; margin-top: 0;'>📊 1. ข้อมูลที่ใช้พัฒนา (Dataset)</h3>
        """, unsafe_allow_html=True)
        st.write("- **ที่มา:** [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new/data)")
        st.write("- **ลักษณะข้อมูล:** รูปภาพใบหน้ามนุษย์แบบ Unstructured Data จำนวนกว่า 20,000 ภาพ")
        st.write("- **ฟีเจอร์หลัก:** พิกเซลของรูปภาพที่ระบุอายุ (Age) 0 ถึง 116 ปี")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #4A90E2; margin-top: 0;'>⚙️ 2. ขั้นตอนการเตรียมข้อมูล</h3>
        """, unsafe_allow_html=True)
        st.write("- **Data Imbalance:** ข้อมูลช่วงอายุเด็กและคนชราน้อยกว่าวัยทำงาน จึงกรองเลือกช่วง 10-60 ปี")
        st.write("- **Data Augmentation:** เพิ่มความหลากหลายด้วย Random Rotation, Flip และ Zoom")
        st.info("💡 หมายเหตุ: การกรองข้อมูลช่วยให้ MAE ลดลงจากเดิมได้ถึง 15%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;'>
            <h3 style='color: #4A90E2; margin-top: 0;'>🔬 3. ทฤษฎีและโครงสร้างโมเดล</h3>
        """, unsafe_allow_html=True)
        st.write("- **สถาปัตยกรรม:** MobileNetV2 ซึ่งเป็น Lightweight CNN เหมาะสำหรับการรันบนเว็บ")
        st.write("- **Transfer Learning:** ใช้ Weights จาก ImageNet เพื่อช่วยในการสกัดฟีเจอร์ใบหน้า (Feature Extraction)")
        st.write("- **Regression Head:** ใช้ Dense 256 (Swish) และ Output 1 unit เพื่อทำนายค่าต่อเนื่อง")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #4A90E2; margin-top: 0;'>📈 4. สรุปผลการทดสอบ</h3>
        """, unsafe_allow_html=True)
        st.warning("⚠️ ผลลัพธ์ Accuracy อยู่ที่ 60.5% เนื่องจากปัจจัยเรื่องแสงและมุมกล้องของภาพใน Dataset ที่มีความหลากหลายสูง")
        st.markdown("</div>", unsafe_allow_html=True)


elif page == "📸 ทดสอบทายอายุ (NN)":
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #4A90E2; border: none;'>📸 Age Prediction System</h1>
        <p style='font-size: 1.2em; color: #7F8C8D;'>Upload a facial image to predict age</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not model_ready:
        st.error("⚠️ โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดล")
    else:
        # Upload section with enhanced styling
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white; margin-bottom: 30px;'>
            <h3 style='margin: 0; color: white;'>🖼️ อัปโหลดรูปภาพใบหน้า</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>รองรับไฟล์: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพ...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown("""
                <div style='background: white; padding: 20px; border-radius: 15px; 
                            box-shadow: 0 4px 20px rgba(0,0,0,0.15);'>
                """, unsafe_allow_html=True)
                
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, use_column_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 เริ่มประมวลผล", use_container_width=True):
                    with st.spinner('กำลังวิเคราะห์...'):
                        prep = np.array(img.resize((128, 128))) / 255.0
                        pred = age_model.predict(np.expand_dims(prep, axis=0))
                        predicted_age = int(pred[0][0])
                        
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #50C878 0%, #38A169 100%); 
                                    padding: 30px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 4px 20px rgba(80, 200, 120, 0.3); margin-top: 20px;'>
                            <h2 style='margin: 0; color: white; border: none;'>🎯 ผลการทำนาย</h2>
                            <h1 style='margin: 20px 0 10px 0; font-size: 4em; color: white; border: none;'>{predicted_age}</h1>
                            <h3 style='margin: 0; color: white; opacity: 0.9;'>ปี (Years Old)</h3>
                        </div>
                        """, unsafe_allow_html=True)


# ==========================================
# 📙 หน้าอธิบาย ML
# ==========================================
elif page == "📙 อธิบาย Machine Learning":
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #50C878; border: none;'>🤖 Machine Learning Model</h1>
        <p style='font-size: 1.2em; color: #7F8C8D;'>Mushroom Classification System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 📊 Metrics Dashboard
    st.markdown("### 📊 Model Performance Metrics")
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    
    with col_metrics1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #50C878 0%, #38A169 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(80, 200, 120, 0.3);'>
            <h4 style='margin: 0; color: white;'>Model Accuracy</h4>
            <h1 style='margin: 10px 0; font-size: 3em; color: white; border: none;'>100%</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Perfect Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metrics2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);'>
            <h4 style='margin: 0; color: white;'>F1-Score</h4>
            <h1 style='margin: 10px 0; font-size: 3em; color: white; border: none;'>1.00</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Precision & Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_metrics3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);'>
            <h4 style='margin: 0; color: white;'>Algorithm</h4>
            <h1 style='margin: 10px 0; font-size: 2.5em; color: white; border: none;'>Random Forest</h1>
            <p style='margin: 0; opacity: 0.9; font-size: 0.9em;'>Ensemble Method</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;'>
            <h3 style='color: #50C878; margin-top: 0;'>📊 1. ข้อมูลที่ใช้พัฒนา (Dataset)</h3>
        """, unsafe_allow_html=True)
        st.write("- **ที่มา:** [Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)")
        st.write("- **ลักษณะข้อมูล:** Structured Data (ข้อมูลแบบตาราง) จำนวน 8,124 รายการ")
        st.write("- **คุณลักษณะ (Features):** 22 รายการ เช่น กลิ่น (Odor), สีสปอร์ (Spore-color)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #50C878; margin-top: 0;'>⚙️ 2. การเตรียมข้อมูล</h3>
        """, unsafe_allow_html=True)
        st.write("- **Encoding:** แปลงข้อมูลหมวดหมู่ (Categorical) เป็นตัวเลขด้วย LabelEncoder")
        st.write("- **Data Splitting:** แบ่งข้อมูล Test Set 20% เพื่อใช้ประเมินผลความแม่นยำ")
        st.write("- **Feature Selection:** เลือกใช้ฟีเจอร์หลักที่มีอิทธิพลสูงในการตัดสินใจ")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;'>
            <h3 style='color: #50C878; margin-top: 0;'>🌳 3. อัลกอริทึม Ensemble</h3>
        """, unsafe_allow_html=True)
        st.write("- **โมเดล:** ใช้ Random Forest Classifier (100 Decision Trees)")
        st.write("- **หลักการทำงาน:** ใช้เทคนิค Bagging เพื่อรวมผลลัพธ์จากหลายต้นไม้ ช่วยลดค่า Variance")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
            <h3 style='color: #50C878; margin-top: 0;'>📈 4. วิเคราะห์ผลการทดสอบ</h3>
        """, unsafe_allow_html=True)
        st.success("✅ ความแม่นยำ 100% (Perfect Classification)")
        st.write("- **เหตุผล:** ข้อมูลมีความสัมพันธ์ (Correlation) ระหว่างฟีเจอร์ 'กลิ่น' และ 'สีสปอร์' ที่ชัดเจนมาก ทำให้โมเดลแยกแยะเห็ดพิษและเห็ดกินได้ได้อย่างเด็ดขาด")
        st.markdown("</div>", unsafe_allow_html=True)
    

# ==========================================
# 🍄 หน้าทดสอบ ML
# ==========================================
elif page == "🍄 ทดสอบจำแนกเห็ด (ML)":
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #50C878; border: none;'>🍄 Mushroom Classification</h1>
        <p style='font-size: 1.2em; color: #7F8C8D;'>Identify Edible vs Poisonous Mushrooms</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not model_ready:
        st.error("⚠️ โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบไฟล์โมเดล")
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 30px;'>
            <h3 style='margin: 0; color: white;'>🔬 กรอกลักษณะของเห็ด</h3>
            <p style='margin: 10px 0 0 0; opacity: 0.9;'>เลือกคุณลักษณะต่างๆ เพื่อวิเคราะห์</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form with enhanced styling
        st.markdown("""
        <div style='background: white; padding: 30px; border-radius: 15px; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👃 กลิ่น (Odor)")
            o_c = st.selectbox("เลือกกลิ่น:", list(MUSHROOM_MAP['odor'].values()), label_visibility="collapsed", key="odor")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🌈 สีครีบเห็ด (Gill Color)")
            g_c = st.selectbox("เลือกสีครีบเห็ด:", list(MUSHROOM_MAP['gill-color'].values()), label_visibility="collapsed", key="gill")
            
        with col2:
            st.markdown("#### 🎨 สีสปอร์ (Spore Color)")
            s_c = st.selectbox("เลือกสีสปอร์:", list(MUSHROOM_MAP['spore-print-color'].values()), label_visibility="collapsed", key="spore")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 👥 ลักษณะประชากร (Population)")
            p_c = st.selectbox("เลือกลักษณะประชากร:", list(MUSHROOM_MAP['population'].values()), label_visibility="collapsed", key="pop")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 วิเคราะห์ผล", use_container_width=True):
                with st.spinner('กำลังวิเคราะห์...'):
                    def get_k(v, m): return next(k for k, val in m.items() if v == val)
                    vec = np.zeros((1, 22))
                    vec[0, 4] = mush_encoders['odor'].transform([get_k(o_c, MUSHROOM_MAP['odor'])])[0]
                    vec[0, 9] = mush_encoders['gill-color'].transform([get_k(g_c, MUSHROOM_MAP['gill-color'])])[0]
                    vec[0, 20] = mush_encoders['spore-print-color'].transform([get_k(s_c, MUSHROOM_MAP['spore-print-color'])])[0]
                    vec[0, 21] = mush_encoders['population'].transform([get_k(p_c, MUSHROOM_MAP['population'])])[0]
                    
                    res = mush_model.predict(vec)
                    
                    if res[0] == 0:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #50C878 0%, #38A169 100%); 
                                    padding: 40px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 4px 20px rgba(80, 200, 120, 0.4); margin-top: 30px;'>
                            <h1 style='margin: 0; font-size: 3em; color: white; border: none;'>✅</h1>
                            <h2 style='margin: 20px 0 10px 0; color: white; border: none;'>เห็ดนี้กินได้</h2>
                            <h3 style='margin: 0; color: white; opacity: 0.9;'>(Edible Mushroom)</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #EE5A6F 100%); 
                                    padding: 40px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4); margin-top: 30px;'>
                            <h1 style='margin: 0; font-size: 3em; color: white; border: none;'>⚠️</h1>
                            <h2 style='margin: 20px 0 10px 0; color: white; border: none;'>เห็ดนี้มีพิษ</h2>
                            <h3 style='margin: 0; color: white; opacity: 0.9;'>(Poisonous Mushroom)</h3>
                        </div>
                        """, unsafe_allow_html=True)