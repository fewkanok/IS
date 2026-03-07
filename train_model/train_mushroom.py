import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ============================================================
# 1. SETTINGS & PATH
# ============================================================
# อ้างอิงจากรูป: ถ้ารันในโฟลเดอร์ models ต้องถอยกลับไป 1 ขั้น (../) แล้วเข้า Dataset
CSV_PATH = '../Dataset/archive2/mushrooms.csv'

print("🍄 กำลังโหลดข้อมูลเห็ด...")
if not os.path.exists(CSV_PATH):
    print(f"❌ หาไฟล์ไม่เจอ! เช็ค Path นี้ด่วน: {CSV_PATH}")
    exit()

df = pd.read_csv(CSV_PATH)
print(f"✅ โหลดสำเร็จ! จำนวนข้อมูลทั้งหมด: {df.shape[0]} ต้น, คุณลักษณะ: {df.shape[1]-1} อย่าง")

# ============================================================
# 2. DATA PREPROCESSING (แปลงตัวอักษรเป็นตัวเลข)
# ============================================================
print("\n⚙️ กำลังแปลงข้อมูลตัวอักษรเป็นตัวเลข...")

# แยกเฉลย (class) ออกจากฟีเจอร์ (X)
# class: e = edible (กินได้), p = poisonous (มีพิษ)
X = df.drop('class', axis=1)
y = df['class']

# สร้างตัวเข้ารหัสสำหรับเป้าหมาย (y)
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y) 
# e จะกลายเป็น 0, p จะกลายเป็น 1

# สร้างตัวเข้ารหัสสำหรับฟีเจอร์ (X) ทั้งหมด
# 🌟 ทริคสำคัญ: เราต้องเก็บ Encoder ของแต่ละคอลัมน์ไว้ เพื่อเอาไปใช้แปลงข้อมูลบนหน้าเว็บ
encoders = {}
X_encoded = pd.DataFrame()

for column in X.columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])
    encoders[column] = le  # เก็บใส่ดิกชันนารีไว้

print("✅ แปลงข้อมูลเสร็จสิ้น!")

# ============================================================
# 3. TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)
print(f"📊 แบ่งข้อมูล Train: {len(X_train)} ต้น | Test: {len(X_test)} ต้น")

# ============================================================
# 4. BUILD & TRAIN MODEL (Ensemble: Random Forest)
# ============================================================
print("\n🌲 กำลังเทรนโมเดล Random Forest (Ensemble)...")
# ใช้ต้นไม้ 100 ต้น ช่วยกันโหวตคำตอบ
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ============================================================
# 5. EVALUATION
# ============================================================
print("\n📈 ประเมินความแม่นยำ (Validation):")
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"🎯 ความแม่นยำ (Accuracy): {acc * 100:.2f}%")
print("-" * 50)
print(classification_report(y_test, y_pred, target_names=['Edible (กินได้ - 0)', 'Poisonous (มีพิษ - 1)']))

# ============================================================
# 6. SAVE MODEL & ENCODERS
# ============================================================
# บันทึกตัวโมเดล
joblib.dump(model, 'mushroom_model.pkl')
# 🌟 บันทึกตัวแปลงข้อมูล (สำคัญมากสำหรับหน้าเว็บ)
joblib.dump(encoders, 'mushroom_encoders.pkl')
joblib.dump(le_y, 'mushroom_target_encoder.pkl')

# ============================================================
# 7. FEATURE IMPORTANCE (โชว์ความโปร่งใสของ AI)
# ============================================================
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False)
print("\n🌟 5 คุณสมบัติหลักที่ AI ใช้แยกเห็ดพิษ:")
print(importances.head(5))

print("\n🎉 บันทึกไฟล์เสร็จสิ้น!")
print("ไฟล์ที่ได้:")
print("1. mushroom_model.pkl (สมอง AI)")
print("2. mushroom_encoders.pkl (ดิกชันนารีแปลงข้อมูลฟีเจอร์)")
print("3. mushroom_target_encoder.pkl (ตัวแปลงคำตอบ e/p)")