import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CPU OPTIMIZATION (รีดพลัง i5-12400F ให้เสถียรที่สุด)
# ============================================================
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.optimizer.set_jit(True) 

# ============================================================
# 2. SETTINGS (ปรับลด BATCH_SIZE แก้อาการ RAM เต็ม)
# ============================================================
IMAGE_DIR    = '../Dataset/archive/UTKFace' # 🌟 เช็ค Path ให้ดีนะโก๋
IMG_SIZE     = 128     # ขนาดที่สมดุลที่สุดระหว่างความไวและความชัด
BATCH_SIZE   = 16      # 🔧 ปรับลดจาก 32 เป็น 16 เพื่อเซฟ RAM ตอน Fine-tune
EPOCHS       = 40      
SAMPLE_SIZE  = 16000   

# ============================================================
# 3. METRICS & LOSS
# ============================================================
def accuracy_within_5_years(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= 5.0, tf.float32))

@tf.function
def huber_loss(y_true, y_pred, delta=3.0):
    """Huber Loss: ทนทานต่อข้อมูลที่ผิดปกติ (Outliers) ได้ดีกว่า MSE"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    error  = y_true - y_pred
    abs_e  = tf.abs(error)
    huber  = tf.where(abs_e <= delta,
                      0.5 * tf.square(error),
                      delta * abs_e - 0.5 * delta ** 2)
    return tf.reduce_mean(huber)

# ============================================================
# 4. LOAD DATA (คัดเฉพาะอายุ 10-60 ปี)
# ============================================================
print("\n--- กำลังโหลดรูปภาพ (กลยุทธ์ผู้เชี่ยวชาญเฉพาะทาง 10-60 ปี) ---")
filenames = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
np.random.shuffle(filenames)

images, ages = [], []
for fname in filenames:
    if len(images) >= SAMPLE_SIZE: break 
    try:
        age = int(fname.split('_')[0])
        
        # คัดเอาเฉพาะช่วงอายุที่เดาง่ายที่สุดคือ 10 ถึง 60 ปี
        if age < 10 or age > 60: continue 
        
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(IMAGE_DIR, fname), 
            target_size=(IMG_SIZE, IMG_SIZE)
        )
        img = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32) / 255.0
        images.append(img)
        ages.append(age)
    except: continue

X = np.array(images, dtype=np.float32)
y = np.array(ages, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"✅ โหลดสำเร็จ! Train: {len(X_train)} รูป | Test: {len(X_test)} รูป")

# ============================================================
# 5. TF.DATA PIPELINE (ถอด .cache() ออก เซฟ RAM)
# ============================================================
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.02)
    
    # Random Crop และ resize กลับ บังคับให้ดูโครงหน้า
    image = tf.image.resize(tf.image.random_crop(image, size=[115, 115, 3]), [IMG_SIZE, IMG_SIZE])
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

# 🔧 ถอด .cache() ออก เพื่อไม่ให้ยัดข้อมูลลง RAM รวดเดียวจนล้น
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    .shuffle(5000).map(augment, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 6. BUILD MODEL (ปรับยาแรงให้พอดี ไม่ให้โมเดลช้ำเกินไป)
# ============================================================
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False # Freeze ช่วงแรก

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# 🔧 ลด L2 และ Dropout ลงนิดนึง ให้โมเดลยังมีเซลล์ประสาทเหลือพอไปจำฟีเจอร์สำคัญ
x = Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.005))(x)
x = Dropout(0.4)(x) 
x = BatchNormalization()(x)

x = Dense(128, activation='swish', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(0.3)(x)

output = Dense(1, activation='linear', dtype='float32')(x)

model = Model(inputs=base.input, outputs=output)

# ============================================================
# 7. CALLBACKS
# ============================================================
checkpoint = ModelCheckpoint('age_model_best.h5', monitor='val_accuracy_within_5_years', save_best_only=True, mode='max', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_mae', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_mae', patience=7, restore_best_weights=True, verbose=1)

# ============================================================
# 8. PHASE 1: เทรนแค่สมองส่วนปลาย (3 รอบ)
# ============================================================
print("\n=== PHASE 1: ตั้งไข่โมเดล (3 Epochs) ===")
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), loss=huber_loss, metrics=['mae', accuracy_within_5_years])
model.fit(train_ds, epochs=3, validation_data=test_ds, verbose=1)

# ============================================================
# 9. PHASE 2: ปลดล็อกเต็มสูบ Fine-Tuning
# ============================================================
print("\n=== PHASE 2: Fine-Tuning ===")
base.trainable = True
for layer in base.layers[:100]: # Freeze 100 ชั้นแรกไว้
    layer.trainable = False

# ใช้ Learning Rate ที่ต่ำลงมากๆ เพื่อค่อยๆ จูนรายละเอียดริ้วรอย
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001), loss=huber_loss, metrics=['mae', accuracy_within_5_years])

history = model.fit(
    train_ds, 
    epochs=EPOCHS, 
    validation_data=test_ds, 
    callbacks=[checkpoint, reduce_lr, early_stop],
    verbose=1
)

model.save("age_model_final.h5")
print("\n🎉 เทรนเสร็จสมบูรณ์! บันทึกไฟล์ที่เก่งที่สุดไว้ที่ 'age_model_best.h5' เรียบร้อย")