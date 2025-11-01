# 🌞 Solar-Synchronous Circadian Rhythm Optimization  
### Deep Learning Project using the MMASH Dataset

---

## 🧠 Overview
This project aims to **analyze, model, and optimize human circadian rhythms** using wearable-derived data.  
By studying sleep, heart rate, movement, stress, and hormonal data, the model estimates an individual’s **current circadian rhythm**, compares it to the **ideal solar-synchronous rhythm**, and generates a **personalized realignment plan**.

---

## 🧩 Objectives
1. Estimate each participant’s **current biological rhythm** using activity and physiological signals.  
2. Compute the **ideal rhythm** aligned with local solar midnight (based on latitude and sunlight).  
3. Train a **Neural Network (CNN + GRU)** to predict circadian phase from wearable data.  
4. Generate **personalized recommendations** (sleep and light exposure plans) to improve alignment.

---

## 📊 Dataset — MMASH (Multi-Modal Mental and Physical Health)
Dataset: [PhysioNet MMASH Dataset](https://physionet.org/content/mmash/)

**Contains:**
- Sleep timing and quality (`sleep.csv`)
- Heart rate and accelerometer data (`actigraph.csv`, `RR.csv`)
- Psychological scores (stress, anxiety, chronotype)
- Saliva-based hormone levels (melatonin, cortisol)
- User demographics (age, height, weight)

---

## 🧱 Repository Structure
