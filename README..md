# 🏭 Industrial High-Speed Quality Control System (YOLOv8n)

This project is an **Automated Vision Inspection System** designed for high-speed production lines. It ensures product integrity by verifying lids, labels, and expiration dates in real-time.

The primary goal is to identify defective products on the conveyor belt and trigger a rejection mechanism (sorting) to remove them from the line.

## 🚀 Key Performance Metrics

- **Accuracy:** Achieved an outstanding **99.9% detection accuracy** during testing.
- **Throughput:** Capable of inspecting **30 products per minute**, matching industrial conveyor speeds.
- **Architecture:** Powered by **YOLOv8n (Nano)** for ultra-fast inference with minimal latency, ensuring the system can make "split-second" rejection decisions.

## 🛠️ Features

- **Dual-Camera Verification:** Uses top and side views to cross-check product data (e.g., Lid must match the side label).
- **Expiration Date Detection:** Instantly detects missing or misprinted expiration dates (`expdate` vs `noexpdate`).
- **Defect Rejection Logic:** Built-in decision-making logic to categorize products as **APPROVED** or **REJECTED**.
- **Visual Feedback:** A real-time status panel provides color-coded alerts for operators.

## 📂 Project Structure

- `ciftkamera.py`: The core engine handling dual-camera synchronization and decision logic.
- `reader.py`: Single-camera test utility for quick diagnostics and model verification.
- `shdfinal.pt`: The production-ready YOLOv8n weights (Optimized for 99% accuracy).

## 💻 Getting Started

### Prerequisites
```bash
pip install ultralytics opencv-python numpy
```
Execution
To start the dual-camera inspection system: python ciftkamera.py

🔍 Inspection Logic
Top View: Inspects lid type and checks for the presence of the expiration date.

Side View: Identifies the product flavor/type via the side label.

Cross-Check: The system compares both views. If the top label and side label do not match, or if the date is missing, the product is flagged for rejection.

Developed for high-efficiency manufacturing and quality assurance.