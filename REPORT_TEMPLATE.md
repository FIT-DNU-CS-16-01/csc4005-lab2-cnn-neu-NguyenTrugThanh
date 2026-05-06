# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Họ và tên:
- Lớp:
- Repo: csc4005-lab2-cnn-neu-NguyenTrugThanh
- W&B project: csc4005-lab2-neu-cnn

## 2. Bài toán
Mô tả ngắn: phân loại ảnh grayscale bề mặt thép vào 6 lớp lỗi (Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches) sử dụng NEU-CLS.

## 3. Mô hình và cấu hình
### 3.1. MLP baseline từ Lab 1
- Run tham chiếu: `baseline_adamw` (Lab 1)

### 3.2. CNN from scratch
- Kiến trúc (tối thiểu): Conv(1->16)->ReLU->MaxPool -> Conv(16->32)->ReLU->MaxPool -> Conv(32->64)->ReLU -> AdaptiveAvgPool/Flatten -> FC(64->128)->ReLU+Dropout -> FC(128->6)
- Cấu hình tiêu biểu: `optimizer=adamw`, `lr=1e-3`, `weight_decay=1e-4`, `dropout=0.3`, `img_size=64`, `batch_size=32`.

### 3.3. Transfer learning
- Backbone dùng: `resnet18` (pretrained ImageNet). Có 2 kiểu run: `transfer` (freeze backbone, train head) và `finetune` (unfreeze và fine-tune).

## 4. Bảng kết quả
| Model | Train mode | Best Val Acc | Test Acc | Epoch time | Trainable Params | Nhận xét |
|---|---|---:|---:|---:|---:|---|
| MLP | scratch | 21.11% | 20.00% | — | — | Underfitting mạnh (baseline Lab 1) |
| CNN-small | scratch | 94.44% | 94.81% | 1.94 s | 32,614 | Học đặc trưng cục bộ tốt, cải thiện lớn so với MLP |
| ResNet18 | transfer / finetune | 100.00% (finetune) / 96.67% (transfer) | 100.00% (finetune) / 96.30% (transfer) | 22.46 s (finetune) / 9.46 s (transfer) | 11,179,590 (finetune) / 3,078 (transfer) | `transfer` chạy nhanh với ít params trainable; `finetune` đạt hiệu năng tốt nhất nhưng tốn tài nguyên |

## 5. Phân tích learning curves
- CNN-small hội tụ nhanh hơn và ổn định hơn so với MLP; val acc ~94% so với MLP ~21%.
- Transfer learning (ResNet18) cho curva hội tụ rất nhanh: `transfer` (freeze) có epoch nhanh và ít tham số trainable; `finetune` đạt hiệu năng cao nhất ở nghiệm này.
- Gap val/train nhỏ cho CNN-small → tổng quát tốt; finetune có tiềm năng overfit trên tập nhỏ nhưng bản ghi test cũng rất cao.

## 6. Confusion matrix và lỗi dự đoán sai
- Một số điểm từ confusion matrix (CNN-small):
	- `Inclusion` bị nhầm sang `Pitted_Surface` (5 mẫu).
	- `Pitted_Surface`: 2 mẫu nhầm `Inclusion`, 2 mẫu nhầm `Rolled-in_Scale`, 1 mẫu nhầm `Scratches`.
	- `Scratches` bị nhầm thành `Inclusion` (4 mẫu).
- ResNet18 (finetune) trên test ghi nhận confusion matrix gần như hoàn hảo (45/45 cho mỗi lớp trong bản lưu), nên lỗi dự đoán rất ít.
- Nguyên nhân lỗi phổ biến: đặc trưng vùng nhỏ/giống nhau giữa một số lớp, hoặc resize/augmentation làm mất chi tiết vi mô.

## 7. Kết luận
- CNN có cải thiện so với MLP không?
	- Có. CNN-small cải thiện rất lớn nhờ giữ được cấu trúc không gian, số liệu: val/test ~94% vs MLP ~20%.
- Transfer learning có tốt hơn không?
	- Có. `transfer` (freeze) cho kết quả rất tốt với chi phí train thấp (test ~96.3%), `finetune` đạt điểm cao nhất (100% trong run này) nhưng cần nhiều thời gian/param hơn.
- Khi nào nên chọn transfer learning thay vì train from scratch?
	- Khi dữ liệu huấn luyện hạn chế hoặc muốn tiết kiệm thời gian tuning: dùng `transfer` (freeze) để train nhanh với ít tham số trainable. Nếu có đủ dữ liệu/tài nguyên và muốn tối ưu hiệu năng, unfreeze và fine-tune một phần/backbone.

