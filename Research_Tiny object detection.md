Tiny object detection: chi phí tính toán và độ bền của high-res feature trong DETR trên thiết bị biên
Paper liên quan
Paper #1 — Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection
Loại gap
Điều kiện/giả định bị bỏ qua
Hình thức đề tài
Master's thesis
1. Bối cảnh & hiện trạng
[Paper nêu] ReDHF đưa lại feature độ phân giải cao nhất vào DETR bằng CSDF, EGSS, FSQI và đạt +1,4/+1,6/+1,5 AP trên AI-TOD-v2/VisDrone/SODA-D.

2. Research Gap (khoảng trống)
[Suy luận] Abstract không nêu chi phí FLOPs/bộ nhớ/latency, cũng không thử trên đầu vào bị suy giảm (mờ, thiếu sáng). Chưa rõ lợi ích còn giữ khi chạy real-time trên edge.

3. Vì sao chưa được giải quyết
[Suy luận] Benchmark tiny object thường báo AP, ít khi báo đồng thời latency và độ bền với suy giảm ảnh.

4. Đóng góp kỳ vọng (Novelty)
Đường cong chi phí–độ chính xác của high-res feature và biến thể bền với suy giảm ảnh.

5. Rủi ro & tính khả thi
Khả thi: dataset công khai; cần GPU 1 card. Rủi ro: code ReDHF chưa nêu.

6. Target Output — Conference (ICORE/CORE từ B trở lên)
CVPR — IEEE/CVF Conf. on Computer Vision and Pattern Recognition · ICORE/CORE A* (ICORE2026) · nguồn ↗
WACV — IEEE/CVF Winter Conf. on Applications of Computer Vision · ICORE/CORE A (CORE2023) · nguồn ↗
ICIP — IEEE Int. Conf. on Image Processing · ICORE/CORE B (CORE2023) · nguồn ↗
BMVC — British Machine Vision Conf. · ICORE/CORE A (CORE2023) · nguồn ↗
7. Target Output — Journal (Scopus, Q3/Q4 trở lên)
IEEE Transactions on Circuits and Systems for Video Technology · Scopus · SJR Q1 (2025) · nguồn ↗
IET Computer Vision · Scopus · SJR Q2 (2025) · nguồn ↗
IEEE Transactions on Pattern Analysis and Machine Intelligence · Scopus · SJR Q1 (2023; nguồn thứ ba) · nguồn ↗
IEEE Transactions on Image Processing · Scopus (có trang SCImago) · quartile chưa xác minh · nguồn ↗
8. Research Opportunities (Solve · Improve · Optimize)
Solve Đo chi phí thật của high-res feature

Ý tưởng: Lập bảng FLOPs/bộ nhớ/latency cho ReDHF so với bỏ high-res feature trên nhiều DETR.

Cách làm:

Tái hiện ReDHF hoặc cài lại theo mô tả
Đo trên GPU và Jetson với TensorRT
Vẽ đường Pareto AP–latency
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

So với DINO/RT-DETR cùng ngân sách
Phân tích theo cỡ vật thể
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Nếu giả thuyết đúng, lấp được khoảng trống trong phạm vi điều kiện thử nghiệm; mức cải thiện chưa thể khẳng định trước khi chạy thí nghiệm.

Rủi ro & giảm thiểu: Giả thuyết có thể sai hoặc hiệu ứng nhỏ; giảm thiểu bằng pilot nhỏ trên một dataset trước khi mở rộng.

Solve Đánh giá dưới suy giảm ảnh

Ý tưởng: Kiểm tra AP khi ảnh mờ, nhiễu, thiếu sáng.

Cách làm:

Tạo bản suy giảm tổng hợp của AI-TOD
Chạy ReDHF và baseline
Phân tích đóng góp từng module
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

AP theo mức suy giảm
So sánh CSDF/EGSS/FSQI
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Nếu giả thuyết đúng, lấp được khoảng trống trong phạm vi điều kiện thử nghiệm; mức cải thiện chưa thể khẳng định trước khi chạy thí nghiệm.

Rủi ro & giảm thiểu: Giả thuyết có thể sai hoặc hiệu ứng nhỏ; giảm thiểu bằng pilot nhỏ trên một dataset trước khi mở rộng.

Improve Chọn vùng high-res thích ứng

Ý tưởng: Chỉ kích hoạt high-res feature ở vùng có khả năng chứa vật thể nhỏ.

Cách làm:

Dùng objectness thô để tạo mask
Fusion chỉ tại vùng mask
Huấn luyện end-to-end
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

AP_tiny so với ReDHF đầy đủ
Tỷ lệ vùng được kích hoạt
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Kỳ vọng tăng độ chính xác/độ bền/khái quát so với baseline ở metric đã nêu; không cam kết con số cụ thể.

Rủi ro & giảm thiểu: Cải thiện có thể nằm trong sai số giữa các seed; giảm thiểu bằng nhiều seed, kiểm định thống kê và baseline cùng ngân sách huấn luyện.

Improve Giám sát hình dạng khác ellipse

Ý tưởng: Thay giám sát ellipse bằng hàm phù hợp với hộp thực tế (ví dụ Gaussian có hướng).

Cách làm:

Cài biến thể giám sát
Huấn luyện trên VisDrone
So sánh với EGSS
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

AP và AP_tiny
Ổn định huấn luyện
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Kỳ vọng tăng độ chính xác/độ bền/khái quát so với baseline ở metric đã nêu; không cam kết con số cụ thể.

Rủi ro & giảm thiểu: Cải thiện có thể nằm trong sai số giữa các seed; giảm thiểu bằng nhiều seed, kiểm định thống kê và baseline cùng ngân sách huấn luyện.

Optimize Lượng tử hoá và chưng cất ReDHF

Ý tưởng: Nén ReDHF xuống INT8 mà giữ lợi ích tiny object.

Cách làm:

PTQ/QAT cho CSDF
Distill từ mô hình lớn
Biên dịch TensorRT
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

AP so với FP32
Latency Jetson
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Kỳ vọng giảm compute, bộ nhớ hoặc latency với suy giảm chất lượng trong ngưỡng chấp nhận; cần đo trên phần cứng thật (ví dụ Jetson/TensorRT hoặc GPU thường).

Rủi ro & giảm thiểu: Nén/rút gọn có thể làm mất chất lượng ở trường hợp khó; giảm thiểu bằng đánh giá theo tập con khó và đo end-to-end.

Optimize Cắt tỉa query toàn thang

Ý tưởng: Giảm số query trong FSQI theo độ tin cậy.

Cách làm:

Xếp hạng query theo score
Cắt tỉa tại suy luận
Đo ảnh hưởng đến recall vật thể nhỏ
Phát hành code, cấu hình và seed để tái lập (chỉ khi giấy phép dataset cho phép).
Kiểm chứng:

Recall tiny
Latency theo số query
Ablation loại từng thành phần mới, báo cáo trung bình ± độ lệch chuẩn qua nhiều seed.
Kỳ vọng: [Suy luận] Kỳ vọng giảm compute, bộ nhớ hoặc latency với suy giảm chất lượng trong ngưỡng chấp nhận; cần đo trên phần cứng thật (ví dụ Jetson/TensorRT hoặc GPU thường).

Rủi ro & giảm thiểu: Nén/rút gọn có thể làm mất chất lượng ở trường hợp khó; giảm thiểu bằng đánh giá theo tập con khó và đo end-to-end.

9. Từ khoá & chuỗi tìm kiếm (Literature Review)
tiny object detection DETR
high-resolution feature small object
AI-TOD VisDrone SODA-D
efficient DETR edge
10. Nhóm tài liệu cần review
DETR và biến thể deformable
Tiny/small object detection benchmark
Feature pyramid đa tỉ lệ
Real-time detection trên edge
11. Baseline · Dataset/Simulator · Metrics
Baseline: DETR variants gốc, DINO, RT-DETR
Dataset / Simulator: AI-TOD-v2, VisDrone, SODA-D
Metrics: AP, AP_tiny, FLOPs, latency (Jetson/GPU)