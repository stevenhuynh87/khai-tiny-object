# Brief plan (bản 2): Đinh Tấn Khải, tiny object detection trong DETR

Đề tài: chi phí tính toán và độ bền của đặc trưng độ phân giải cao trong DETR ở điều kiện tài nguyên hạn chế. Paper nền: ReDHF (arXiv 2611.07994, NeurIPS 2026). Đầu ra: một bài IEEE Letter nộp trước 14/11/2027, tối đa lùi 2 sprint. Chi tiết trong `PLAN.md`; việc từng hai tuần trong tab Issues (#1 đến #26).

Giả định: 6 giờ mỗi tuần (12 giờ mỗi sprint), GPU 12 GB, không có Jetson, chỉ Python và PyTorch.

## Hướng đi

Phân tích chi phí và độ bền của đặc trưng độ phân giải cao, kèm một cải tiến nhẹ (chọn token top-k). Phần "biên" là proxy (CPU luồng cố định, GPU FP16, lượng tử hóa động), không phải số đo trên thiết bị biên, và bài phải nói rõ. Dự phòng: bài phân tích thuần nếu cải tiến không đạt ngưỡng.

## Lộ trình

| Giai đoạn | Sprint | Thời gian | Đầu ra chính |
|---|---|---|---|
| P0 Khởi động | 1-2 | 16/11/2026 - 13/12/2026 | Repo, môi trường, công cụ đo chi phí, ghi chú ReDHF |
| P1 Dữ liệu và baseline | 3-7 | 14/12/2026 - 21/02/2027 | VisDrone và AI-TOD-v2 dạng COCO, baseline, ngân sách tài nguyên, độ nhiễu seed |
| P2 Tái hiện ReDHF rút gọn | 8-12 | 22/02/2027 - 02/05/2027 | CSDF, EGSS, FSQI và bảng ablation rút gọn |
| P3 Chi phí proxy biên | 13-15 | 03/05/2027 - 13/06/2027 | Chi phí theo module, đo proxy CPU và GPU FP16, chốt câu hỏi nghiên cứu |
| P4 Độ bền | 16-17 | 14/06/2027 - 11/07/2027 | Giao thức độ bền rút gọn, bảng độ bền, giả thuyết |
| P5 Phương pháp đề xuất | 18-21 | 12/07/2027 - 05/09/2027 | Cải tiến top-k và kết quả so sánh |
| P6 Thí nghiệm cuối | 22-23 | 06/09/2027 - 03/10/2027 | Bảng, hình, gói tái lập |
| P7 Viết và nộp | 24-26 | 04/10/2027 - 14/11/2027 | Bản thảo, nộp bài, phát hành code |

## Các cổng quyết định

- Sprint 07: G1: baseline lặp lại được nội bộ (hai seed chênh lệch không quá 0.5 AP, đề xuất) và ngân sách tài nguyên đã đóng băng. Không đòi khớp số công bố, vì ngân sách của ta nhỏ hơn. SODA-D bị loại khỏi phạm vi.
- Sprint 12: G2: bản đầy đủ cho dấu tăng AP cùng chiều với paper, vượt ngưỡng ý nghĩa, trên ít nhất một tập. Nếu không đạt, hạ mục tiêu thành 'phân tích độc lập' (phương án B) và báo thầy trước khi sang P3.
- Sprint 15: G3: chốt câu hỏi nghiên cứu, danh sách đóng góp và tạp chí mục tiêu. Tiêu chí: phạm vi, giới hạn trang, chính sách preprint, và việc tạp chí có chấp nhận số đo proxy thay cho thiết bị biên không. Kiểm tra trên trang chính thức.
- Sprint 17: G4: chốt giả thuyết cải tiến. Nếu không giả thuyết nào có dữ liệu ủng hộ, chuyển sang phương án B (bài phân tích).
- Sprint 21: G5: phương pháp đạt một trong hai: giảm chi phí nhánh độ phân giải cao ít nhất 25% với mất AP không quá 0.3 (và không vượt ngưỡng ý nghĩa của sprint 7), hoặc cải thiện độ bền tương đối ít nhất 1.0 điểm. Ngưỡng là đề xuất, cần thầy chốt. Nếu không đạt, viết theo phương án B.

## Việc cần làm ngay

1. Khải: Issue 01 (môi trường, repo, đọc ReDHF lần 1).
2. Thầy: duyệt ngân sách tài nguyên ở Issue 04; chốt ngưỡng ở Issue 07.
3. Cả hai: nhớ rằng chưa thấy code công khai của ReDHF; tái hiện dựa trên mô tả trong paper.

## Rủi ro chính

Ngân sách giờ; không có code ReDHF; reviewer có thể không chấp nhận số đo proxy. Xem bảng đầy đủ ở `PLAN.md` mục 12.
