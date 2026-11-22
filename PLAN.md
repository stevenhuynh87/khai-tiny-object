# Kế hoạch nghiên cứu 12 tháng (bản 2): Đinh Tấn Khải

Đề tài: Tiny object detection, chi phí tính toán và độ bền của đặc trưng độ phân giải cao trong DETR ở điều kiện tài nguyên hạn chế.
Paper nền (Paper #1): Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF).
Mục tiêu đầu ra: một bài báo ngắn nộp tạp chí IEEE Letters.
Repo: https://github.com/stevenhuynh87/khai-tiny-object
Thời gian: 16/11/2026 đến 14/11/2027, 26 sprint hai tuần.

Bản 2 thay bản đầu theo ba giả định mới: Khải làm 6 giờ mỗi tuần (12 giờ mỗi sprint), GPU 12 GB, lab không có Jetson, và toàn bộ code chỉ dùng Python và PyTorch.

## 1. Kết luận

Hướng khuyến nghị giữ nguyên (phương án A: phân tích chi phí và độ bền kèm một cải tiến nhẹ), nhưng phạm vi thu hẹp rõ rệt. Bỏ TensorRT và Jetson, bỏ SODA-D, bỏ INT8 toàn mô hình, giảm số loại corruption, chỉ làm một ý tưởng cải tiến. Phần "thiết bị biên" trở thành proxy: CPU với số luồng cố định, GPU 12 GB ở FP16, và lượng tử hóa động của PyTorch. Bài báo phải ghi đúng là proxy và không được viết là đo trên thiết bị biên.
Yếu tố quyết định là khả thi vận hành (C13 mức 6) chặn lại phạm vi, trong khi chất lượng học thuật (mức 1) giữ lại các cổng kiểm tra. Với 312 giờ tổng cộng, kế hoạch vừa đủ; bất kỳ sprint nào trễ hai tuần sẽ đẩy ngày nộp lùi.
Rủi ro cao nhất: ngân sách giờ, không có code ReDHF, và việc reviewer có chấp nhận số đo proxy thay cho thiết bị biên hay không.
Bước tiếp theo: Khải làm Issue 01; thầy duyệt ngân sách tài nguyên ở Issue 04 và chốt ngưỡng ở Issue 07.

## 2. Bối cảnh và những gì đã xác minh

Đã xác minh trong phiên lập kế hoạch (đọc trang arXiv và NeurIPS):

- ReDHF: Zehao Wang, Si Wu, Guiping Cao, Xin Li, Yaowei Wang; arXiv 2611.07994, nộp ngày 5/11/2026, chấp nhận NeurIPS 2026. Ba thành phần: CSDF (Cross-Scale Deformable Fusion), EGSS (Ellipse-Guided Shallow Supervision), FSQI (Full-Scale Query Interaction). Mức tăng AP tốt nhất mà abstract nêu: +1.4 trên AI-TOD-v2, +1.6 trên VisDrone, +1.5 trên SODA-D.
- Abstract không nêu FLOPs, latency, bộ nhớ hay độ bền. Không thấy liên kết code trên trang arXiv.
- Paper liên quan đã kiểm tra tồn tại: Dome-DETR (ACM MM 2025, abstract nêu có code), TinyFormer (arXiv 2605.25046), DQ-DETR, SO-DETR, UHR-DETR, DFIR-DETR, RT-DETR, RT-DETRv2.

Chưa xác minh, cần làm ở Sprint 1 và 8:

- Chi tiết phương pháp, bảng chi phí, hạn chế và danh sách tham khảo của ReDHF: mới đọc abstract, chưa đọc toàn văn.
- Một số ID arXiv trong danh sách đọc lấy từ kiến thức nền, đánh dấu "cần kiểm tra". Khải phải xác nhận từng ID trước khi đưa vào `refs.bib`.

Giả định của kế hoạch (do thầy cung cấp):

- 6 giờ mỗi tuần, tức khoảng 12 giờ mỗi sprint; tổng khoảng 312 giờ.
- Một GPU 12 GB. Không có Jetson. Chỉ Python và PyTorch.

Hệ quả của ràng buộc:

- 12 GB VRAM: dùng backbone nhỏ pretrained, ảnh 640 px, AMP, batch nhỏ kèm gradient accumulation. Không thể tái hiện đúng recipe của paper gốc, nên bài chỉ so với baseline do mình huấn luyện cùng ngân sách.
- Chỉ PyTorch: deformable attention dùng bản `grid_sample` thuần PyTorch (chậm hơn bản CUDA). Không xuất ONNX, không TensorRT, không plugin.
- Không Jetson: không có số đo trên thiết bị biên. Thay bằng latency CPU với số luồng cố định, latency GPU ở FP16, FLOPs, params, bộ nhớ đỉnh, và lượng tử hóa động của PyTorch cho lớp Linear.

## 3. Phản biện kế hoạch

Có ba điểm cần nói thẳng.

Thứ nhất, 6 giờ mỗi tuần chỉ bằng một nửa công suất của bản đầu. Mỗi sprint có 12 giờ để đọc, viết code và làm báo cáo; GPU chạy nền qua đêm không tính vào giờ làm việc. Vì vậy mỗi sprint chỉ có 1 đến 2 paper cần đọc, nhiều chỗ là "đọc lướt", và phạm vi bị cắt.

Thứ hai, "chi phí trên thiết bị biên" khi không có thiết bị biên là điểm yếu trước reviewer. Cách giảm: gọi đúng tên là proxy, nêu rõ cấu hình CPU và luồng, và nói trong Limitations rằng chưa đo trên phần cứng nhúng. Nếu có người đo giúp trên Jetson trong giai đoạn review, thêm vào bản sửa. Không khẳng định hơn những gì đã đo.

Thứ ba, tái hiện ReDHF dưới ngân sách nhỏ có thể không ra dấu tăng AP, nhất là khi chưa có code gốc. Vì vậy sprint 7 đo độ nhiễu giữa hai seed, và các cổng G1, G2, G5 đều dùng ngưỡng "có ý nghĩa" tính từ độ nhiễu này, không dùng so sánh trực tiếp với số công bố.

Ước tính thời gian huấn luyện trên GPU 12 GB chưa có số thật. Sprint 5 đo, và nếu tổng khoảng 25 đến 30 lần huấn luyện của kế hoạch vượt khả năng thì phải cắt cấu hình (xem mục 9).

## 4. Câu hỏi nghiên cứu và các phương án đóng góp

Câu hỏi chính: đặc trưng độ phân giải cao đóng góp bao nhiêu AP cho tiny object so với chi phí ở điều kiện tài nguyên hạn chế, và nó có làm detector kém bền với nhiễu không?

Giả thuyết kiểm chứng (đề xuất, chưa chứng minh):

- H1: chi phí tăng chủ yếu do số token ở stride thấp và cross-attention ở decoder, nên chọn lọc token giảm chi phí với mất AP nhỏ.
- H2 (chỉ làm nếu còn thời gian): đặc trưng độ phân giải cao tăng độ nhạy với nhiễu cao tần; một bộ lọc nhẹ cải thiện độ bền.

Ba phương án đóng góp:

- Phương án A (khuyến nghị): phân tích chi phí và độ bền kèm cải tiến H1. Phù hợp khi G2 và G5 đạt. Đánh đổi: ít dư địa về giờ.
- Phương án B: bài phân tích thuần (so sánh chi phí và độ bền của các cách đưa đặc trưng độ phân giải cao vào DETR). Phù hợp khi cải tiến không đạt G5; đóng góp phương pháp yếu và phụ thuộc tạp chí chấp nhận bài kiểu benchmark.
- Phương án C: module mới độc lập với ReDHF. Chỉ cân nhắc nếu G2 thất bại nặng; rủi ro thời gian cao nhất, không khuyến nghị với 6 giờ mỗi tuần.

Khuyến nghị A, quyết định theo C13: khả thi vận hành giới hạn phạm vi, chất lượng học thuật giữ qua các cổng. Phương án B là dự phòng sẵn sàng.

Tạp chí mục tiêu chọn ở cổng G3 theo phạm vi, giới hạn trang, chính sách preprint, và thái độ với số đo proxy. Phải kiểm tra trên trang chính thức; kế hoạch này chưa xác minh các giới hạn đó.

## 5. Tổng quan giai đoạn

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

Sprint 6 trùng dịp Tết Nguyên đán nên chỉ khoảng 8 giờ. Mỗi sprint khác có khoảng 12 giờ làm việc cộng thời gian GPU chạy nền.

## 6. Các cổng quyết định

- Sprint 07: G1 (hết sprint 7): baseline lặp lại được nội bộ (hai seed chênh lệch không quá 0.5 AP, đề xuất) và ngân sách tài nguyên đã đóng băng. Không đòi khớp số công bố, vì ngân sách của ta nhỏ hơn. SODA-D bị loại khỏi phạm vi.
- Sprint 12: G2 (hết sprint 12): bản đầy đủ cho dấu tăng AP cùng chiều với paper, vượt ngưỡng ý nghĩa, trên ít nhất một tập. Nếu không đạt, hạ mục tiêu thành 'phân tích độc lập' (phương án B) và báo thầy trước khi sang P3.
- Sprint 15: G3 (hết sprint 15): chốt câu hỏi nghiên cứu, danh sách đóng góp và tạp chí mục tiêu. Tiêu chí: phạm vi, giới hạn trang, chính sách preprint, và việc tạp chí có chấp nhận số đo proxy thay cho thiết bị biên không. Kiểm tra trên trang chính thức.
- Sprint 17: G4 (hết sprint 17): chốt giả thuyết cải tiến. Nếu không giả thuyết nào có dữ liệu ủng hộ, chuyển sang phương án B (bài phân tích).
- Sprint 21: G5 (hết sprint 21): phương pháp đạt một trong hai: giảm chi phí nhánh độ phân giải cao ít nhất 25% với mất AP không quá 0.3 (và không vượt ngưỡng ý nghĩa của sprint 7), hoặc cải thiện độ bền tương đối ít nhất 1.0 điểm. Ngưỡng là đề xuất, cần thầy chốt. Nếu không đạt, viết theo phương án B.

## 7. Chi tiết từng sprint

Chi tiết task code và paper của từng sprint nằm trong tab Issues của repo (#1 đến #26, số issue trùng số sprint). Phần dưới tóm tắt mục tiêu, bước thực hiện và paper đọc.

### P0 Khởi động

#### Sprint 01 (16/11/2026 - 29/11/2026): Môi trường, repo và đọc paper chính lần 1

Mục tiêu: Có repo chạy được, ghi chú đọc ReDHF lần 1 và danh sách câu hỏi mở.

Các bước thực hiện:

1. Tạo repo `khai-tiny-object` với cấu trúc `configs/ src/ scripts/ docs/ results/ paper/ notes/`.
1. Dựng môi trường Python (conda hoặc uv), ghim phiên bản PyTorch, torchvision trong `requirements.txt`. Không dùng thư viện cần biên dịch CUDA riêng.
1. Viết `scripts/env_check.py`: in phiên bản, tên GPU, VRAM, chạy thử một phép nhân ma trận ở FP16 có đồng bộ CUDA.
1. Thiết lập pre-commit (ruff, black) và quy ước nhánh `issue-XX-ten-ngan`; chọn TensorBoard làm công cụ log.

Đọc paper:

- Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF), Wang et al., NeurIPS 2026 (arXiv:2611.07994; đã xác minh). Trả lời: Pass 1 (abstract, hình, kết luận): bài toán, ba thành phần CSDF, EGSS, FSQI ở mức ý tưởng. Ghi 5 câu hỏi mở vào `notes/redhf_pass1.md`.

Hoàn thành khi: `env_check.py` chạy sạch trên GPU 12 GB; `notes/redhf_pass1.md` có ít nhất 5 câu hỏi mở; PR đầu tiên đã merge.

Lưu ý: Trang arXiv của ReDHF chưa liệt kê code công khai. Ghi vào `notes/redhf_code_status.md` và hỏi thầy có nên liên hệ tác giả không.

#### Sprint 02 (30/11/2026 - 13/12/2026): Nền tảng DETR và công cụ đo chi phí bằng PyTorch

Mục tiêu: Hiểu luồng DETR đa tỉ lệ và có công cụ đo params, FLOPs, latency, bộ nhớ.

Các bước thực hiện:

1. Chạy suy luận một DETR pretrained (torchvision hoặc hub) trên 10 ảnh, vẽ box, lưu `results/sanity/`.
1. Viết `src/utils/profile.py`: params, FLOPs (`torch.utils.flop_counter.FlopCounterMode`), latency (warmup 50, đo 200 lần, đồng bộ CUDA), bộ nhớ đỉnh (`torch.cuda.max_memory_allocated`).
1. Viết test cho `profile.py` trên một mô hình nhỏ có FLOPs tính tay được.
1. Vẽ sơ đồ kích thước tensor của Deformable DETR vào `docs/detr_dataflow.md`.

Đọc paper:

- Deformable DETR: Deformable Transformers for End-to-End Object Detection, ICLR 2021 (arXiv:2010.04159; ID từ kiến thức nền, cần kiểm tra). Trả lời: Multi-scale deformable attention: số điểm lấy mẫu, số tỉ lệ, chi phí theo số token. Ghi lại cách cài đặt thuần PyTorch bằng `grid_sample`, vì ta sẽ dùng bản này.
- End-to-End Object Detection with Transformers (DETR), ECCV 2020 (arXiv:2005.12872; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chỉ đọc lướt: object query và Hungarian matching.

Hoàn thành khi: Có `results/profile_sanity.csv`; test của `profile.py` pass.


### P1 Dữ liệu và baseline

#### Sprint 03 (14/12/2026 - 27/12/2026): Dữ liệu VisDrone

Mục tiêu: VisDrone ở dạng COCO, có thống kê kích thước object kiểm chứng được. AI-TOD-v2 để sprint 6.

Các bước thực hiện:

1. Viết `scripts/download_visdrone.sh`; ghi nguồn và giấy phép vào `docs/datasets.md`.
1. Chuyển nhãn về COCO JSON, kiểm tra số ảnh, số box, box lỗi.
1. Viết `scripts/dataset_stats.py`: phân bố kích thước box, tỉ lệ box dưới 8 px, 8-16 px, 16-32 px.
1. Dataloader và test; vẽ 30 ảnh có box để kiểm tra bằng mắt.

Đọc paper:

- Detection and Tracking Meet Drones Challenge (VisDrone), TPAMI 2021 (tìm trên IEEE Xplore; ID từ kiến thức nền, cần kiểm tra). Trả lời: Cấu trúc nhãn, số lớp, vùng ignore.
- Tiny Object Detection in Aerial Images (AI-TOD), Wang et al., ICPR 2020 (tìm trên arXiv/IEEE Xplore; ID từ kiến thức nền, cần kiểm tra). Trả lời: Đọc phần định nghĩa very tiny, tiny, small và metric; chuẩn bị cho sprint 6.

Hoàn thành khi: `docs/datasets.md` có bảng thống kê; test dataloader pass.

#### Sprint 04 (28/12/2026 - 10/01/2027): Chọn codebase chạy thuần PyTorch trên GPU 12 GB

Mục tiêu: Một baseline DETR chạy end-to-end (train, eval) ở quy mô nhỏ, không cần biên dịch CUDA.

Các bước thực hiện:

1. Chọn codebase theo tiêu chí: code công khai, giấy phép cho phép, có đường chạy thuần PyTorch cho deformable attention, vừa 12 GB. Ứng viên: Deformable DETR (có bản `ms_deform_attn_core_pytorch`), repo Dome-DETR (abstract nêu có code), RT-DETR. Ghi `docs/adr-001-codebase.md`.
1. Đặt ngân sách tài nguyên: backbone nhỏ (ResNet-18 hoặc ResNet-50 pretrained), ảnh 640 px, AMP, batch nhỏ kèm gradient accumulation. Ghi vào `docs/compute_budget.md`.
1. Smoke test 1 epoch trên 5% dữ liệu, kiểm tra loss giảm và VRAM không tràn.
1. Cho `profile.py` tự ghi params, FLOPs, latency vào CSV sau mỗi lần eval.

Đọc paper:

- DQ-DETR: DETR with Dynamic Query for Tiny Object Detection (arXiv:2404.03507; đã xác minh). Trả lời: Cách chọn query động cho tiny object; chi phí tăng bao nhiêu?
- Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection, ACM MM 2025 (arXiv:2505.05741; đã xác minh). Trả lời: Chỉ đọc cách dùng density map để thưa hóa attention trên đặc trưng độ phân giải cao (MWAS) và bảng chi phí.

Hoàn thành khi: Smoke test pass; ADR được thầy duyệt.

Lưu ý: Nếu codebase chọn buộc dùng op CUDA tự biên dịch, chuyển sang bản thuần PyTorch. Đổi lại tốc độ chậm hơn; đã tính trong ngân sách.

#### Sprint 05 (11/01/2027 - 24/01/2027): Huấn luyện baseline VisDrone

Mục tiêu: Baseline VisDrone có AP, APs, APt kèm params, FLOPs, latency và ước lượng thời gian huấn luyện thật.

Các bước thực hiện:

1. Chạy huấn luyện đầy đủ theo ngân sách, fine-tune từ trọng số pretrained, cố định seed, chạy nền qua đêm; lưu checkpoint tốt nhất.
1. Viết `scripts/aggregate_results.py` gộp log thành `results/baseline_visdrone.csv`.
1. Viết script đánh giá theo nhóm kích thước, vì metric COCO chuẩn che mất nhóm tiny.
1. Ghi thời gian chạy thật vào `docs/compute_actuals.md`, rồi nhân với số lần chạy còn lại trong plan. Nếu tổng vượt khả năng, báo thầy để cắt cấu hình.

Đọc paper:

- RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-time Detection Transformer (arXiv:2407.17140; đã xác minh). Trả lời: Chỉ đọc danh sách bag-of-freebies: mục nào rẻ và đáng bật trong baseline.

Hoàn thành khi: `results/baseline_visdrone.csv` có ít nhất một lần chạy đầy đủ; `docs/compute_actuals.md` có số liệu thật.

#### Sprint 06 (25/01/2027 - 07/02/2027): Dữ liệu và baseline AI-TOD-v2

Mục tiêu: AI-TOD-v2 ở dạng COCO và có baseline. Sprint nhẹ vì trùng dịp Tết Nguyên đán (kiểm tra lịch năm 2027).

Các bước thực hiện:

1. Tải và chuyển AI-TOD-v2 sang COCO bằng script tái dùng từ sprint 3; chạy `dataset_stats.py`.
1. Chạy baseline AI-TOD-v2 theo ngân sách, nền qua đêm; ghi `results/baseline_aitod.csv`.
1. Quan sát: tiny object ở AI-TOD nhỏ hơn VisDrone nhiều, nên có thể phải tăng độ phân giải hoặc giảm batch. Ghi lại thay đổi nếu có.

Đọc paper:

- DEIM: DETR with Improved Matching for Fast Convergence, CVPR 2025 (arXiv:2412.04234; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chỉ đọc cơ chế matching và vì sao hội tụ nhanh hơn; ảnh hưởng tới số epoch phải chạy.

Hoàn thành khi: Baseline AI-TOD-v2 chạy xong; thống kê dữ liệu có trong `docs/datasets.md`.

#### Sprint 07 (08/02/2027 - 21/02/2027): Báo cáo baseline và đo độ nhiễu giữa các seed

Mục tiêu: Biết baseline đáng tin tới mức nào và mức chênh lệch nhỏ nhất có thể tin được.

Các bước thực hiện:

1. Chạy lại baseline VisDrone với seed thứ hai; tính chênh lệch AP giữa hai seed.
1. Viết `docs/baseline_report.md`: so với số công bố (Dome-DETR, ReDHF) và giải thích chênh lệch do backbone, độ phân giải, số epoch. Bài báo chỉ so với baseline do mình huấn luyện cùng ngân sách.
1. Đóng băng ngân sách tài nguyên và đặt ngưỡng 'có ý nghĩa' bằng khoảng 2 lần độ chênh giữa hai seed.

Đọc paper:

- DFIR-DETR: Frequency-Domain Iterative Refinement and Dynamic Feature Aggregation for Small Object Detection (arXiv:2512.07078; đã xác minh). Trả lời: Chỉ đọc cách chỉnh neck của RT-DETR để so với ý tưởng của ReDHF.

Hoàn thành khi: Báo cáo baseline và ngưỡng ý nghĩa được thầy ký.

Cổng quyết định: G1 (hết sprint 7): baseline lặp lại được nội bộ (hai seed chênh lệch không quá 0.5 AP, đề xuất) và ngân sách tài nguyên đã đóng băng. Không đòi khớp số công bố, vì ngân sách của ta nhỏ hơn. SODA-D bị loại khỏi phạm vi.


### P2 Tái hiện ReDHF rút gọn

#### Sprint 08 (22/02/2027 - 07/03/2027): Đọc sâu ReDHF và đo chi phí đặc trưng độ phân giải cao thô

Mục tiêu: Hiểu thiết kế ReDHF và định lượng chi phí khi thêm đặc trưng stride thấp nhất một cách ngây thơ.

Các bước thực hiện:

1. Đọc toàn văn (có thể đọc TeX source). Viết `docs/redhf_design.md` với kích thước tensor ở từng bước của CSDF, EGSS, FSQI.
1. Cài `src/models/highres_naive.py`: đưa đặc trưng stride thấp nhất vào encoder không chọn lọc.
1. Đo số token, FLOPs, latency, bộ nhớ so với baseline; kiểm tra có vừa 12 GB không. Vẽ `results/naive_highres_cost.png`.
1. Ghi các chi tiết paper không nói rõ vào `docs/redhf_ambiguities.md`, kèm giả định.

Đọc paper:

- Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF), Wang et al., NeurIPS 2026 (arXiv:2611.07994; đã xác minh). Trả lời: Pass 2 và 3: phương pháp, thí nghiệm, ablation, hạn chế. Tìm bảng chi phí nếu có.

Hoàn thành khi: `docs/redhf_design.md` được thầy duyệt; có số chi phí của phương án ngây thơ.

#### Sprint 09 (08/03/2027 - 21/03/2027): Cài đặt CSDF

Mục tiêu: CSDF chạy đúng hình dạng, gradient và có test.

Các bước thực hiện:

1. Cài `src/models/csdf.py` theo thiết kế, bằng `grid_sample`, có cờ bật tắt trong config.
1. Test: kích thước tensor, gradient khác không, kết quả xác định theo seed.
1. Overfit 100 ảnh để kiểm tra loss giảm; đo chi phí thêm.

Đọc paper:

- Feature Pyramid Networks for Object Detection (arXiv:1612.03144; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chỉ đọc con đường top-down, để so với cách CSDF đưa ngữ nghĩa xuống đặc trưng nông.

Hoàn thành khi: Test pass; có `results/cost_csdf.csv`.

#### Sprint 10 (22/03/2027 - 04/04/2027): Cài đặt EGSS

Mục tiêu: Supervision phụ elip chạy được và có test.

Các bước thực hiện:

1. Cài EGSS: sinh vùng elip từ box (trục theo w và h), gán nhãn phụ cho đặc trưng nông, loss phụ có hệ số trong config.
1. Test hình dạng vùng elip, trường hợp box rất nhỏ, và gradient.
1. Overfit 100 ảnh với CSDF và EGSS cùng bật.

Đọc paper:

- A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (arXiv:2110.13389; ID từ kiến thức nền, cần kiểm tra). Trả lời: Vì sao mô hình hóa box nhỏ bằng Gaussian giúp gán nhãn? So sánh với elip trong EGSS.

Hoàn thành khi: Test pass; overfit thành công.

#### Sprint 11 (05/04/2027 - 18/04/2027): Cài đặt FSQI và ma trận cấu hình

Mục tiêu: FSQI chạy được, ba thành phần bật tắt độc lập.

Các bước thực hiện:

1. Cài FSQI: cho query ở decoder truy cập cả đặc trưng độ phân giải cao lẫn cấp ngữ nghĩa cao.
1. Tạo `configs/redhf_*.yaml` cho bốn cấu hình: baseline, +CSDF, +CSDF+EGSS, đầy đủ.
1. Test FSQI; kiểm tra bốn cấu hình đều vừa 12 GB.

Đọc paper:

- DINO: DETR with Improved DeNoising Anchor Boxes (arXiv:2203.03605; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chỉ đọc cách DINO dùng đặc trưng nhiều tỉ lệ cho query, để tham chiếu cho FSQI.

Hoàn thành khi: Bốn cấu hình khởi chạy được; test pass.

#### Sprint 12 (19/04/2027 - 02/05/2027): Chạy tái hiện và ablation rút gọn

Mục tiêu: Bảng ablation bốn cấu hình trên VisDrone và so sánh baseline với đầy đủ trên AI-TOD-v2.

Các bước thực hiện:

1. Huấn luyện bốn cấu hình trên VisDrone (một seed, nền qua đêm); baseline và đầy đủ trên AI-TOD-v2.
1. Lập `results/ablation_repro.csv` và bảng LaTeX tự sinh.
1. So với paper nêu: tăng AP tốt nhất +1.4 (AI-TOD-v2) và +1.6 (VisDrone). Khác biệt dưới ngưỡng 'có ý nghĩa' của sprint 7 thì ghi là chưa kết luận.
1. Nếu không ra dấu tăng, dùng tối đa 1 sprint để tìm nguyên nhân, ghi `docs/repro_debug.md`.

Đọc paper:

- TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors, 2026 (arXiv:2605.25046; đã xác minh). Trả lời: Đọc phần SSA và PBM: cách khác để đưa đặc trưng độ phân giải cao vào token; ghi điểm khác biệt.

Hoàn thành khi: Bảng ablation và kết luận tái hiện thành công hay chưa.

Cổng quyết định: G2 (hết sprint 12): bản đầy đủ cho dấu tăng AP cùng chiều với paper, vượt ngưỡng ý nghĩa, trên ít nhất một tập. Nếu không đạt, hạ mục tiêu thành 'phân tích độc lập' (phương án B) và báo thầy trước khi sang P3.


### P3 Chi phí proxy biên

#### Sprint 13 (03/05/2027 - 16/05/2027): Đo chi phí theo module và độ phân giải

Mục tiêu: Biết module nào gây chi phí khi bật đặc trưng độ phân giải cao.

Các bước thực hiện:

1. Viết `scripts/profile_modules.py` dùng `torch.profiler`: thời gian backbone, encoder, decoder, từng thành phần ReDHF.
1. Quét độ phân giải đầu vào (512, 640, 800) cho baseline và bản ReDHF; ghi FLOPs, bộ nhớ đỉnh, latency GPU ở FP32 và FP16.
1. Đo latency CPU với số luồng cố định (ví dụ 4, `torch.set_num_threads`) làm proxy thiết bị biên; ghi rõ CPU và điều kiện đo vào `docs/proxy_protocol.md`.

Đọc paper:

- DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) (arXiv:2304.08069; đã xác minh). Trả lời: Đọc phần phân tích latency: cách họ đo, điều kiện công bằng. Áp dụng cho `proxy_protocol.md`.

Hoàn thành khi: Bảng chi phí theo module, độ phân giải, kiểu số học; `proxy_protocol.md` được duyệt.

Lưu ý: Không có Jetson, nên mọi số 'biên' là proxy. Bài báo phải nói rõ là proxy (CPU luồng giới hạn và GPU FP16), không được viết là đo trên thiết bị biên.

#### Sprint 14 (17/05/2027 - 30/05/2027): Đường Pareto và khung bài báo

Mục tiêu: Biểu đồ Pareto và bộ khung bài báo LaTeX.

Các bước thực hiện:

1. Viết `scripts/plot_pareto.py`: AP và APt theo FLOPs, bộ nhớ, latency GPU và CPU.
1. Viết `docs/cost_analysis.md`: yếu tố chi phí chính (số token ở stride thấp, số điểm lấy mẫu, cross-attention ở decoder).
1. Tạo `paper/main.tex` từ template IEEE, viết nháp Introduction và dàn ý Related Work; tạo `paper/refs.bib`.

Đọc paper:

- Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity (arXiv:2111.14330; ID từ kiến thức nền, cần kiểm tra). Trả lời: Cơ chế thưa hóa token ở encoder; có áp dụng được cho đặc trưng stride thấp không?

Hoàn thành khi: Có biểu đồ Pareto; `paper/main.tex` biên dịch được.

#### Sprint 15 (31/05/2027 - 13/06/2027): Nén bằng PyTorch và chốt câu hỏi nghiên cứu

Mục tiêu: Biết FP16 và lượng tử hóa động ảnh hưởng tiny object thế nào, và chốt hướng bài báo.

Các bước thực hiện:

1. Đo AP và APt ở FP16 và bf16 (nếu GPU hỗ trợ).
1. Thử `torch.ao.quantization.quantize_dynamic` cho các lớp Linear trên CPU; ghi AP và latency. Ghi rõ đây là lượng tử hóa động, không phải INT8 toàn mô hình.
1. Soạn `docs/research_question.md`: câu hỏi chính, giả thuyết, đóng góp dự kiến (mục 4 của PLAN).
1. Họp thầy chốt hướng và tạp chí mục tiêu.

Đọc paper:

- A White Paper on Neural Network Quantization (arXiv:2106.08295; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chỉ đọc phần PTQ và lượng tử hóa động; ghi hạn chế của việc chỉ dùng PyTorch.
- Author guidelines của tạp chí IEEE Letters mục tiêu (giới hạn trang, template, chính sách preprint) (trang tạp chí trên ieee.org; cần kiểm tra). Trả lời: Đọc phạm vi và giới hạn trang của hai tạp chí ứng viên để chọn.

Hoàn thành khi: `docs/research_question.md` được thầy ký duyệt.

Cổng quyết định: G3 (hết sprint 15): chốt câu hỏi nghiên cứu, danh sách đóng góp và tạp chí mục tiêu. Tiêu chí: phạm vi, giới hạn trang, chính sách preprint, và việc tạp chí có chấp nhận số đo proxy thay cho thiết bị biên không. Kiểm tra trên trang chính thức.


### P4 Độ bền

#### Sprint 16 (14/06/2027 - 27/06/2027): Giao thức và thí nghiệm độ bền rút gọn

Mục tiêu: Bộ dữ liệu validation bị làm hỏng tái lập được, và bảng độ bền của hai mô hình.

Các bước thực hiện:

1. Viết `scripts/make_corrupted_val.py` bằng numpy, PIL và torch: nhiễu Gaussian, mờ Gaussian, nén JPEG, thiếu sáng; 3 mức độ; seed cố định.
1. Viết `scripts/eval_robust.py`: AP, APt, độ giảm tương đối so với dữ liệu sạch.
1. Kiểm tra bằng mắt 10 ảnh mỗi loại; ghi `docs/robustness_protocol.md`.
1. Đánh giá baseline và ReDHF đầy đủ trên VisDrone validation.

Đọc paper:

- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations (arXiv:1903.12261; ID từ kiến thức nền, cần kiểm tra). Trả lời: Cách định nghĩa corruption và mức độ; ghi lại để trích dẫn đúng.

Hoàn thành khi: Giao thức được duyệt; có `results/robustness_*.csv`.

#### Sprint 17 (28/06/2027 - 11/07/2027): Phân tích độ bền và giả thuyết

Mục tiêu: Biết thành phần nào nhạy nhiễu và chốt giả thuyết cải tiến.

Các bước thực hiện:

1. Vẽ độ giảm AP theo mức độ cho từng loại corruption.
1. Trực quan hóa đáp ứng của đặc trưng độ phân giải cao trên ảnh nhiễu (heatmap) cho 3 ảnh để kiểm tra giả thuyết nhiễu nền dày đặc.
1. Phân tích lỗi: tỉ lệ bỏ sót theo kích thước và corruption.
1. Viết `docs/robustness_findings.md`; chọn giả thuyết H1 (và H2 nếu còn thời gian).

Đọc paper:

- Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming (arXiv:1907.07484; ID từ kiến thức nền, cần kiểm tra). Trả lời: Chuyển giao giao thức độ bền sang phát hiện đối tượng; metric độ bền tương đối.

Hoàn thành khi: Findings được thầy duyệt; giả thuyết có tiêu chí bác bỏ rõ ràng.

Cổng quyết định: G4 (hết sprint 17): chốt giả thuyết cải tiến. Nếu không giả thuyết nào có dữ liệu ủng hộ, chuyển sang phương án B (bài phân tích).


### P5 Phương pháp đề xuất

#### Sprint 18 (12/07/2027 - 25/07/2027): Thiết kế phương pháp và kiểm tra tính mới

Mục tiêu: Một ý tưởng cải tiến có thiết kế, và nhật ký tìm kiếm chứng minh tính mới.

Các bước thực hiện:

1. Ý tưởng A (đề xuất, chưa kiểm chứng): chọn top-k vị trí trên đặc trưng độ phân giải cao theo điểm objectness rồi mới lấy mẫu deformable. Ý tưởng B (bộ lọc khử nhiễu) chỉ làm nếu A xong sớm.
1. Viết `docs/method_design.md` có sơ đồ tensor và ước lượng chi phí lý thuyết.
1. Viết `docs/novelty_check.md`: truy vấn đã chạy (arXiv, Google Scholar, IEEE Xplore), paper gần nhất, điểm khác biệt.

Đọc paper:

- Lite DETR: An Interleaved Multi-Scale Encoder for Efficient DETR (arXiv:2303.07335; ID từ kiến thức nền, cần kiểm tra). Trả lời: Mã hóa đa tỉ lệ xen kẽ để giảm chi phí; điểm tương đồng với ý tưởng A.
- Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection, ACM MM 2025 (arXiv:2505.05741; đã xác minh). Trả lời: So sánh MWAS với ý tưởng A; ghi rõ điểm khác để tránh trùng.

Hoàn thành khi: Thiết kế được duyệt; nhật ký tính mới có ít nhất 10 truy vấn.

Lưu ý: Nếu ý tưởng A quá gần MWAS của Dome-DETR, thầy và Khải quyết định ở sprint này, trước khi viết code.

#### Sprint 19 (26/07/2027 - 08/08/2027): Cài đặt ý tưởng A

Mục tiêu: Ý tưởng A chạy được, có test và số đo chi phí vi mô.

Các bước thực hiện:

1. Cài `src/models/topk_hr.py` với tham số k và ngưỡng trong config.
1. Test hình dạng, gradient và trường hợp k lớn hơn số token.
1. Microbenchmark latency và bộ nhớ theo k; overfit 100 ảnh.

Đọc paper:

- DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (arXiv:2106.02034; ID từ kiến thức nền, cần kiểm tra). Trả lời: Thưa hóa token động: cách huấn luyện với mặt nạ token để áp dụng khi cắt token.

Hoàn thành khi: Test pass; đường cong chi phí theo k.

#### Sprint 20 (09/08/2027 - 22/08/2027): Huấn luyện ý tưởng A trên VisDrone

Mục tiêu: Bảng AP và chi phí theo k trên VisDrone.

Các bước thực hiện:

1. Huấn luyện ba giá trị k trên VisDrone (nền qua đêm).
1. Cập nhật `results/method_vs_baseline.csv`; chọn một k.
1. Đánh giá độ bền của k đã chọn theo giao thức P4.

Đọc paper:

- SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection (arXiv:2504.11470; đã xác minh). Trả lời: Đọc lướt phần miền tần số và distillation; nguồn ý tưởng cho ý tưởng B nếu có thời gian.

Hoàn thành khi: Bảng theo k; chọn được cấu hình đi tiếp.

#### Sprint 21 (23/08/2027 - 05/09/2027): So sánh trên AI-TOD-v2 và cổng G5

Mục tiêu: Cấu hình chọn được chạy trên AI-TOD-v2 và so với baseline và ReDHF.

Các bước thực hiện:

1. Huấn luyện cấu hình chọn trên AI-TOD-v2.
1. So sánh với baseline và ReDHF về AP, APt, FLOPs, bộ nhớ, latency GPU và CPU proxy.
1. Đánh giá theo ngưỡng G5; nếu không đạt, quyết định chuyển sang phương án B.

Đọc paper:

- UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery (arXiv:2604.21435; đã xác minh). Trả lời: Đọc lướt cách xử lý ảnh siêu phân giải cho small object, để đặt kết quả vào bối cảnh.

Hoàn thành khi: Bảng kết quả chính nháp.

Cổng quyết định: G5 (hết sprint 21): phương pháp đạt một trong hai: giảm chi phí nhánh độ phân giải cao ít nhất 25% với mất AP không quá 0.3 (và không vượt ngưỡng ý nghĩa của sprint 7), hoặc cải thiện độ bền tương đối ít nhất 1.0 điểm. Ngưỡng là đề xuất, cần thầy chốt. Nếu không đạt, viết theo phương án B.


### P6 Thí nghiệm cuối

#### Sprint 22 (06/09/2027 - 19/09/2027): Chạy thí nghiệm cuối cùng

Mục tiêu: Mọi con số trong bài có nguồn từ file kết quả và có khoảng sai số.

Các bước thực hiện:

1. Chạy thêm seed thứ hai cho các hàng chính trên VisDrone (và AI-TOD-v2 nếu còn compute); nền qua đêm.
1. Viết `scripts/make_tables.py` sinh bảng LaTeX từ CSV, không sửa tay.
1. Đóng băng checkpoint cuối, ghi hash vào `results/CHECKPOINTS.md`.

Đọc paper:

- Author guidelines của tạp chí IEEE Letters mục tiêu (trang tạp chí trên ieee.org; cần kiểm tra). Trả lời: Đọc hướng dẫn tác giả: giới hạn trang, định dạng, chính sách preprint, chính sách dùng AI.

Hoàn thành khi: Bảng sinh tự động; checkpoint đóng băng.

#### Sprint 23 (20/09/2027 - 03/10/2027): Hình vẽ và gói tái lập

Mục tiêu: Bộ hình đủ cho bài báo và repo tái lập được.

Các bước thực hiện:

1. Vẽ sơ đồ kiến trúc, đường Pareto, ảnh định tính (thành công và thất bại), đường cong độ bền.
1. Dọn code, viết README có lệnh tái lập từng bảng; chạy toàn bộ test.
1. Kiểm tra giấy phép dữ liệu và code kế thừa.

Đọc paper:

- Ba bài IEEE Letters gần đây của tạp chí mục tiêu (trang tạp chí trên ieee.org; cần kiểm tra). Trả lời: Cấu trúc, độ dài mục, cách trình bày bảng.

Hoàn thành khi: Bộ hình chốt; repo chạy được từ máy sạch.


### P7 Viết và nộp

#### Sprint 24 (04/10/2027 - 17/10/2027): Bản nháp 1

Mục tiêu: Bản thảo đầy đủ trong giới hạn trang của tạp chí.

Các bước thực hiện:

1. Hoàn thiện Introduction, Related Work, Method, Experiments, Conclusion (một phần đã viết từ sprint 14).
1. Mỗi câu khẳng định liên kết tới bảng, hình hoặc trích dẫn; mô tả rõ phần 'biên' là proxy.
1. Kiểm tra số trang; cắt theo giới hạn; cập nhật `refs.bib` và kiểm tra từng mục với nguồn.

Đọc paper:

- Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF), Wang et al., NeurIPS 2026 (arXiv:2611.07994; đã xác minh). Trả lời: Đọc lại để mô tả đúng ReDHF trong Related Work.

Hoàn thành khi: Bản nháp 1 gửi thầy.

#### Sprint 25 (18/10/2027 - 31/10/2027): Bản nháp 2 và rà soát nội bộ

Mục tiêu: Bản thảo đã qua rà soát và sửa theo nhận xét.

Các bước thực hiện:

1. Sửa theo nhận xét; lập bảng 'nhận xét, cách xử lý'.
1. Đối chiếu từng con số trong bài với `results/`.
1. Kiểm tra trùng lặp văn bản; ghi nhận việc dùng công cụ AI theo yêu cầu của tạp chí; nhờ một bạn trong lab đọc thử.

Đọc paper:

- Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection, ACM MM 2025 (arXiv:2505.05741; đã xác minh). Trả lời: Đọc lại để mô tả đúng điểm khác biệt.

Hoàn thành khi: Bản nháp 2 sạch lỗi; đối chiếu số liệu đã ký.

#### Sprint 26 (01/11/2027 - 14/11/2027): Nộp bài và phát hành code

Mục tiêu: Bài đã nộp, code đã phát hành, có kế hoạch phản hồi review.

Các bước thực hiện:

1. Chuẩn bị cover letter, tên và đơn vị tác giả, khai báo dữ liệu và code.
1. Gắn tag phát hành trên GitHub; quyết định đăng arXiv theo chính sách của tạp chí.
1. Nộp, lưu mã bài; ghi `notes/after_submission.md` gồm hạn trả lời review và thí nghiệm dự phòng.

Đọc paper:

- Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF), Wang et al., NeurIPS 2026 (arXiv:2611.07994; đã xác minh). Trả lời: Kiểm tra lại trạng thái code công khai của ReDHF và ghi nếu có thay đổi.

Hoàn thành khi: Có mã nộp bài, tag GitHub, `notes/after_submission.md`.

Lưu ý: Sprint đệm: nếu bản thảo trễ, lùi nộp bài tối đa 2 sprint (đến cuối tháng 12/2027). Cảnh báo ở mục 12 của PLAN.


## 8. Danh sách đọc tổng hợp

Quy ước: "đã xác minh" nghĩa là trang paper đã được mở và đối chiếu trong phiên lập kế hoạch; "ID từ kiến thức nền, cần kiểm tra" nghĩa là Khải phải xác nhận định danh trước khi trích dẫn. Với 12 giờ mỗi sprint, quy trình đọc là: lượt 1 (abstract, hình, kết luận) cho mọi paper; lượt 2 và 3 (phương pháp, thí nghiệm) chỉ cho ReDHF, Deformable DETR và các paper ghi "đọc kỹ" trong issue. Mỗi paper kết thúc bằng ghi chú 10 đến 15 dòng trong `notes/`.

| Paper | Định danh | Trạng thái | Sprint |
|---|---|---|---|
| Reviving the Discarded High-Resolution Feature for Transformer-Based Tiny Object Detection (ReDHF), Wang et al., NeurIPS 2026 | arXiv:2611.07994 | đã xác minh | 1, 8, 24, 26 |
| Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection, ACM MM 2025 | arXiv:2505.05741 | đã xác minh | 4, 18, 25 |
| TinyFormer: Preserving Tiny Objects in YOLO-DETR Hybrid Real-time Detectors, 2026 | arXiv:2605.25046 | đã xác minh | 12 |
| DQ-DETR: DETR with Dynamic Query for Tiny Object Detection | arXiv:2404.03507 | đã xác minh | 4 |
| SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection | arXiv:2504.11470 | đã xác minh | 20 |
| UHR-DETR: Efficient End-to-End Small Object Detection for Ultra-High-Resolution Remote Sensing Imagery | arXiv:2604.21435 | đã xác minh | 21 |
| DFIR-DETR: Frequency-Domain Iterative Refinement and Dynamic Feature Aggregation for Small Object Detection | arXiv:2512.07078 | đã xác minh | 7 |
| DETRs Beat YOLOs on Real-time Object Detection (RT-DETR) | arXiv:2304.08069 | đã xác minh | 13 |
| RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-time Detection Transformer | arXiv:2407.17140 | đã xác minh | 5 |
| End-to-End Object Detection with Transformers (DETR), ECCV 2020 | arXiv:2005.12872 | ID từ kiến thức nền, cần kiểm tra | 2 |
| Deformable DETR: Deformable Transformers for End-to-End Object Detection, ICLR 2021 | arXiv:2010.04159 | ID từ kiến thức nền, cần kiểm tra | 2 |
| DINO: DETR with Improved DeNoising Anchor Boxes | arXiv:2203.03605 | ID từ kiến thức nền, cần kiểm tra | 11 |
| Feature Pyramid Networks for Object Detection | arXiv:1612.03144 | ID từ kiến thức nền, cần kiểm tra | 9 |
| DEIM: DETR with Improved Matching for Fast Convergence, CVPR 2025 | arXiv:2412.04234 | ID từ kiến thức nền, cần kiểm tra | 6 |
| A Normalized Gaussian Wasserstein Distance for Tiny Object Detection | arXiv:2110.13389 | ID từ kiến thức nền, cần kiểm tra | 10 |
| Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity | arXiv:2111.14330 | ID từ kiến thức nền, cần kiểm tra | 14 |
| Lite DETR: An Interleaved Multi-Scale Encoder for Efficient DETR | arXiv:2303.07335 | ID từ kiến thức nền, cần kiểm tra | 18 |
| DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification | arXiv:2106.02034 | ID từ kiến thức nền, cần kiểm tra | 19 |
| Benchmarking Neural Network Robustness to Common Corruptions and Perturbations | arXiv:1903.12261 | ID từ kiến thức nền, cần kiểm tra | 16 |
| Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming | arXiv:1907.07484 | ID từ kiến thức nền, cần kiểm tra | 17 |
| A White Paper on Neural Network Quantization | arXiv:2106.08295 | ID từ kiến thức nền, cần kiểm tra | 15 |
| Tiny Object Detection in Aerial Images (AI-TOD), Wang et al., ICPR 2020 | tìm trên arXiv/IEEE Xplore | ID từ kiến thức nền, cần kiểm tra | 3 |
| Detection and Tracking Meet Drones Challenge (VisDrone), TPAMI 2021 | tìm trên IEEE Xplore | ID từ kiến thức nền, cần kiểm tra | 3 |
| Author guidelines của tạp chí IEEE Letters mục tiêu (giới hạn trang, template, chính sách preprint) | trang tạp chí trên ieee.org | ID từ kiến thức nền, cần kiểm tra | 15, 22, 23 |

## 9. Kế hoạch compute

- Huấn luyện: một GPU 12 GB; chạy nền qua đêm; ước lượng thời gian thật cập nhật ở Sprint 5 (`docs/compute_actuals.md`).
- Tổng số lần huấn luyện dự kiến khoảng 25 đến 30: 2 baseline VisDrone, 1 baseline AI-TOD-v2, 4 cấu hình ReDHF trên VisDrone, 1 ReDHF trên AI-TOD-v2, 3 giá trị k, 1 cấu hình chọn trên AI-TOD-v2, seed bổ sung cho hàng chính. Nếu một lần chạy quá 12 giờ thì cắt: bỏ AI-TOD-v2 cho ReDHF, hoặc giảm epoch, hoặc giảm số giá trị k.
- Biên (proxy): latency CPU với số luồng cố định, latency GPU FP16, FLOPs, params, bộ nhớ đỉnh, lượng tử hóa động cho lớp Linear. Không có số đo phần cứng nhúng.
- Dữ liệu và checkpoint lưu ngoài repo (Drive hoặc ổ lab); repo chỉ chứa config, script, kết quả CSV.

## 10. Quy trình làm việc

- Mỗi sprint một issue (tab Issues của repo, #1 đến #26), một nhánh, một Pull Request; thầy review trong 3 ngày làm việc.
- Họp cuối sprint 20 đến 30 phút; Khải gửi tóm tắt 5 dòng trước 24 giờ.
- Mọi con số trong bài có nguồn từ `results/`, bảng sinh tự động.
- Thay đổi hướng đi ghi thành ADR trong `docs/`.
- Viết bài rải đều: khung ở Sprint 14, Related Work ở 15 đến 18, Method ở 19 đến 20, Experiments ở 22 đến 23, nên bản nháp 1 ở Sprint 24 không phải viết từ đầu.

## 11. Chỉ số thành công

- Tái hiện: dấu tăng AP cùng chiều với paper gốc, vượt ngưỡng ý nghĩa (G2).
- Phương pháp: giảm ít nhất 25% chi phí nhánh độ phân giải cao với mất AP không quá 0.3, hoặc tăng độ bền tương đối ít nhất 1.0 điểm (G5; ngưỡng chờ thầy chốt).
- Quy trình: bản thảo nộp trước 14/11/2027, tối đa lùi 2 sprint; repo chạy được từ máy sạch.

## 12. Rủi ro và cách giảm

| Rủi ro | Dấu hiệu | Giảm thiểu |
|---|---|---|
| Hết giờ (6 giờ mỗi tuần) | Hai sprint liên tiếp không xong issue | Cắt theo thứ tự: ý tưởng B, AI-TOD-v2 cho ReDHF, seed bổ sung; lùi nộp tối đa 2 sprint |
| Không có code ReDHF, tái hiện không ra | G2 thất bại | Giới hạn 1 sprint debug; chuyển phương án B |
| 12 GB VRAM không đủ | OOM ở Sprint 4 hoặc 8 | Giảm độ phân giải, backbone nhỏ hơn, gradient checkpointing; ghi vào bài |
| Deformable attention thuần PyTorch quá chậm | Một lần chạy quá 12 giờ | Giảm epoch, giảm số lần chạy, giảm kích thước dữ liệu huấn luyện |
| Reviewer không chấp nhận proxy thay thiết bị biên | Phản hồi review | Gọi đúng tên proxy, nêu Limitations, xin đo giúp trên Jetson khi sửa bài |
| Cải tiến trùng công trình có sẵn (Dome-DETR, Sparse DETR) | Nhật ký tính mới | Quyết định ở Sprint 18 trước khi viết code |
| Trích dẫn sai | ID chưa kiểm | Kiểm tra từng mục `refs.bib` ở Sprint 24 và 25 |

## 13. Cải tiến liên tục

Cuối mỗi giai đoạn, thầy và Khải xem lại: ước lượng giờ có đúng không, GPU có đủ không, cần cắt phần nào. Điều chỉnh ghi vào cuối file này kèm ngày và lý do.

Bước tiếp theo: Khải làm Issue 01; thầy duyệt ngân sách tài nguyên ở Issue 04 và chốt ngưỡng ở Issue 07.
