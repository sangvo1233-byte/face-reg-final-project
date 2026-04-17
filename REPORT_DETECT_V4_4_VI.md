# Báo Cáo Chi Tiết Detect V4.4

Thời điểm lập báo cáo: 18/04/2026

## 1. Tóm tắt điều hành

`Detect V4.4` là runtime quét khuôn mặt và chống giả mạo thế hệ mới của hệ thống chấm công khuôn mặt. Phiên bản này chạy song song với `V3`, nhưng mở rộng đáng kể phần anti-spoof bằng cách kết hợp nhiều lớp tín hiệu: moire FFT nhiều dải tần, screen-context detection, phone-rectangle detection, passive liveness, streaming liveness và active challenge.

Xét trên codebase hiện tại, V4.4 không còn là nhánh thử nghiệm rời rạc. Nó đã được tích hợp xuyên suốt từ backend đến frontend:

- Runtime backend: `core/detect_v4.py`, `core/runtime_v4.py`
- API và WebSocket: `app/routes/scan_v4.py`
- Local direct runner: `core/local_runner_v4.py`
- Frontend auto scan và debug overlay: `web/js/scan.js`
- Khai báo capability và version: `main.py`, `app/routes/system.py`

Kết luận ngắn:

- V4.4 là một runtime production-ready ở mức kiến trúc và luồng xử lý.
- So với V3, V4.4 mạnh hơn rõ rệt ở bài toán chống replay bằng điện thoại hoặc màn hình.
- Bộ test hiện có bao phủ các logic lõi quan trọng, nhưng chưa phải coverage end-to-end đầy đủ.

## 2. Mục tiêu của Detect V4.4

Detect V4.4 được thiết kế để xử lý tốt hơn các tình huống giả mạo thực tế mà V3 khó phân biệt, đặc biệt là:

- Phát lại khuôn mặt bằng điện thoại dựng dọc.
- Phát lại trên màn hình có viền hoặc phản xạ sáng.
- Các case moire không quá rõ nhưng vẫn có ngữ cảnh giống màn hình.

Theo mô tả trong `dev/detect-v4.4.py`, V4.4 kế thừa và mở rộng các ý tưởng từ các bản trước:

- V4.1: screen-context evidence
- V4.2: phone rectangle evidence
- V4.3: enhanced moire pipeline
- V4.4: portrait phone ROI và tăng độ nhạy phát hiện khung điện thoại

Điểm cốt lõi của V4.4 là chuyển từ mô hình “một detector quyết định tất cả” sang mô hình “hợp nhất nhiều tín hiệu nghi ngờ”.

## 3. Vị trí của V4.4 trong hệ thống

Trong `config.py`, hệ thống đang khai báo `APP_VERSION = "4.4.0"`. Trong `main.py`, endpoint `/version` trả về:

- `detect_runtime: "v4.4"`
- `scan_versions: ["v3", "v4.4"]`

Ngoài ra, router tổng trong `app/routes/__init__.py` đã include `scan_v4_router`. Điều này xác nhận V4.4 là một phần chính thức của API runtime hiện tại.

## 4. Kiến trúc tổng thể

### 4.1 Các thành phần chính

1. `core/detect_v4.py`
   Chứa các detector và helper service cho kết quả scan.

2. `core/runtime_v4.py`
   Runtime trung tâm, độc lập transport. Browser WebSocket và local-direct đều đưa frame vào đây. Runtime chịu trách nhiệm cho detect face, liveness, anti-spoof, match embedding, challenge và ghi nhận attendance.

3. `app/routes/scan_v4.py`
   Cung cấp HTTP compatibility endpoint, browser WebSocket stream, local-direct runner control và local-direct event WebSocket.

4. `core/local_runner_v4.py`
   Chạy vòng quét V4.4 bằng camera phía server và broadcast event cho frontend.

5. `web/js/scan.js`
   Chọn mode scan, kết nối WebSocket, hiển thị challenge overlay và debug overlay.

### 4.2 Mô hình transport

V4.4 hỗ trợ hai mode vận hành chính:

- `browser_ws`
- `local_direct`

Với `browser_ws`, camera nằm ở phía trình duyệt, frame JPEG được gửi qua `/ws/scan-v4`.

Với `local_direct`, camera nằm ở phía server, frontend chỉ subscribe event qua `/ws/scan-v4/local`.

Frontend có chế độ `auto` để tự chọn mode phù hợp theo hostname và trạng thái camera server.

## 5. Luồng xử lý runtime

Luồng xử lý một frame trong `DetectV4RuntimeSession` có thể tóm tắt như sau:

1. Nhận frame.
2. Cập nhật `StreamingLivenessTracker`.
3. Nếu đang có active challenge thì chuyển vào nhánh xử lý challenge.
4. Nếu đang cooldown sau khi điểm danh thì trả event trạng thái.
5. Nếu chưa đến chu kỳ detect tiếp theo thì trả event trạng thái nhẹ.
6. Nếu đến chu kỳ detect thì:
   - detect face bằng InsightFace
   - với từng face:
     - moire rolling
     - screen context
     - phone rectangle
     - passive liveness
     - streaming liveness
     - match embedding
     - ra quyết định `block`, `challenge`, `unknown`, hoặc `attendance`

## 6. Quy tắc ra quyết định chính

Runtime ưu tiên block sớm khi có tín hiệu mạnh:

- Moire rolling là `block` hoặc moire dưới ngưỡng block.
- Streaming liveness trả về `spoof`.
- Screen context ở mức `strong` và đi cùng moire hoặc passive suspicious.
- Phone rectangle rolling ở mức `block`.
- Phone rectangle ở mức `strong` và đồng thời có moire, context hoặc passive nghi ngờ.
- Passive liveness ở mức `block`.

Nếu chưa đủ điều kiện block nhưng vẫn có tín hiệu nghi ngờ:

- Runtime tạo `challenge_required`.

Nếu khuôn mặt là người thật, match đúng và không có lý do nghi ngờ:

- Runtime ghi nhận điểm danh.

## 7. Các lớp detector trong V4.4

### 7.1 Enhanced Moire Detector

`MoireDetectorV4` là detector quan trọng nhất của V4.4.

Đặc điểm chính:

- Phân tích FFT ở kích thước `128x128`
- Tách thành nhiều band: `low_mid`, `mid`, `mid_high`, `high`
- Dùng các tín hiệu: peak ratio, periodicity, anisotropy, grid score
- Trả về `moire_score` và `decision_hint`

Thiết kế này mạnh hơn V3 vì không chỉ hỏi “có moire hay không” mà đánh giá mức độ giống tín hiệu màn hình.

### 7.2 Rolling Moire Decision

`RollingMoireDecision` giữ lịch sử tối đa 18 mẫu cho từng track khuôn mặt.

Các chỉ số tổng hợp chính:

- `mean_score`
- `min_score`
- `p10_score`
- số mẫu suspicious, block, clean

Rule chính:

- Block nếu có `block_count >= 1` hoặc `min_score < MOIRE_BLOCK_THRESHOLD`
- Suspicious nếu có mẫu nghi ngờ hoặc `mean_score` hoặc `p10_score` xuống dưới ngưỡng screen
- Clean nếu đủ mẫu sạch và `mean_score` đủ cao

Điểm mạnh của lớp này là giảm jitter và tăng độ ổn định theo chuỗi frame, thay vì quyết định quá nặng ở một frame đơn lẻ.

### 7.3 Screen Context Detector

`ScreenContextDetectorV41` phân tích vùng xung quanh mặt để tìm dấu hiệu “đang nhìn vào màn hình”.

Hai nhóm đặc trưng chính:

- `flatness`
  - Laplacian variance
  - gradient entropy
  - color standard deviation
  - edge density
  - boundary edge strength
- `glare`
  - bright ratio
  - highlight blob count
  - highlight area ratio
  - linear glare score
  - highlight edge sharpness

Các signal đầu ra:

- `flat_background`
- `glass_glare`

Decision đầu ra:

- `clean`
- `suspicious`
- `strong`

Detector này đặc biệt hữu ích cho những case điện thoại full-screen, khi moire không đủ mạnh nhưng bối cảnh xung quanh lại rất giống màn hình phẳng.

### 7.4 Phone Rectangle Detector

`PhoneRectangleDetectorV42` là phần khác biệt nổi bật của V4.4.

Ý tưởng của detector này là:

- Mở ROI theo chiều dọc quanh khuôn mặt
- Tìm contour hoặc rectangle giống viền điện thoại hoặc khung màn hình
- Chấm điểm rectangle theo các tiêu chí:
  - rectangularity
  - aspect ratio
  - face-inside score
  - rect area ratio
  - border edge score
  - black border score
  - corner score
  - margin score

Các signal điển hình:

- `face_inside_rect`
- `dark_border`
- `sharp_rect_edge`
- `rect_corners`

Đây là lớp detector làm rõ nhất tinh thần của bản V4.4: chống replay bằng điện thoại dựng dọc trong môi trường thực tế.

### 7.5 Rolling Phone Rectangle Decision

`RollingPhoneRectDecision` lưu lịch sử 8 mẫu.

Rule chính:

- `block` nếu đủ số lần `strong` liên tiếp theo ngưỡng cấu hình
- `suspicious` nếu có 1 strong, hoặc nhiều suspicious, hoặc mean score cao
- `clean` nếu có đủ mẫu sạch

Logic rolling này giúp giảm false positive từ background ngẫu nhiên có hình chữ nhật.

### 7.6 Passive và Streaming Liveness

V4.4 tái sử dụng:

- `core.anti_spoof` cho passive liveness
- `StreamingLivenessTracker` cho blink và movement tracking theo stream

Decision passive hiện dùng lại ngưỡng của V3:

- `block` nếu dưới `DETECT_V3_LIVENESS_BLOCK_THRESHOLD`
- `suspicious` nếu dưới `DETECT_V3_LIVENESS_CHALLENGE_THRESHOLD`
- `pass` nếu đạt ngưỡng

Điểm cần lưu ý là V4.4 chưa có namespace config passive riêng.

## 8. Challenge flow trong V4.4

Khi match đúng danh tính nhưng còn tín hiệu nghi ngờ, runtime không pass ngay mà yêu cầu challenge.

### 8.1 Loại challenge

Các challenge hiện có:

- `BLINK`
- `TURN_LEFT`
- `TURN_RIGHT`

Frontend nhận các type API-friendly:

- `blink`
- `turn_left`
- `turn_right`

### 8.2 Tham số challenge

- Timeout: `7 giây`
- Cooldown sau khi pass: `12 giây`
- Pose baseline frames: `5`
- Pose pass frames: `2`
- Turn threshold: `0.06`

### 8.3 Luồng challenge

1. Runtime sinh `challenge_required`.
2. Frontend hiển thị hướng dẫn cho người dùng.
3. Runtime tiếp tục nhận frame để xác minh.
4. Runtime kiểm tra:
   - còn đúng danh tính hay không
   - có bị block bởi moire trong lúc challenge hay không
   - đã blink hoặc quay đầu đúng hướng hay chưa
5. Nếu pass:
   - ghi nhận attendance
6. Nếu fail hoặc timeout:
   - trả `challenge_failed`

Đây là điểm cân bằng tốt của V4.4: không quá cứng tay ở nhóm tín hiệu nghi ngờ vừa phải, nhưng vẫn không dễ cho replay qua mặt.

## 9. API và transport của V4.4

### 9.1 HTTP endpoint

- `POST /api/scan/v4`

Vai trò:

- Endpoint tương thích cho scan một ảnh
- Chủ yếu dùng cho compatibility hoặc test
- Không phải luồng production chính vì V4 phụ thuộc stream liveness và rolling state

### 9.2 Browser WebSocket

- `WS /ws/scan-v4`

Vai trò:

- Nhận frame JPEG từ camera trình duyệt
- Trả lại event scan, challenge và attendance theo thời gian thực

Payload `stream_ready` hiện bao gồm:

- `target_fps`
- `detect_fps`
- `client_frame_width`
- `client_jpeg_quality`
- `scan_version = "v4.4"`

### 9.3 Local Direct Runner API

- `POST /api/scan/v4/local/start`
- `POST /api/scan/v4/local/stop`
- `GET /api/scan/v4/local/status`
- `WS /ws/scan-v4/local`

Vai trò:

- Quản lý scanner dùng camera server
- Frontend chỉ subscribe event, không cần gửi frame

## 10. Tích hợp frontend

Frontend V4.4 nằm chủ yếu trong `web/js/scan.js`.

### 10.1 Chế độ scan

Frontend hỗ trợ:

- `auto`
- `local_direct`
- `browser_ws`

Rule `auto`:

- Nếu truy cập từ localhost hoặc LAN và camera server sẵn sàng, ưu tiên `local_direct`
- Nếu không, dùng `browser_ws`

### 10.2 Trạng thái runtime đã được xử lý

Frontend hiện đã có nhánh xử lý cho:

- `stream_ready`
- `heartbeat`
- `error`
- `attendance`
- `challenge_required`
- `challenge_active`
- `challenge_failed`
- các `scan_state` khác

### 10.3 Debug overlay

V4.4 có overlay kỹ thuật khá hoàn chỉnh:

- Vẽ `face bbox`
- Vẽ `moire ROI`
- Vẽ `screen context ROI`
- Vẽ `phone ROI`
- Vẽ polygon của rectangle tốt nhất
- Hiển thị panel:
  - status
  - match
  - moire
  - screen
  - phone
  - rolling

Đây là điểm mạnh lớn cho tuning và demo kỹ thuật, vì giúp quan sát detector đang quyết định dựa trên hình học nào.

## 11. Ngưỡng và tham số quan trọng

### 11.1 Threshold match

- `V4_COSINE_THRESHOLD = 0.52`

Ngưỡng này chặt hơn threshold mặc định `0.45`, giúp giảm nhận nhầm.

### 11.2 Threshold moire

- `MOIRE_SCREEN_THRESHOLD = 0.60`
- `MOIRE_BLOCK_THRESHOLD = 0.45`
- `MOIRE_SMOOTH_WINDOW = 7`
- `MOIRE_EVERY_N_DETECT = 3`

Ý nghĩa:

- Dưới `0.60`: bắt đầu nghi ngờ màn hình
- Dưới `0.45`: có thể block mạnh

### 11.3 Threshold screen context

- `SCREEN_CONTEXT_WEIGHT = 0.35`
- `FLATNESS_SUSPICIOUS_THRESHOLD = 0.65`
- `GLARE_SUSPICIOUS_THRESHOLD = 0.45`
- `SCREEN_CONTEXT_STRONG_THRESHOLD = 0.78`

### 11.4 Threshold phone rectangle

- `PHONE_RECT_CONTEXT_SCALE = 2.80`
- `PHONE_RECT_VERTICAL_RATIO = 1.6`
- `PHONE_RECT_SUSPICIOUS_THRESHOLD = 0.38`
- `PHONE_RECT_STRONG_THRESHOLD = 0.58`
- `PHONE_RECT_ROLLING_STRONG_COUNT = 2`

### 11.5 Threshold challenge

- `CHALLENGE_TIMEOUT = 7.0`
- `CHALLENGE_COOLDOWN = 12.0`
- `TURN_THRESHOLD = 0.06`

## 12. Event payload và khả năng debug

Một điểm tốt của V4.4 là payload event trả về rất giàu thông tin.

Payload chẩn đoán có thể bao gồm:

- `frame_size`
- `face_bbox`
- `moire_roi_bbox`
- `faces_detected`
- `liveness`
- `moire`
- `moire_rolling`
- `screen_context`
- `phone_rect`
- `phone_rect_rolling`
- `passive_liveness`
- `match`
- `scan_version`

Điều này cho phép:

- debug frontend overlay
- tuning threshold
- phân tích false positive và false negative
- hỗ trợ demo kỹ thuật và QA

## 13. Kiểm thử hiện tại

Project đã có file test riêng cho V4.4:

- `tests/test_detect_v4.py`

Các nhóm test hiện có:

- Moire detector với input rỗng
- Rolling moire dùng đúng ngưỡng suspicious
- Rolling phone rectangle block sau 2 mẫu strong
- Gom đúng `suspicious reasons`
- Diagnostics expose đúng geometry cho overlay
- System capability có khai báo V4
- Endpoint `/api/scan/v4` xử lý đúng khi không có session

Đã chạy xác minh:

- `pytest tests\test_detect_v4.py`

Kết quả:

- `7 passed`
- Có `1 warning` do không ghi được `.pytest_cache` vì quyền truy cập, không ảnh hưởng logic test

Đánh giá:

- Test lõi đang ổn
- Chưa thấy test end-to-end cho browser WebSocket hoặc local-direct runner
- Chưa thấy benchmark hoặc latency test cho runtime V4.4

## 14. Điểm mạnh của V4.4

### 14.1 Chống replay đa lớp

V4.4 không phụ thuộc duy nhất vào moire. Nó kết hợp:

- moire
- bối cảnh màn hình
- khung điện thoại
- passive liveness
- streaming liveness
- challenge

Điều này làm hệ thống khó bị qua mặt hơn bằng các kiểu replay thực tế.

### 14.2 Runtime độc lập transport

Thiết kế `transport-agnostic` trong `runtime_v4.py` là đúng hướng:

- cùng một logic cho browser stream và local direct
- giảm duplication
- dễ bảo trì

### 14.3 Hỗ trợ tuning tốt

Debug overlay và diagnostics payload giúp đội phát triển:

- quan sát detector
- tinh chỉnh ngưỡng
- so sánh hành vi giữa môi trường thật và giả lập

### 14.4 Tích hợp production khá sạch

V4.4 đã có:

- route riêng
- capability endpoint
- version endpoint
- frontend integration
- local runner
- test riêng

Đây là mức hoàn thiện cao hơn hẳn một script thử nghiệm.

## 15. Hạn chế và rủi ro hiện tại

### 15.1 Tham số tuning đang nằm cứng trong `core/detect_v4.py`

Nhiều threshold quan trọng của V4.4 đang nằm trực tiếp trong module thay vì đưa vào `config.py` hoặc env.

Tác động:

- khó tinh chỉnh theo môi trường triển khai
- khó A/B test
- khó cấu hình riêng cho các loại camera khác nhau

### 15.2 V4.4 vẫn dùng một số config của V3

Ví dụ:

- detect fps
- stream enable
- giới hạn kích thước JPEG
- passive liveness thresholds

Tác động:

- V4.4 chưa tách namespace cấu hình hoàn toàn
- chỉnh V3 có thể vô tình ảnh hưởng V4.4

### 15.3 HTTP `/api/scan/v4` chỉ phù hợp compatibility

Do V4.4 phụ thuộc rolling state và stream liveness, endpoint scan một ảnh không phản ánh đầy đủ chất lượng thật của V4.4.

Tác động:

- nếu tích hợp ngoài dùng nhầm endpoint này như flow chính, hiệu quả chống spoof sẽ giảm

### 15.4 Thiếu test end-to-end WebSocket

Hiện test chủ yếu là unit hoặc integration nhỏ.

Thiếu:

- Browser WebSocket flow hoàn chỉnh
- Local direct runner flow hoàn chỉnh
- Challenge lifecycle qua WebSocket
- Attendance sau challenge pass

### 15.5 Chi phí tính toán cao hơn V3

V4.4 chạy thêm:

- enhanced moire
- screen context
- phone rectangle
- rolling state

Tác động:

- CPU hoặc GPU load cao hơn
- cần theo dõi FPS và độ trễ trên phần cứng yếu

## 16. So sánh nhanh V3 và V4.4

### V3 mạnh ở

- Luồng scan production cơ bản ổn định
- Liveness stream, passive anti-spoof và challenge đơn giản
- Cấu trúc đã quen thuộc

### V4.4 bổ sung thêm

- Enhanced moire pipeline
- Screen context detector
- Phone rectangle detector
- Portrait phone ROI
- Debug overlay kỹ thuật mạnh hơn
- Runtime transport-agnostic cho cả browser và local

### Kết luận so sánh

Nếu mục tiêu là chống replay bằng điện thoại hoặc màn hình tốt hơn, V4.4 rõ ràng là bước nâng cấp thực chất so với V3.

## 17. Mức độ sẵn sàng triển khai

Đánh giá tổng quan:

- Mức tích hợp: tốt
- Mức hoàn thiện detector: tốt
- Mức observability: tốt
- Mức test: khá
- Mức configurability: trung bình

Kết luận:

V4.4 đủ điều kiện để dùng làm runtime quét chính trong môi trường cần anti-spoof mạnh hơn, với điều kiện:

- có theo dõi FPS thực tế trên máy triển khai
- có thêm test end-to-end cho WebSocket và challenge
- có kế hoạch tách cấu hình V4 ra riêng để tuning dễ hơn

## 18. Khuyến nghị kỹ thuật

### 18.1 Khuyến nghị ngắn hạn

1. Tách threshold V4.4 từ `core/detect_v4.py` sang `config.py`.
2. Bổ sung test WebSocket end-to-end cho:
   - `challenge_required`
   - `challenge_active`
   - `challenge_failed`
   - `attendance`
3. Ghi log hoặc tổng hợp thống kê cho:
   - tỷ lệ block
   - tỷ lệ challenge
   - tỷ lệ pass sau challenge
   - FPS trung bình

### 18.2 Khuyến nghị trung hạn

1. Tạo namespace config riêng cho V4.
2. Thêm calibration mode hoặc dashboard tuning nội bộ.
3. Bổ sung profile hiệu năng theo:
   - CPU only
   - GPU
   - camera độ phân giải thấp, trung bình, cao

## 19. Kết luận cuối

`Detect V4.4` là một bản nâng cấp anti-spoof có giá trị thực tế, không chỉ là tăng version. Điểm mạnh cốt lõi của phiên bản này là khả năng kết hợp nhiều bằng chứng cùng lúc để phát hiện replay bằng màn hình hoặc điện thoại, trong khi vẫn giữ trải nghiệm hợp lý bằng cơ chế challenge khi cần.

Ở trạng thái hiện tại, V4.4 đã được tích hợp đầy đủ xuyên suốt backend, API, local runner và frontend. Đây là nền tảng tốt để dùng trong production hoặc làm bản mặc định mới cho scan, miễn là tiếp tục đầu tư vào cấu hình hóa tham số và kiểm thử end-to-end.
