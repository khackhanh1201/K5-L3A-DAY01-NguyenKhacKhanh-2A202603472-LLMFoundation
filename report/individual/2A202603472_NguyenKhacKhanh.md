# Script để tạo file báo cáo Markdown
md_content = """# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603472  
- **Họ và tên:** Nguyễn Khắc Khánh
- **SEED** (in ở ô đầu notebook): 1179

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |
| [`hand_calc_2.jpg`](hand_calc_1.jpg) — ảnh tính tay có các bước | [x] |
## 2. Kết quả chính

SEED=1179 · thử thách #4
Bộ kiểm tra: 9/15 đạt
  ✅ 1.1 vectorize
  ✅ 1.2 sigmoid / predict_proba
  ✅ 1.3 train_logreg
  ✅ 1.4 metrics
  ✅ 2.1 build_mlp
  ✅ 2.2 train_mlp
  ✅ 3.1–3.2 cosine / top_k_similar
  ✅ 3.3 knn_predict
  ✅ 4.2–4.3 prompt / parse_label
  ❌ 4.1 + 4.4 call_openai / llm_classify_batch
  ❌ tính tay: baseline_accuracy
  ❌ tính tay: sigmoid_0
  ❌ tính tay: f1_example
  ❌ tính tay: cosine_example
  ❌ tính tay: llm_cost_per_day

Dự đoán đã ghi:
  [13:33:16] ml_f1: 0.7
  [13:33:33] ml_lr: lr=0.001 thì loss giảm cực kỳ chậm, đường cong lài lài chưa kịp hội tụ tới đáy; lr=50 thì loss sẽ dao động mạnh (zigzag nhảy múa) hoặc phân kỳ do bước nhảy quá lớn bị văng khỏi đáy.
  [13:34:11] dl_curve: Train loss sẽ giảm liên tục, nhưng Val loss sẽ giảm ở các epoch đầu rồi có xu hướng đi lên (hình chữ U) do hiện tượng overfitting
  [13:34:22] dl_relu: Mô hình tương đương với Logistic Regression. F1-score sẽ trở về mức tương đương với Logistic Regression ở Phần 1 do đã mất khả năng học các tương tác phi tuyến tính.
  [13:36:29] emb_search: Kết quả gần nhất sẽ là spam vì tuy không có từ khoá như free/win/prize, câu truy vấn mang ý nghĩa dụ dỗ, quà tặng, claim phần thưởng rất đặc trưng của spam và embedding sẽ nhận diện được sự tương đồng về nghĩa này.
  [13:40:14] emb_tricky: Mô hình Encoder + LogReg sẽ đoán đúng nhiều hơn. BoW dễ đoán sai vì chỉ nhìn vào từ khóa (thấy 'free', 'won', 'prize' là đoán spam dù ngữ cảnh vô hại) hoặc bỏ sót spam dùng từ đồng nghĩa ('complimentary', 'no cost'). Ngược lại, Embedding (Encoder) hiểu được toàn bộ ngữ cảnh và ý nghĩa thực sự của câu.
  [13:41:42] llm_models: 1) Model lớn sẽ có F1 cao hơn model nhỏ nhưng chênh lệch không nhiều (chỉ khoảng 2-5%) vì bài toán phân loại spam khá đơn giản. (2) Model lớn sẽ đắt gấp khoảng 15-30 lần so với model nhỏ. (3) Ở chế độ zero-shot, LLM có thể KHÔNG thắng được (hoặc chỉ ngang ngửa) mô hình 3b (Embedding + LogReg), vì mô hình 3b đã được huấn luyện (fine-tune) trực tiếp trên đặc trưng của chính tập dữ liệu này, trong khi LLM zero-shot rất dễ 'suy nghĩ quá đà' (overthink) và đánh giá nhầm các từ lóng/viết tắt.
  [13:42:37] llm_fewshot: F1 sẽ tăng lên vì mô hình được cung cấp ví dụ thực tế, giúp nó hiểu rõ hơn về văn phong lóng và cách dán nhãn của tập dữ liệu. Tuy nhiên, chi phí cho mỗi tin sẽ tăng lên gấp 3 đến 5 lần (hoặc hơn) do số lượng input token phình to để chứa 4 ví dụ này trong mỗi lần gọi API.
  [13:49:34] tradeoff: Trên đường Pareto sẽ là BoW + LogReg (nhanh nhất tuyệt đối), BoW + MLP (chậm hơn chút nhưng F1 tốt hơn), và Encoder + LogReg (chính xác nhất, F1=0.974, trễ 29.4ms). Đạt SLA 50ms sẽ gồm toàn bộ họ BoW và Encoder (trừ kNN). kNN và các API LLM chắc chắn nằm bên phải vạch 50ms do độ trễ cao.
  [14:25:08] hard_eval: (1) Accuracy của các model sẽ giảm mạnh so với tập test bình thường, có thể chỉ còn khoảng 50-70%. (2) Nhóm khó nhất với BoW là lách luật bằng từ đồng nghĩa/viết sai chính tả hoặc tin bình thường nhưng bàn luận về spam (vì BoW chỉ đếm từ độc lập). Nhóm khó nhất với LLM là Prompt Injection (tin nhắn chứa lệnh thao túng AI) vì LLM dễ bị nhầm lẫn giữa dữ liệu đầu vào và chỉ dẫn hệ thống.

                                     f1  f1_SUB
1. BoW + LogReg (tự code)        0.898   0.824
2. BoW + MLP (deep learning)     0.939   0.889
3a. Encoder (MiniLM) + kNN       0.940   0.974
3b. Encoder (MiniLM) + LogReg    0.966   0.974
3c. Encoder (MiniLM) + MLP head  0.938   0.974


## 3. Dự đoán và thực tế

- **Dự đoán ml_lr (Learning Rate 50):**
  - **Dự đoán:** Rằng với lr=50 đường loss sẽ dao động cực kỳ mạnh dạng zigzag nhấp nhô hoặc bị phân kỳ văng khỏi đáy (NaN) do bước nhảy quá lớn.
  - **Kết quả thực tế:** Đường loss chỉ giật mạnh đúng một nhịp (Spike) ở epoch đầu tiên, sau đó hội tụ rất nhanh xuống mức cực thấp (0.005).
  - **Nguyên nhân lệch:** Do bản chất hàm mất mát (Binary Cross Entropy) của Logistic Regression là một hàm lồi tuyệt đối (convex) dạng cái bát trơn tru, không có các bẫy cực tiểu cục bộ. Khi nhảy cực mạnh, mô hình vô tình ép các trọng số lên vô cùng lớn khiến các dự đoán xác suất sát nút 0 và 1, ép loss về sát 0 mà không bị sụp đổ tính toán như mạng nơ-ron sâu MLP.

- **Dự đoán hard_eval (Hiệu năng trên 50 case khó):**
  - **Dự đoán:** Các mô hình sẽ giữ được khoảng 50-70% độ chính xác trung bình.
  - **Kết quả thực tế:** Mô hình BoW tụt điểm thê thảm với F1 chỉ còn 0.368 (đoán đúng 52%), trong khi mô hình Embedding tốt nhất cũng chỉ có F1 0.759.
  - **Nguyên nhân lệch:** Do chưa lường hết được độ tinh vi của các ca bẫy (như lừa đảo dạng "Hi Mum" mạo danh người thân hoặc viết bóp méo chữ) hoàn toàn triệt tiêu sức mạnh thống kê tần suất từ vựng đơn giản của BoW.

- **Dự đoán emb_tricky (Các câu bẫy từ khóa):**
  - **Dự đoán:** Encoder + LogReg sẽ vượt trội hơn hẳn BoW trên mọi câu bẫy.
  - **Kết quả thực tế:** BoW đoán đúng 5/6 câu, trong khi Encoder đúng tuyệt đối 6/6 câu.
  - **Nguyên nhân lệch:** Sự chênh lệch trong thực tế không quá rộng ở một số câu bẫy đơn giản vì BoW đã tích lũy được bộ từ vựng tương đối tốt từ tập huấn luyện khổng lồ, chỉ chịu thua hoàn toàn ở các ca lách chữ tinh vi hơn.

## 4. Phân tích 50 case khó

- **Model tụt điểm nhiều nhất:** Mô hình Bag-of-Words (BoW + LogReg và BoW + MLP) chịu sự sụp đổ nghiêm trọng nhất. F1-score của BoW + LogReg tụt từ 0.824 (ở tập test thường SUB 40) xuống chỉ còn 0.368 trên 50 case khó.
- **Nhóm case gây lỗi nặng nhất:** Nhóm Spam giọng tự nhiên (natural_spam) gây thất bại 100% cho toàn bộ tất cả các mô hình trong bài lab (0% đoán đúng).

**2 Case cụ thể minh họa cơ chế lỗi:**
- **Case #12 (natural_spam):** "hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club"
  - **Cơ chế lỗi:** Tin nhắn này hoàn toàn sử dụng văn phong tâm sự đời thường, lịch sự, không hề chứa các "từ khóa rác" điển hình như FREE, WIN, CALL, claim. Bộ đếm từ vựng BoW coi đây hoàn toàn là tin nhắn Ham lành mạnh. Embedding cũng bị đánh lừa vì cấu trúc ngữ nghĩa rất gần với các lời hỏi thăm, làm quen thông thường.
- **Case #21 (manipulation):** "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"
  - **Cơ chế lỗi:** Đây là kịch bản lừa đảo mạo danh người thân cực kỳ nguy hiểm. Đối với các mô hình máy học truyền thống, các từ ngữ xuất hiện là các từ gia đình vô hại (Mum, friend, phone, toilet). Do đó, mô hình gán nhãn Ham. Chỉ có LLM có khả năng suy luận ngữ cảnh hành vi mới nhận diện được đây là một yêu cầu chuyển tiền bất thường gấp gáp.

## 5. Thử thách cá nhân

**Đề bài 4:** Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?

- **Kết quả:**
  - `k=1`  | F1-score: 0.952 (Precision: 0.935, Recall: 0.971)
  - `k=5`  | F1-score: 0.940 (Precision: 0.902, Recall: 0.981)
  - `k=25` | F1-score: 0.895 (Precision: 0.845, Recall: 0.951)

- **Giải thích:**
  - **Khi k quá nhỏ (k=1):** Mô hình cực kỳ nhạy cảm với các điểm nhiễu (noise) hoặc các mẫu bị gán nhãn sai trong tập huấn luyện. Ranh giới quyết định rất ngoằn ngoèo, dễ dẫn đến hiện tượng Overfitting.
  - **Khi k quá lớn (k=25):** Mô hình có xu hướng kéo cả những hàng xóm ở khoảng cách quá xa không còn tương đồng ngữ nghĩa vào biểu quyết. Do tập dữ liệu mất cân bằng nghiêm trọng (lớp Ham chiếm tới ~87%), lớp đa số (Ham) sẽ áp đảo phiếu bầu, triệt tiêu khả năng phát hiện các mẫu Spam thiểu số (lỗi Underfitting, Recall sụt giảm).

## 6. Phương án cho sản phẩm thật

Để hệ thống lọc tin nhắn đáp ứng yêu cầu xử lý 10 triệu SMS/ngày (~116 tin/giây trung bình) với ràng buộc độ trễ cực ngặt p95 < 50 ms/tin:

- **Phương án lựa chọn:** 3b. Encoder (MiniLM) + LogReg làm hạt nhân xử lý chính.
- **Dẫn chứng số liệu thực nghiệm:**
  - Đạt chất lượng tối ưu tuyệt đối trên tập test: F1 = 0.966.
  - Thời gian suy luận (inference) cực kỳ nhanh: p95 = 29.47 ms / tin (hoàn toàn vượt qua ngưỡng SLA 50 ms).
  - Thông lượng xử lý lô lớn đạt 40.4 tin/giây trên CPU thường (dễ dàng mở rộng chiều ngang - horizontal scaling để gánh tải 116 tin/s).
  - Chi phí vận hành gần như bằng 0 (không tốn tiền API).
- **Sự đánh đổi chấp nhận:** Chấp nhận bỏ qua các mô hình LLM lớn dù chúng có khả năng chống Prompt Injection tốt hơn, vì LLM có độ trễ p95 vượt quá hàng ngàn mili-giây và chi phí cực kỳ khổng lồ (vận hành 10M tin/ngày bằng mô hình nhỏ nhất cũng tốn hơn $15,000/tháng).

## 7. AI log và điều đã học

- **Phần 4 (LLM Phân loại):** Hỏi cách bọc dữ liệu đầu vào chống tấn công hướng dẫn.
- **Bài học:**
  - Độ chính xác (Accuracy) là một cái bẫy.
  - Tầm quan trọng của tiền xử lý.
  - Mô hình lai (Router Architecture) là chìa khóa.
