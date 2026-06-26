# Bộ câu hỏi và gợi ý trả lời bảo vệ tốt nghiệp - FarmLink AI Image Verification

Tài liệu này dùng để chuẩn bị phần vấn đáp khi bảo vệ đồ án tốt nghiệp. Nội dung tập trung vào hệ thống nhận diện ảnh nông sản trong FarmLink, pipeline AI hai giai đoạn, ResNet50 production v1, MobileNetV2 dùng để so sánh, dataset, training, deployment, testing và hướng phát triển.

## Thông tin nhanh cần nhớ

- Bài toán: xác minh ảnh người dùng upload có phải ảnh nông sản và nhận diện đúng loại nông sản hay không.
- Pipeline production: hai giai đoạn.
- Stage A: phân loại ảnh `fruit` và `other`.
- Stage B: phân loại loại nông sản trong nhóm supported classes.
- Model chính: ResNet50 pretrained ImageNet.
- Model so sánh: MobileNetV2.
- Dataset production v1: `dataset_fruit_supported_v1`.
- Số class production v1: 32 loại nông sản được hỗ trợ.
- Kích thước ảnh đầu vào: `320 x 320 x 3`.
- Stage B test accuracy tham khảo: `0.8089`.
- Stage B macro F1-score: `0.8023`.
- Stage B macro recall: `0.7939`.
- Stage A accuracy: `0.9942`.
- Stage A macro F1-score: `0.9848`.
- Deployment: website gọi Supabase Edge Function, sau đó gọi AI API để dự đoán.
- Khi model không chắc chắn: đưa vào manual review hoặc lớp kiểm tra bổ sung, không ép model trả kết quả.

---

# 1. Câu hỏi tổng quan

## 1. Đề tài của em giải quyết vấn đề gì?

Đề tài giải quyết bài toán xây dựng hệ thống FarmLink hỗ trợ người dùng đăng và kiểm tra sản phẩm nông sản. Trong đó, chức năng AI được dùng để xác minh ảnh người dùng upload có đúng là ảnh nông sản hay không và nếu đúng thì nhận diện loại nông sản tương ứng.

Mục tiêu thực tế là giảm tình trạng người dùng đăng ảnh sai nội dung, ảnh không liên quan hoặc ảnh không đúng loại sản phẩm. Nhờ đó hệ thống có thể hỗ trợ kiểm duyệt tự động một phần trước khi sản phẩm được hiển thị trên website.

## 2. Vì sao website FarmLink cần chức năng nhận diện ảnh nông sản?

FarmLink là hệ thống liên quan đến giao dịch hoặc đăng tải sản phẩm nông sản, vì vậy ảnh sản phẩm là thông tin quan trọng. Nếu người dùng upload ảnh không đúng loại nông sản, ảnh không phải sản phẩm hoặc ảnh không liên quan thì độ tin cậy của hệ thống sẽ giảm.

Chức năng nhận diện ảnh giúp kiểm tra tự động ảnh đầu vào. Hệ thống có thể cảnh báo, chặn, hoặc đưa vào manual review khi ảnh không phù hợp hoặc model chưa đủ tự tin.

## 3. AI trong hệ thống đóng vai trò gì?

AI đóng vai trò là lớp kiểm tra ảnh sản phẩm. Khi người dùng upload ảnh, AI sẽ phân tích ảnh để xác định ảnh có phải nông sản hay không, sau đó phân loại loại nông sản nếu ảnh thuộc nhóm được hệ thống hỗ trợ.

AI không thay thế hoàn toàn kiểm duyệt con người. Với các trường hợp model không chắc chắn, ảnh thuộc class chưa hỗ trợ hoặc ảnh có dấu hiệu bất thường, hệ thống sẽ đưa vào manual review.

## 4. Người dùng upload ảnh thì hệ thống xử lý như thế nào?

Quy trình tổng quát là: người dùng upload ảnh trên website, frontend gửi ảnh đến backend hoặc Supabase Edge Function, sau đó hệ thống gọi AI API. AI API tiền xử lý ảnh, đưa ảnh qua Stage A để kiểm tra ảnh có phải nông sản hay không, nếu là nông sản thì đưa tiếp qua Stage B để phân loại loại nông sản.

Kết quả trả về gồm nhãn dự đoán, độ tin cậy, trạng thái xử lý và lý do. Nếu model đủ tự tin, hệ thống có thể auto accept. Nếu model không chắc chắn, ảnh được đưa vào manual review.

## 5. Vì sao cần kiểm tra ảnh nông sản trước khi đăng sản phẩm?

Việc kiểm tra ảnh giúp đảm bảo ảnh sản phẩm phù hợp với nội dung người bán đăng tải. Điều này làm tăng độ tin cậy của dữ liệu trên hệ thống, giảm rủi ro spam, giảm ảnh không liên quan và hỗ trợ quản trị viên kiểm duyệt nhanh hơn.

Trong hệ thống thương mại hoặc kết nối nông sản, ảnh sản phẩm ảnh hưởng trực tiếp đến trải nghiệm người mua. Vì vậy việc xác minh ảnh là một chức năng quan trọng.

## 6. Hệ thống hiện tại nhận diện được bao nhiêu loại nông sản?

Phiên bản production v1 hiện tại nhận diện 32 loại nông sản được hỗ trợ. Đây là tập class đã được lọc từ dataset sau khi đánh giá chất lượng dữ liệu và kết quả model.

Một số class yếu hoặc chưa đủ tin cậy được loại khỏi production v1 và đưa vào nhóm unsupported/manual review để tránh hệ thống tự động dự đoán sai trong môi trường thực tế.

## 7. Vì sao chỉ chọn 32 class production v1 mà không dùng toàn bộ class?

Không phải tất cả class đều có chất lượng nhận diện đủ tốt. Một số class có dữ liệu ít, ảnh không đồng nhất, hình dạng dễ nhầm lẫn hoặc metric như recall/F1-score thấp. Nếu đưa toàn bộ class vào production, hệ thống dễ dự đoán sai và làm giảm độ tin cậy.

Vì vậy em chọn hướng production v1 chỉ giữ các class đủ ổn định để đưa vào hệ thống. Các class yếu được đưa vào manual review và có thể cải thiện ở phiên bản sau khi bổ sung dữ liệu tốt hơn.

## 8. Những class không hỗ trợ thì hệ thống xử lý thế nào?

Các class chưa hỗ trợ không được đưa vào Stage B production v1. Nếu ảnh upload có nhãn thuộc nhóm unsupported trong quá trình test hoặc hệ thống nhận diện được trường hợp không thuộc phạm vi hỗ trợ, ảnh sẽ được đưa vào manual review thay vì ép model dự đoán thành một class supported.

Cách này giúp tránh lỗi nghiêm trọng là nhận nhầm một sản phẩm chưa hỗ trợ thành một sản phẩm khác đang được hỗ trợ.

## 9. Điểm khác biệt giữa hệ thống của em và upload ảnh thông thường là gì?

Upload ảnh thông thường chỉ lưu ảnh lên hệ thống mà không kiểm tra nội dung ảnh. Còn trong FarmLink, ảnh được đưa qua pipeline AI để kiểm tra tính phù hợp và phân loại nông sản.

Nhờ đó hệ thống có thêm một lớp xác minh tự động. Đây là điểm khác biệt quan trọng vì nó giúp tăng chất lượng dữ liệu sản phẩm và hỗ trợ quy trình kiểm duyệt.

---

# 2. Câu hỏi về dataset

## 1. Dataset lấy từ đâu?

Dataset được xây dựng từ dữ liệu ảnh nông sản lưu trong thư mục `dataset/raw`. Đây là dữ liệu gốc, gồm các thư mục tương ứng với từng class nông sản và class `other`.

Trong quá trình làm đồ án, dữ liệu raw không được sửa trực tiếp trong pipeline xử lý. Các bước xử lý như clean, crop, split được thực hiện ở các thư mục output riêng để đảm bảo có thể truy vết và rebuild dataset khi cần.

## 2. Dataset ban đầu có bao nhiêu class?

Dataset chính sau khi chuẩn hóa taxonomy có 41 class, bao gồm 40 class nông sản và 1 class `other`. Class `other` được dùng cho Stage A để giúp model phân biệt ảnh nông sản và ảnh không phải nông sản.

Với Stage B production v1, hệ thống chỉ giữ 32 class nông sản được hỗ trợ để train model phân loại chi tiết.

## 3. Vì sao có class `other`?

Class `other` đại diện cho ảnh không phải nông sản hoặc ảnh ngoài phạm vi hệ thống. Class này rất quan trọng vì trong thực tế người dùng có thể upload ảnh sai, ảnh không liên quan, ảnh nền, ảnh vật thể khác hoặc ảnh không đọc được nội dung sản phẩm.

Nếu không có class `other`, model có xu hướng ép mọi ảnh đầu vào thành một loại nông sản nào đó, dẫn đến sai nghiêm trọng trong production.

## 4. Class `other` dùng để làm gì?

Class `other` chỉ dùng trong Stage A để phân loại nhị phân: ảnh có phải nông sản hay không. Nếu ảnh bị xác định là `other`, hệ thống không đưa ảnh vào Stage B.

Trong Stage B, class `other` bị loại bỏ hoàn toàn vì Stage B chỉ có nhiệm vụ phân loại giữa các loại nông sản đã được xác định là hợp lệ.

## 5. Vì sao Stage B không có class `other`?

Stage B được thiết kế là model multi-class phân loại loại nông sản cụ thể. Input của Stage B giả định đã đi qua Stage A và đã được xác định là ảnh nông sản.

Nếu đưa `other` vào Stage B, bài toán sẽ bị lẫn giữa phân loại nông sản và phát hiện ảnh ngoài phạm vi. Điều này làm model khó học hơn và dễ gây nhầm lẫn. Vì vậy hệ thống tách rõ: Stage A xử lý `fruit vs other`, Stage B xử lý `fruit class classification`.

## 6. Dataset đã được làm sạch như thế nào?

Dataset được làm sạch qua pipeline preprocess. Mỗi ảnh được mở bằng PIL, sửa EXIF orientation, chuyển về RGB, resize/pad về kích thước chuẩn, loại bỏ ảnh corrupt, ảnh quá nhỏ, ảnh quá mờ và ảnh trùng lặp.

Sau bước clean, ảnh được lưu vào `dataset/processed_clean`. Bước này giúp giảm nhiễu dữ liệu và tránh model học từ ảnh lỗi hoặc ảnh không có giá trị.

## 7. Ảnh lỗi, ảnh mờ, ảnh trùng được xử lý ra sao?

Ảnh lỗi hoặc không đọc được sẽ bị loại bỏ. Ảnh quá nhỏ hoặc quá mờ cũng bị loại vì không cung cấp đủ thông tin hình ảnh cho model học. Ảnh trùng được phát hiện bằng perceptual hash để tránh cùng một ảnh xuất hiện nhiều lần sau khi gộp class.

Việc loại ảnh trùng và ảnh lỗi giúp model học ổn định hơn, đồng thời giảm nguy cơ đánh giá sai do dữ liệu lặp lại.

## 8. Vì sao phải resize/pad ảnh về 320x320?

Model deep learning cần input có kích thước cố định. Trong project này, ảnh được chuẩn hóa về `320 x 320 x 3` để phù hợp với cấu hình ResNet50.

Kích thước 320 giúp giữ nhiều chi tiết hơn so với kích thước quá nhỏ như 224, trong khi vẫn không quá nặng để train và inference. Đây là lựa chọn cân bằng giữa chất lượng ảnh và tài nguyên tính toán.

## 9. Resize/pad khác gì resize kéo méo ảnh?

Resize kéo méo ảnh sẽ ép ảnh về kích thước cố định mà không giữ tỉ lệ gốc, làm biến dạng hình dạng quả hoặc nông sản. Điều này có thể làm model học sai đặc trưng hình dạng.

Resize/pad giữ nguyên tỉ lệ ảnh, sau đó thêm padding để đủ kích thước chuẩn. Cách này bảo toàn hình dạng tổng thể của nông sản tốt hơn.

## 10. Vì sao phải chia train/validation/test?

Train set dùng để model học, validation set dùng để theo dõi quá trình training và chọn checkpoint tốt, test set dùng để đánh giá cuối cùng sau khi train xong.

Nếu không chia dữ liệu, ta không biết model có thật sự học được đặc trưng tổng quát hay chỉ ghi nhớ dữ liệu train. Việc chia train/val/test giúp đánh giá khách quan hơn.

## 11. Tỉ lệ chia dữ liệu là bao nhiêu?

Dataset được split theo tỉ lệ khoảng `70% train`, `15% validation`, `15% test` theo từng class. Cách chia theo từng class giúp mỗi class đều có ảnh trong cả ba tập.

Trong production v1, `dataset_fruit_supported_v1` có 20,814 ảnh train, 4,467 ảnh validation và 4,454 ảnh test.

## 12. Làm sao đảm bảo không bị data leakage?

Pipeline split kiểm tra để cùng một file không xuất hiện đồng thời trong nhiều split. Ngoài ra, dữ liệu được chia sau khi preprocess/crop và giữ seed cố định để kết quả split ổn định.

Data leakage rất nguy hiểm vì nếu ảnh giống nhau hoặc cùng file xuất hiện ở cả train và test, kết quả test sẽ cao giả tạo. Vì vậy project có validation kiểm tra overlap giữa train/val/test.

## 13. Vì sao phải gộp taxonomy như `jackfruit_cempedak`, `mulberry`, `gourd`?

Một số class trong dữ liệu có hình ảnh rất giống nhau hoặc ranh giới phân biệt không rõ ràng. Ví dụ `black_mulberry` và `red_mulberry` đều thuộc nhóm mulberry; `cempedak` và `jackfruit` có ngoại hình gần nhau; `bitter_gourd` và `ridged_gourd` cùng nhóm gourd.

Việc gộp taxonomy giúp giảm nhiễu nhãn, làm bài toán thực tế hơn và giúp model học ổn định hơn. Đây là quyết định dựa trên chất lượng dữ liệu và khả năng phân biệt bằng ảnh.

## 14. Những class yếu bị loại khỏi production v1 dựa trên tiêu chí nào?

Các class yếu được xác định dựa trên kết quả đánh giá như recall thấp, F1-score thấp, nhầm lẫn nhiều trong confusion matrix hoặc dữ liệu không đủ ổn định. Những class này không bị xóa khỏi project mà được đưa vào nhóm unsupported/manual review.

Mục tiêu là production v1 ưu tiên độ tin cậy. Các class yếu có thể được cải thiện ở phiên bản sau bằng cách bổ sung dữ liệu thật, làm sạch dữ liệu hoặc thiết kế pipeline riêng.

## 15. Dataset production v1 có bao nhiêu ảnh train/val/test?

Dataset production v1 là `dataset_fruit_supported_v1`, gồm 32 class nông sản được hỗ trợ. Số lượng ảnh là:

- Train: 20,814 ảnh.
- Validation: 4,467 ảnh.
- Test: 4,454 ảnh.
- Tổng: 29,735 ảnh.

Đây là dataset đã qua xử lý clean/crop và đã loại class `other` cùng các class unsupported.

---

# 3. Câu hỏi về pipeline AI

## 1. Vì sao hệ thống dùng pipeline 2-stage?

Pipeline 2-stage giúp tách bài toán thành hai phần rõ ràng. Stage A kiểm tra ảnh có phải nông sản hay không, Stage B chỉ phân loại loại nông sản nếu ảnh đã được xác nhận là nông sản.

Cách thiết kế này giảm rủi ro model phân loại nhầm ảnh ngoài phạm vi thành một loại nông sản. Nó cũng giúp hệ thống dễ kiểm soát threshold và manual review hơn.

## 2. Stage A làm nhiệm vụ gì?

Stage A làm nhiệm vụ phân loại nhị phân giữa `fruit` và `other`. Nếu ảnh là nông sản, ảnh được chuyển tiếp sang Stage B. Nếu ảnh không phải nông sản, hệ thống chặn hoặc đưa vào trạng thái `stage_a_other`.

Stage A là lớp lọc đầu vào, giúp bảo vệ Stage B khỏi các ảnh không thuộc phạm vi bài toán.

## 3. Stage B làm nhiệm vụ gì?

Stage B làm nhiệm vụ phân loại ảnh nông sản thành một trong các class supported production v1. Stage B không xử lý ảnh `other` và không xử lý các class unsupported.

Kết quả Stage B gồm class dự đoán, confidence top-1, top-2 và margin để quyết định auto accept hay manual review.

## 4. Vì sao không train một model duy nhất phân loại tất cả class?

Nếu train một model duy nhất gồm cả `other` và tất cả nông sản, model phải giải quyết đồng thời hai bài toán: phát hiện ảnh ngoài phạm vi và phân loại chi tiết nông sản. Điều này làm bài toán khó hơn và khó kiểm soát lỗi.

Pipeline 2-stage dễ giải thích, dễ debug và phù hợp với production hơn. Stage A chịu trách nhiệm loại ảnh không phù hợp, Stage B tập trung học đặc trưng giữa các loại nông sản.

## 5. Nếu ảnh không phải nông sản thì đi qua flow nào?

Ảnh sẽ đi qua Stage A. Nếu xác suất fruit rất thấp, hệ thống route ảnh sang `stage_a_other`, tức là ảnh được xem là không phải nông sản.

Nếu Stage A không đủ chắc chắn, ảnh không bị kết luận ngay là sai mà được đưa vào `manual_review_stage_a_uncertain` để tránh chặn nhầm ảnh nông sản thật.

## 6. Nếu ảnh là nông sản nhưng thuộc class chưa hỗ trợ thì xử lý thế nào?

Với các class chưa hỗ trợ trong production v1, hệ thống không đưa vào Stage B để ép dự đoán thành class supported. Thay vào đó, ảnh được đưa vào `manual_review_unsupported`.

Cách xử lý này giúp đảm bảo hệ thống không trả kết quả sai một cách tự tin cho những loại nông sản chưa được hỗ trợ.

## 7. Nếu model không chắc chắn thì xử lý thế nào?

Nếu Stage A hoặc Stage B không đủ tự tin, hệ thống đưa ảnh vào manual review. Với Stage B, hai điều kiện được dùng là confidence top-1 và khoảng cách giữa top-1 với top-2.

Nếu top-1 confidence thấp hoặc top-1 và top-2 quá gần nhau, model có thể đang phân vân giữa hai class. Khi đó không nên auto accept.

## 8. Manual review có vai trò gì trong hệ thống?

Manual review là cơ chế an toàn để xử lý các trường hợp model không chắc chắn, ảnh khó, ảnh class chưa hỗ trợ hoặc ảnh ngoài phạm vi. Nó giúp giảm rủi ro dự đoán sai trong production.

Trong thực tế, một hệ thống AI tốt không nhất thiết phải tự động quyết định mọi trường hợp. Hệ thống cần biết khi nào nên từ chối hoặc chuyển cho con người kiểm tra.

## 9. Vì sao cần threshold thay vì luôn lấy kết quả top-1?

Softmax luôn trả ra một class có xác suất cao nhất, kể cả khi ảnh không rõ hoặc không thuộc class nào. Nếu luôn lấy top-1, hệ thống sẽ dễ trả kết quả sai nhưng có vẻ hợp lệ.

Threshold giúp kiểm soát độ tin cậy. Chỉ khi confidence đủ cao và margin đủ rõ, hệ thống mới auto accept. Nếu không, ảnh được đưa vào manual review.

## 10. Flow thực tế từ frontend đến AI API diễn ra như thế nào?

Frontend của FarmLink gửi ảnh upload đến Supabase Edge Function hoặc backend trung gian. Edge Function gọi AI API, AI API load model đã train và thực hiện preprocess, Stage A, Stage B, threshold logic.

Kết quả được trả về frontend gồm trạng thái xử lý, class dự đoán, confidence và lý do. Frontend dựa vào kết quả này để hiển thị thông báo hoặc cho phép tiếp tục quy trình đăng sản phẩm.

---

# 4. Câu hỏi về model

## 1. Vì sao chọn ResNet50?

ResNet50 là kiến trúc CNN mạnh, phổ biến và đã được kiểm chứng trong nhiều bài toán phân loại ảnh. Điểm mạnh của ResNet50 là residual connection, giúp train mạng sâu ổn định hơn và học được đặc trưng ảnh tốt.

Trong project này, ResNet50 cho kết quả tốt hơn MobileNetV2 ở macro F1-score và accuracy nên được chọn làm model chính cho production v1.

## 2. ResNet50 có ưu điểm gì?

ResNet50 có khả năng trích xuất đặc trưng ảnh tốt nhờ mạng sâu và residual block. Nó cũng có pretrained weights trên ImageNet, giúp tận dụng kiến thức thị giác đã học từ tập dữ liệu lớn.

So với một model tự xây từ đầu, ResNet50 giúp tiết kiệm thời gian train, cần ít dữ liệu hơn và thường ổn định hơn.

## 3. ResNet50 khác MobileNetV2 ở điểm nào?

MobileNetV2 nhẹ hơn, nhanh hơn và phù hợp hơn với thiết bị tài nguyên thấp. ResNet50 nặng hơn nhưng thường có khả năng biểu diễn đặc trưng mạnh hơn trong bài toán phân loại ảnh.

Trong project này, MobileNetV2 được dùng làm baseline để so sánh. Kết quả cho thấy ResNet50 đạt macro F1-score cao hơn nên được chọn làm model chính.

## 4. Vì sao vẫn train MobileNetV2 nếu cuối cùng chọn ResNet50?

Việc train MobileNetV2 giúp có baseline để so sánh. Nếu chỉ train một model, ta khó chứng minh lựa chọn model là hợp lý.

Khi báo cáo, có thể trình bày rằng nhóm đã thử nghiệm MobileNetV2 và ResNet50, sau đó chọn ResNet50 vì kết quả tốt hơn trên các metric quan trọng như macro F1-score và recall theo class.

## 5. Kết quả ResNet50 tốt hơn MobileNetV2 ở metric nào?

Trong báo cáo so sánh, ResNet50 production v1 đạt accuracy khoảng `0.8089` và macro F1-score khoảng `0.8023`. MobileNetV2 baseline cũ có accuracy khoảng `0.7503` và macro F1-score khoảng `0.7113`.

Điều này cho thấy ResNet50 tốt hơn ở cả accuracy và macro F1-score. Tuy nhiên cần ghi chú rằng hai model có thể được train trên phiên bản taxonomy/dataset khác nhau, nên so sánh này dùng để tham khảo định hướng.

## 6. Vì sao dùng transfer learning?

Transfer learning tận dụng model đã được học trước trên ImageNet. Model đã biết các đặc trưng thị giác cơ bản như cạnh, màu, texture, hình dạng, vật thể.

Nhờ đó, project không cần train model từ đầu. Với dataset nông sản có quy mô vừa phải, transfer learning giúp model hội tụ nhanh hơn và thường đạt kết quả tốt hơn.

## 7. Pretrained ImageNet giúp ích gì?

ImageNet là tập dữ liệu ảnh lớn với nhiều loại vật thể. Khi ResNet50 được pretrained trên ImageNet, các layer đầu và giữa đã học được đặc trưng thị giác phổ quát.

Trong bài toán nông sản, những đặc trưng như màu sắc, texture vỏ, hình dạng, cạnh và vùng vật thể đều hữu ích. Fine-tuning giúp model điều chỉnh các đặc trưng đó cho domain nông sản.

## 8. Classifier head của model gồm những lớp nào?

Classifier head của Stage B gồm:

- GlobalAveragePooling.
- BatchNormalization.
- Dense(512, activation ReLU).
- Dropout(0.5).
- Dense(num_classes, activation Softmax).

Backbone ResNet50 dùng để trích xuất đặc trưng, còn classifier head dùng để phân loại đặc trưng đó thành từng class nông sản.

## 9. GlobalAveragePooling dùng để làm gì?

GlobalAveragePooling chuyển feature map từ backbone thành vector đặc trưng gọn hơn. Nó giảm số lượng tham số so với Flatten và giúp model ít bị overfit hơn.

Trong transfer learning, GlobalAveragePooling là lựa chọn phổ biến để nối backbone CNN với classifier head.

## 10. Dropout dùng để làm gì?

Dropout tắt ngẫu nhiên một phần neuron trong quá trình train. Điều này giúp model không phụ thuộc quá nhiều vào một số đặc trưng cụ thể và giảm overfitting.

Trong project này, Dropout(0.5) được dùng trong classifier head của Stage B để tăng khả năng tổng quát hóa.

## 11. BatchNormalization có tác dụng gì?

BatchNormalization giúp ổn định phân phối đầu vào của các layer tiếp theo, hỗ trợ quá trình train ổn định hơn. Trong classifier head, nó giúp vector đặc trưng sau GlobalAveragePooling được chuẩn hóa trước khi đi qua Dense.

Khi fine-tune ResNet50, các BatchNorm layer trong backbone được giữ frozen để tránh làm sai lệch thống kê đã học từ pretrained model, đặc biệt khi batch size không lớn.

## 12. Softmax dùng để làm gì trong Stage B?

Softmax chuyển output của model thành phân phối xác suất trên các class nông sản. Tổng xác suất các class bằng 1, class có xác suất cao nhất là dự đoán top-1.

Tuy nhiên, softmax confidence không phải lúc nào cũng phản ánh chắc chắn tuyệt đối. Vì vậy hệ thống vẫn dùng threshold và margin để quyết định auto accept hay manual review.

## 13. Sigmoid dùng để làm gì trong Stage A?

Stage A là bài toán nhị phân `fruit vs other`, nên sigmoid được dùng để trả về xác suất ảnh thuộc nhóm fruit. Nếu fruit probability cao, ảnh được chuyển sang Stage B.

Nếu fruit probability thấp, ảnh được xem là other hoặc đưa vào manual review nếu nằm trong vùng không chắc chắn.

## 14. Vì sao không dùng YOLO/detector phức tạp?

YOLO là model object detection, phù hợp khi cần xác định vị trí vật thể bằng bounding box. Trong project này, mục tiêu chính là xác minh và phân loại ảnh sản phẩm, không phải phát hiện nhiều vật thể trong ảnh.

Thêm detector sẽ làm pipeline phức tạp hơn, cần dữ liệu bounding box và tăng chi phí triển khai. Vì vậy project hiện tại dùng smart crop nhẹ và classification pipeline trước, phù hợp với phạm vi đồ án.

## 15. Vì sao không dùng ViT/EfficientNet?

ViT hoặc EfficientNet có thể cho kết quả tốt, nhưng phạm vi đồ án cần ưu tiên pipeline ổn định, dễ giải thích và dễ triển khai. ResNet50 là kiến trúc kinh điển, tài liệu nhiều, dễ trình bày với hội đồng và phù hợp với transfer learning.

Ngoài ra, project đã có so sánh với MobileNetV2 và ResNet50. ResNet50 đạt kết quả đủ tốt để tích hợp vào hệ thống nên được chọn làm baseline production v1.

---

# 5. Câu hỏi về training

## 1. Vì sao train Stage B thành 2 giai đoạn?

Stage B được train theo hai giai đoạn để ổn định quá trình học. Giai đoạn 1 đóng băng backbone ResNet50 và chỉ train classifier head. Giai đoạn 2 mở một phần các layer cuối của backbone để fine-tune theo dữ liệu nông sản.

Cách này giúp model không phá vỡ pretrained weights ngay từ đầu, đồng thời vẫn có khả năng thích nghi với domain nông sản ở giai đoạn sau.

## 2. Stage 1 freeze backbone để làm gì?

Freeze backbone nghĩa là giữ nguyên trọng số ResNet50 pretrained và chỉ cập nhật classifier head. Giai đoạn này giúp classifier head học cách ánh xạ đặc trưng pretrained sang các class nông sản.

Nếu unfreeze toàn bộ model ngay từ đầu, gradient có thể làm thay đổi mạnh pretrained weights, gây training không ổn định, đặc biệt khi dataset không quá lớn.

## 3. Stage 2 fine-tune để làm gì?

Fine-tune giúp một số layer cuối của ResNet50 được điều chỉnh theo đặc trưng riêng của ảnh nông sản. Ví dụ, model có thể học tốt hơn texture vỏ, màu sắc, hình dạng và đặc điểm phân biệt giữa các loại quả.

Stage 2 thường dùng learning rate thấp để cập nhật nhẹ nhàng, tránh làm hỏng kiến thức pretrained.

## 4. Vì sao learning rate Stage 1 lớn hơn Stage 2?

Ở Stage 1, chỉ classifier head mới được train, nên có thể dùng learning rate lớn hơn như `3e-4`. Các layer này mới khởi tạo nên cần học nhanh hơn.

Ở Stage 2, backbone pretrained được fine-tune nên phải dùng learning rate thấp hơn như `1e-5` để tránh thay đổi quá mạnh các trọng số đã học.

## 5. Vì sao BatchNorm trong ResNet50 được giữ frozen khi fine-tune?

BatchNorm lưu thống kê trung bình và phương sai của dữ liệu. Nếu batch size nhỏ hoặc dữ liệu không đủ lớn, cập nhật BatchNorm trong backbone có thể làm thống kê bị lệch và gây giảm hiệu năng.

Vì vậy khi fine-tune ResNet50, các BatchNorm layer trong backbone được giữ frozen để training ổn định hơn.

## 6. Vì sao batch size chọn 16?

Batch size 16 là lựa chọn cân bằng giữa độ ổn định và giới hạn tài nguyên máy, đặc biệt khi train trên máy cá nhân hoặc Mac/Metal GPU. Batch size quá lớn có thể gây thiếu bộ nhớ, batch size quá nhỏ có thể làm training nhiễu hơn.

Với ảnh 320x320 và ResNet50, batch size 16 là lựa chọn hợp lý cho baseline production v1.

## 7. Vì sao dùng SparseCategoricalCrossentropy?

Stage B dùng label dạng integer, ví dụ class 0, 1, 2,..., 31. Với label integer, loss phù hợp là `SparseCategoricalCrossentropy`.

Nếu dùng one-hot label thì có thể dùng `CategoricalCrossentropy`, nhưng project đã chuẩn hóa Stage B theo label integer để pipeline đơn giản và tránh sai lệch label/loss.

## 8. Vì sao không dùng one-hot label trong Stage B?

Label integer giúp dataloader đơn giản hơn và giảm rủi ro mismatch giữa label shape và loss. Với nhiều class, integer label kết hợp `SparseCategoricalCrossentropy` là lựa chọn phổ biến và hiệu quả.

Điều quan trọng là phải đảm bảo label nằm trong range `[0, num_classes - 1]`. Project có sanity check để phát hiện lỗi label trước khi train.

## 9. Augmentation dùng những kỹ thuật nào?

Stage B production v1 dùng augmentation nhẹ:

- RandomFlip horizontal.
- RandomRotation 0.05.
- RandomZoom 0.08.

Các augmentation này giúp model quen với thay đổi nhỏ về hướng, góc chụp và tỷ lệ ảnh mà không làm biến dạng quá mạnh đặc trưng nông sản.

## 10. Vì sao chỉ dùng augmentation nhẹ?

Bài toán phân loại nông sản là fine-grained classification, tức là nhiều class có đặc trưng khá giống nhau. Nếu augmentation quá mạnh, ví dụ thay đổi màu sắc hoặc contrast quá nhiều, model có thể học sai đặc trưng.

Vì vậy baseline production v1 chỉ dùng augmentation nhẹ để tăng tính tổng quát nhưng vẫn giữ đặc điểm tự nhiên của ảnh.

## 11. Vì sao không dùng augmentation màu mạnh?

Màu sắc là đặc trưng quan trọng của nhiều loại nông sản, ví dụ dâu, chuối, cà chua, sầu riêng, dưa hấu. Nếu thay đổi màu quá mạnh, ảnh train có thể không còn giống thực tế.

Do đó bản production v1 không dùng augmentation màu mạnh để tránh làm nhiễu dữ liệu và làm giảm khả năng phân biệt class.

## 12. Class weight có dùng không? Vì sao production v1 mặc định không dùng?

Production v1 mặc định không dùng class weight để tránh model over-predict các class được tăng trọng số. Class weight có thể hữu ích khi dữ liệu mất cân bằng, nhưng nếu dùng không cẩn thận có thể làm precision của một số class giảm mạnh.

Script vẫn cho phép bật class weight bằng CLI và clip trọng số trong khoảng hợp lý, nhưng baseline chính ưu tiên sự ổn định và dễ so sánh.

## 13. Làm sao biết model không bị overfit?

Có thể quan sát train/validation loss và accuracy qua `history.json` hoặc `train.log`. Nếu train accuracy tăng cao nhưng validation accuracy thấp hoặc validation loss tăng liên tục, đó là dấu hiệu overfit.

Trong project, model dùng validation set, early stopping/checkpoint và đánh giá cuối cùng trên test set. Điều này giúp kiểm tra khả năng tổng quát của model.

## 14. Dựa vào log nào để biết training ổn định?

Các log quan trọng gồm:

- `train.log`: quá trình train, config, sanity check, stage training.
- `history.json`: lịch sử loss/metric qua từng epoch.
- `classification_report.txt`: precision/recall/F1 từng class.
- `confusion_matrix.png`: ma trận nhầm lẫn.
- `test_results.txt`: metric tổng thể trên test set.

Các file này nằm trong thư mục experiment của từng lần train.

## 15. Model cuối cùng được lưu ở đâu?

Model Stage B production v1 được lưu trong thư mục experiment:

`experiments/stage_b_supported_v1_resnet50_20260602_223530/model.keras`

Ngoài model, thư mục này còn lưu `labels.json`, `config.json`, `train.log`, `history.json`, `classification_report.txt`, `confusion_matrix.png` và các file prediction phục vụ đánh giá.

---

# 6. Câu hỏi về metrics

## 1. Accuracy là gì?

Accuracy là tỷ lệ dự đoán đúng trên tổng số mẫu. Ví dụ nếu có 100 ảnh và model dự đoán đúng 80 ảnh thì accuracy là 80%.

Accuracy dễ hiểu nhưng không đủ trong bài toán nhiều class, đặc biệt khi dataset mất cân bằng hoặc một số class yếu bị che bởi kết quả tốt của class lớn.

## 2. Precision là gì?

Precision cho biết trong số các ảnh model dự đoán là một class, có bao nhiêu ảnh thật sự đúng class đó. Công thức đơn giản là `TP / (TP + FP)`.

Precision thấp nghĩa là model hay đoán nhầm ảnh class khác thành class này. Ví dụ precision của `pomelo` thấp nghĩa là nhiều ảnh không phải pomelo bị model dự đoán thành pomelo.

## 3. Recall là gì?

Recall cho biết trong số ảnh thật sự thuộc một class, model nhận diện đúng được bao nhiêu ảnh. Công thức đơn giản là `TP / (TP + FN)`.

Recall thấp nghĩa là model bỏ sót nhiều ảnh thật của class đó. Ví dụ recall của `guava` thấp nghĩa là nhiều ảnh guava thật bị dự đoán sang class khác.

## 4. F1-score là gì?

F1-score là trung bình điều hòa giữa precision và recall. Nó cân bằng giữa việc model dự đoán đúng khi đã dự đoán một class và việc model không bỏ sót class đó.

F1-score hữu ích khi cần đánh giá tổng quát từng class, đặc biệt trong bài toán nhiều class có dữ liệu không hoàn toàn cân bằng.

## 5. Macro F1 là gì?

Macro F1 là trung bình F1-score của tất cả class, mỗi class có trọng số ngang nhau. Class ít ảnh và class nhiều ảnh đều đóng góp như nhau vào macro F1.

Vì vậy macro F1 phản ánh tốt hơn chất lượng model trên toàn bộ class, không để class lớn che lấp class nhỏ.

## 6. Vì sao không chỉ dùng accuracy?

Accuracy có thể cao nếu model làm tốt trên các class nhiều ảnh, nhưng vẫn kém ở các class ít ảnh hoặc class khó. Trong hệ thống nông sản, mỗi class đều quan trọng vì người dùng có thể upload bất kỳ loại nào.

Do đó cần xem thêm macro F1, macro recall, per-class recall và confusion matrix để biết model yếu ở đâu.

## 7. Vì sao macro F1 quan trọng trong bài toán nhiều class?

Macro F1 đánh giá đều các class, không phụ thuộc quá nhiều vào số lượng ảnh của từng class. Nếu một vài class có kết quả rất thấp, macro F1 sẽ phản ánh điều đó rõ hơn accuracy.

Trong báo cáo, macro F1 là metric chính để chứng minh model không chỉ tốt tổng thể mà còn tương đối ổn trên nhiều class.

## 8. Per-class recall giúp phát hiện vấn đề gì?

Per-class recall giúp biết class nào model hay bỏ sót. Ví dụ nếu recall của `apple` thấp, nghĩa là nhiều ảnh apple thật không được nhận ra là apple.

Thông tin này rất quan trọng để quyết định class nào cần bổ sung dữ liệu, làm sạch nhãn hoặc tạm thời đưa vào manual review.

## 9. Confusion matrix dùng để làm gì?

Confusion matrix cho biết model nhầm class nào sang class nào. Hàng thường là nhãn thật, cột là nhãn dự đoán.

Thông qua confusion matrix, ta có thể phát hiện các cặp class dễ nhầm, ví dụ một loại quả bị dự đoán sang loại quả có màu sắc hoặc hình dạng tương tự.

## 10. Class nào model nhận diện tốt nhất?

Theo F1-score trên test set Stage B supported v1, các class tốt nhất gồm:

- `pineapple`: F1-score `0.9851`.
- `durian`: F1-score `0.9645`.
- `mulberry`: F1-score `0.9055`.

Các class này có precision và recall đều cao, nghĩa là model vừa ít nhầm vào class đó vừa ít bỏ sót ảnh thật của class đó.

## 11. Class nào model nhận diện yếu nhất?

Các class yếu nhất theo F1-score gồm:

- `guava`: F1-score `0.4805`.
- `pomelo`: F1-score `0.5212`.
- `tomato`: F1-score `0.5659`.
- `apple`: F1-score `0.6415`.

Các class này cần được phân tích thêm về dữ liệu, hình ảnh nhầm lẫn và chất lượng nhãn.

## 12. Vì sao `guava`, `pomelo`, `tomato` có vấn đề?

`guava` có precision cao nhưng recall thấp, nghĩa là khi model đoán guava thì thường đúng, nhưng model bỏ sót nhiều ảnh guava thật. Điều này có thể do guava có hình dạng/màu sắc đa dạng hoặc dễ nhầm với quả khác.

`pomelo` và `tomato` có recall cao nhưng precision thấp, nghĩa là model hay dự đoán ảnh class khác thành pomelo hoặc tomato. Đây là dấu hiệu model bị hút về các class này trong một số trường hợp.

## 13. Precision cao nhưng recall thấp nghĩa là gì?

Precision cao nhưng recall thấp nghĩa là model rất thận trọng khi dự đoán class đó. Khi model dự đoán thì thường đúng, nhưng nó bỏ sót nhiều ảnh thật của class đó.

Ví dụ `guava` có precision `1.0000` nhưng recall `0.3162`, nghĩa là model ít khi đoán guava sai, nhưng rất nhiều ảnh guava thật bị dự đoán thành class khác.

## 14. Recall cao nhưng precision thấp nghĩa là gì?

Recall cao nhưng precision thấp nghĩa là model nhận ra được hầu hết ảnh thật của class đó, nhưng cũng kéo nhầm nhiều ảnh class khác vào class này.

Ví dụ `tomato` có recall `1.0000` nhưng precision `0.3946`, nghĩa là model bắt được ảnh tomato thật nhưng đồng thời cũng dự đoán nhiều ảnh không phải tomato thành tomato.

## 15. Metric chính để chọn model là gì?

Metric chính là macro F1-score, macro recall, per-class recall và confusion matrix. Accuracy chỉ dùng làm chỉ số tham khảo.

Lý do là bài toán nhiều class cần quan tâm đến chất lượng từng class, không chỉ tỷ lệ đúng tổng thể.

---

# 7. Câu hỏi về kết quả

## 1. Accuracy cuối cùng của ResNet50 là bao nhiêu?

Stage B ResNet50 supported v1 đạt test accuracy tham khảo khoảng `0.8089`. Đây là accuracy trên tập test của `dataset_fruit_supported_v1`.

Tuy nhiên trong báo cáo nên nhấn mạnh accuracy chỉ là metric phụ. Metric quan trọng hơn là macro F1-score và recall theo từng class.

## 2. Macro F1 của ResNet50 là bao nhiêu?

Macro F1-score của Stage B ResNet50 supported v1 là khoảng `0.8023`. Đây là chỉ số quan trọng vì nó đánh giá trung bình F1 của các class, mỗi class có trọng số như nhau.

Kết quả này cho thấy model đạt mức khá tốt cho production v1, nhưng vẫn còn một số class yếu cần cải thiện.

## 3. ResNet50 tốt hơn MobileNetV2 bao nhiêu?

Trong kết quả so sánh, ResNet50 đạt accuracy khoảng `0.8089` và macro F1-score `0.8023`, trong khi MobileNetV2 baseline cũ đạt accuracy khoảng `0.7503` và macro F1-score `0.7113`.

Như vậy ResNet50 cao hơn khoảng `0.0586` accuracy và `0.0910` macro F1-score. Đây là lý do ResNet50 được chọn làm model chính.

## 4. Những class có F1-score cao nhất là class nào?

Các class có F1-score cao nhất gồm:

- `pineapple`: `0.9851`.
- `durian`: `0.9645`.
- `mulberry`: `0.9055`.
- `sugar_apple`: `0.8988`.
- `eggplant`: `0.8960`.

Các class này có đặc trưng hình ảnh khá rõ hoặc dữ liệu đủ ổn định để model học tốt.

## 5. Những class có F1-score thấp nhất là class nào?

Các class có F1-score thấp nhất gồm:

- `guava`: `0.4805`.
- `pomelo`: `0.5212`.
- `tomato`: `0.5659`.
- `apple`: `0.6415`.
- `ambarella`: `0.6554`.

Các class này cần được cải thiện bằng cách bổ sung dữ liệu, kiểm tra nhãn, phân tích ảnh nhầm lẫn và có thể điều chỉnh threshold.

## 6. Vì sao một số class bị nhầm lẫn?

Một số class bị nhầm vì chúng có hình dạng, màu sắc hoặc texture gần giống nhau. Ngoài ra, ảnh chụp thực tế có thể khác nhau về ánh sáng, góc chụp, nền ảnh, mức độ che khuất và trạng thái chín của quả.

Một nguyên nhân khác là dữ liệu của class đó chưa đủ đa dạng hoặc có nhãn chưa thật sự sạch, khiến model học chưa ổn định.

## 7. Class nào hay bị nhầm với class nào?

Theo test upload thực tế, một số cặp nhầm lẫn đáng chú ý gồm:

- `durian` bị nhầm sang `jackfruit_cempedak`.
- `apple` bị nhầm sang `tomato`.
- `avocado` bị nhầm sang `zucchini`.
- `guava` bị nhầm sang `pomelo` hoặc `caimito`.

Các cặp này cần được phân tích bằng ảnh lỗi thực tế để hiểu nguyên nhân cụ thể.

## 8. Kết quả trên tập test khác gì kết quả upload thực tế?

Tập test thường được chia từ dataset đã xử lý, có phân phối gần với dữ liệu train. Ảnh upload thực tế có thể khác nhiều hơn về góc chụp, nền, ánh sáng, độ mờ, kích thước và cách người dùng chụp.

Vì vậy kết quả upload thực tế thường thấp hơn hoặc có nhiều manual review hơn. Đây là lý do cần test bằng `test_uploads_labeled` ngoài tập test chuẩn.

## 9. Vì sao accuracy trên ảnh upload thực tế có thể thấp hơn test set?

Ảnh upload thực tế có domain shift so với dataset train/test. Người dùng có thể chụp ảnh trong điều kiện thiếu sáng, nền phức tạp, vật thể bị che, nhiều vật thể trong một ảnh hoặc crop không chuẩn.

Do đó accuracy thực tế phản ánh khả năng triển khai production tốt hơn test set nội bộ. Đây là điểm cần trình bày trung thực trong báo cáo.

## 10. Model đã đủ dùng cho đồ án chưa? Vì sao?

Với phạm vi đồ án tốt nghiệp, model đã đủ ổn để tích hợp vào website và demo chức năng nhận diện ảnh nông sản. Hệ thống có pipeline rõ ràng, có model so sánh, có metric đánh giá, có threshold, có manual review và đã triển khai qua API.

Tuy nhiên, nếu triển khai thương mại thực tế, cần tiếp tục cải thiện dataset, class yếu, ảnh upload thực tế và monitoring sau triển khai.

---

# 8. Câu hỏi về inference và threshold

## 1. Khi user upload ảnh, model dự đoán như thế nào?

Ảnh được preprocess về định dạng phù hợp với ResNet50, sau đó đi qua Stage A. Nếu Stage A xác định ảnh là nông sản, ảnh tiếp tục đi qua Stage B để phân loại loại nông sản.

Stage B trả về xác suất softmax cho 32 class supported. Hệ thống lấy top-1, top-2 và margin để quyết định kết quả cuối.

## 2. Softmax output có ý nghĩa gì?

Softmax output là phân phối xác suất trên các class. Class có xác suất cao nhất là top-1 prediction.

Tuy nhiên, softmax không đảm bảo model thật sự đúng. Vì vậy production flow không chỉ dựa vào top-1 mà còn xét confidence threshold và top1-top2 margin.

## 3. Top-1 confidence là gì?

Top-1 confidence là xác suất cao nhất mà model gán cho class dự đoán. Ví dụ nếu model dự đoán `pineapple` với xác suất 0.92 thì top-1 confidence là 0.92.

Top-1 confidence càng cao thì model càng tự tin, nhưng vẫn cần kiểm tra margin và bối cảnh ảnh.

## 4. Top-1 và top-2 margin là gì?

Top-1/top-2 margin là khoảng cách giữa confidence của class cao nhất và class cao thứ hai. Ví dụ top-1 là 0.86, top-2 là 0.78 thì margin là 0.08.

Margin thấp nghĩa là model đang phân vân giữa hai class. Khi đó ảnh nên đưa vào manual review thay vì auto accept.

## 5. Vì sao dùng threshold confidence 0.85?

Threshold 0.85 được chọn để ưu tiên độ tin cậy khi auto accept. Nếu confidence dưới 0.85, hệ thống xem model chưa đủ chắc chắn và đưa ảnh vào manual review.

Ngưỡng này giúp giảm số ảnh bị auto accept sai, phù hợp với mục tiêu production v1 là an toàn và đáng tin cậy hơn là cố gắng tự động hóa mọi trường hợp.

## 6. Nếu giảm threshold xuống 0.80 thì điều gì xảy ra?

Giảm threshold xuống 0.80 sẽ tăng số ảnh được auto accept và giảm số ảnh manual review. Tuy nhiên, accuracy của nhóm auto accept có thể giảm vì hệ thống chấp nhận cả những dự đoán kém chắc chắn hơn.

Trong test hiện tại, giảm từ 0.85 xuống 0.80 làm coverage tăng nhưng số lỗi Stage B cũng tăng. Vì vậy cần cân nhắc giữa tự động hóa và độ chính xác.

## 7. Vì sao ảnh confidence thấp không nên tự động kết luận?

Confidence thấp nghĩa là model không đủ chắc chắn về class dự đoán. Nếu vẫn tự động kết luận, hệ thống có thể gắn sai loại sản phẩm cho người dùng.

Trong production, kết quả sai có thể ảnh hưởng đến trải nghiệm người dùng và độ tin cậy của hệ thống. Vì vậy ảnh confidence thấp nên được đưa vào manual review hoặc kiểm tra bổ sung.

## 8. Khi nào ảnh được đưa vào manual review?

Ảnh được đưa vào manual review khi:

- Stage A không chắc ảnh là fruit hay other.
- Ảnh thuộc class unsupported.
- Stage B top-1 confidence thấp.
- Stage B top-1 và top-2 quá gần nhau.
- Ảnh lỗi, khó đọc hoặc không phù hợp.

Manual review là cơ chế an toàn của hệ thống.

## 9. Khi nào gọi Gemini API?

Gemini API có thể được gọi khi model ResNet50 không đủ tự tin, ví dụ top-1 confidence dưới ngưỡng hoặc margin thấp. Gemini đóng vai trò lớp hỗ trợ kiểm tra thêm hoặc tạo gợi ý cho manual review.

Không nên dùng Gemini để thay thế hoàn toàn ResNet50 trong mọi trường hợp, vì pipeline chính vẫn là model đã train cho bài toán cụ thể của hệ thống.

## 10. Gemini đóng vai trò thay thế model hay hỗ trợ kiểm tra?

Gemini nên đóng vai trò hỗ trợ kiểm tra hoặc fallback cho các trường hợp không chắc chắn. ResNet50 vẫn là model chính vì được train trên taxonomy và dataset của project.

Thiết kế tốt là ResNet50 auto accept khi đủ tin cậy; Gemini hoặc manual review xử lý các trường hợp khó, mơ hồ hoặc ngoài phạm vi.

## 11. Nếu Stage A chặn nhầm ảnh nông sản thì xử lý thế nào?

Stage A có vùng uncertain để tránh chặn nhầm ảnh nông sản thật. Nếu fruit probability không quá thấp nhưng chưa đủ cao, ảnh được đưa vào `manual_review_stage_a_uncertain` thay vì kết luận là other.

Điều này đặc biệt quan trọng với các ảnh nông sản khó, thiếu sáng hoặc khác phân phối train.

## 12. Nếu Stage B đoán nhầm class thì hệ thống log ra sao?

Script test upload flow ghi log chi tiết từng ảnh gồm path, true label, group, Stage A probability, Stage B top-1, top-2, confidence, margin, final route, evaluation result và reason.

Các lỗi Stage B được lưu trong `upload_flow_error_cases.csv` và confused pairs được lưu trong `upload_flow_confused_pairs.csv`. Nhờ đó có thể phân tích class nào sai nhiều nhất và sai theo hướng nào.

---

# 9. Câu hỏi về deployment

## 1. Model được deploy ở đâu?

Model được lưu dưới dạng file `.keras` trong thư mục experiment và được AI API load để inference. Website không chạy model trực tiếp mà gọi API để nhận kết quả dự đoán.

Cách này giúp tách frontend khỏi logic AI, dễ cập nhật model và bảo mật tốt hơn.

## 2. Website gọi AI thông qua API như thế nào?

Frontend gửi ảnh upload đến Supabase Edge Function hoặc backend trung gian. Edge Function gọi AI API kèm ảnh hoặc đường dẫn ảnh, sau đó nhận kết quả dự đoán từ AI API và trả về frontend.

Flow này giúp frontend không cần biết chi tiết model, threshold hay xử lý ảnh bên trong.

## 3. Supabase Edge Function đóng vai trò gì?

Supabase Edge Function đóng vai trò lớp trung gian giữa frontend và AI API. Nó có thể xử lý xác thực, validate request, gọi AI API, chuẩn hóa response và bảo vệ thông tin nhạy cảm.

Edge Function cũng giúp tránh việc frontend gọi trực tiếp đến các service bên trong hoặc để lộ API key.

## 4. Vì sao không chạy model trực tiếp trong frontend?

Model ResNet50 tương đối nặng và cần môi trường runtime phù hợp như Python/TensorFlow. Chạy trực tiếp trong frontend sẽ khó triển khai, tốn tài nguyên client và khó bảo mật.

Đưa model lên AI API giúp quản lý version model, threshold, logging và monitoring dễ hơn.

## 5. Vì sao không để frontend gọi trực tiếp model/Gemini?

Nếu frontend gọi trực tiếp model hoặc Gemini, API key và logic xử lý có thể bị lộ. Ngoài ra, frontend không phải nơi phù hợp để kiểm soát threshold, logging và route production.

Thông qua backend hoặc Edge Function, hệ thống kiểm soát tốt hơn về bảo mật, quota, log và lỗi.

## 6. Khi thay đổi threshold có cần train lại model không?

Không cần train lại model nếu chỉ thay đổi threshold inference. Threshold là logic quyết định sau khi model đã trả xác suất, không phải trọng số model.

Tuy nhiên cần test lại upload flow để xem thay đổi threshold ảnh hưởng thế nào đến accuracy, manual review rate và số lỗi auto accept.

## 7. Khi thay đổi threshold có cần deploy lại không?

Có thể cần deploy lại nếu threshold được hardcode trong AI API hoặc Supabase Edge Function. Nếu threshold được cấu hình bằng environment variable, chỉ cần cập nhật biến môi trường và restart/redeploy service tương ứng.

Không cần deploy lại model nếu file model không thay đổi.

## 8. Nếu threshold nằm trong AI API thì deploy lại phần nào?

Nếu threshold nằm trong AI API, chỉ cần deploy lại AI API hoặc restart service AI API sau khi cập nhật code/config. Frontend và Supabase Edge Function không cần thay đổi nếu request/response không đổi.

Sau khi deploy, cần chạy test upload flow hoặc test một số ảnh mẫu để đảm bảo logic mới hoạt động đúng.

## 9. Nếu threshold nằm trong Supabase Edge Function thì deploy lại phần nào?

Nếu threshold nằm trong Supabase Edge Function, cần deploy lại Edge Function. AI API không cần deploy lại nếu nó chỉ trả raw prediction và Edge Function quyết định route.

Nên đặt threshold ở nơi dễ quản lý, ví dụ environment variable, để thay đổi an toàn hơn.

## 10. Làm sao bảo mật API key?

API key phải được lưu trong environment variables ở server hoặc Supabase secrets, không đưa vào frontend code. Frontend chỉ gọi Edge Function hoặc backend đã được bảo vệ.

Ngoài ra cần giới hạn quyền, kiểm tra request, log lỗi và tránh trả thông tin nhạy cảm trong response.

## 11. Làm sao xử lý ảnh lỗi hoặc ảnh không đọc được?

AI API cần validate file upload. Nếu ảnh không đọc được, corrupt hoặc không đúng định dạng, hệ thống trả route `error` cùng reason rõ ràng.

Lỗi ảnh không nên làm crash service. Nó phải được log lại để backend/frontend hiển thị thông báo phù hợp cho người dùng.

## 12. Làm sao hệ thống trả kết quả về frontend?

AI API trả response dạng JSON gồm class dự đoán, confidence, route, reason và các thông tin cần thiết. Supabase Edge Function nhận response này, chuẩn hóa lại nếu cần và trả về frontend.

Frontend dựa vào route để quyết định cho phép tiếp tục, cảnh báo người dùng hoặc đưa ảnh vào manual review.

## 13. Thời gian dự đoán ảnh khoảng bao lâu?

Thời gian dự đoán phụ thuộc vào server, CPU/GPU, kích thước ảnh, thời gian upload và việc có gọi thêm Gemini hay không. Với một model ResNet50 đã load sẵn trong memory, inference một ảnh thường đủ nhanh cho trải nghiệm upload.

Điểm quan trọng là không nên load model lại cho mỗi request. Model nên được load khi service khởi động và tái sử dụng cho các request.

## 14. Nếu nhiều người upload cùng lúc thì hệ thống xử lý thế nào?

Trong production thực tế, AI API cần hỗ trợ nhiều request đồng thời bằng cơ chế server phù hợp, queue hoặc scale instance. Nếu traffic tăng, có thể triển khai API trên server mạnh hơn hoặc dùng container/cloud service.

Với phạm vi đồ án, cần trình bày rằng hệ thống đã có API inference riêng và có thể mở rộng bằng cách scale backend service.

## 15. Hạn chế hiện tại của deployment là gì?

Hạn chế hiện tại gồm: model còn yếu ở một số class, tập ảnh upload thực tế chưa đủ lớn, chưa có monitoring production dài hạn và chưa có cơ chế active learning tự động.

Ngoài ra, nếu gọi Gemini fallback, cần kiểm soát chi phí, độ trễ và bảo mật API key.

---

# 10. Câu hỏi về testing

## 1. Em đã test hệ thống bằng những phương pháp nào?

Project được test ở nhiều mức:

- Test dataset split và kiểm tra data leakage.
- Test training bằng validation/test set.
- Test metric bằng classification report và confusion matrix.
- Test upload flow bằng thư mục ảnh thật `test_uploads_labeled`.
- Test API/website integration khi người dùng upload ảnh thật.

Cách test này giúp đánh giá cả model offline và flow production thực tế.

## 2. Test dataset khác gì test upload thực tế?

Test dataset là tập test được chia từ dữ liệu đã preprocess, thường cùng phân phối với train. Test upload thực tế là ảnh do người dùng upload hoặc ảnh mô phỏng người dùng upload, có thể khác về ánh sáng, nền, góc chụp và chất lượng ảnh.

Vì vậy test upload thực tế quan trọng để kiểm tra khả năng triển khai của hệ thống, không chỉ khả năng trên dataset nội bộ.

## 3. Test upload flow gồm những nhóm nào?

Test upload flow chia ảnh thành các nhóm:

- `supported`: class được Stage B production v1 hỗ trợ.
- `unsupported_v1`: class nông sản nhưng chưa hỗ trợ trong production v1.
- `out_of_scope/other`: ảnh không thuộc phạm vi nông sản.

Mỗi nhóm có cách đánh giá khác nhau để phản ánh đúng logic production.

## 4. Thế nào là `stage_a_other`?

`stage_a_other` là route khi Stage A xác định ảnh không phải nông sản. Ảnh này không được đưa sang Stage B.

Với ảnh out-of-scope, route này là đúng. Nhưng nếu ảnh thật sự là nông sản supported mà bị Stage A chặn thành other, đó là lỗi cần phân tích.

## 5. Thế nào là `stage_b_supported`?

`stage_b_supported` hoặc `stage_b_supported_auto_accept` là trường hợp ảnh đi qua Stage A, được Stage B dự đoán thành một class supported và vượt qua threshold confidence/margin.

Nếu class dự đoán trùng true label, đây là dự đoán đúng. Nếu khác true label, đây là lỗi Stage B.

## 6. Thế nào là `low_confidence`?

`low_confidence` là trường hợp Stage B có top-1 confidence thấp hơn threshold. Model chưa đủ chắc chắn để auto accept.

Trong production, ảnh low confidence nên đưa vào manual review hoặc kiểm tra bổ sung, không nên ép kết luận.

## 7. Thế nào là `manual_review_unsupported`?

`manual_review_unsupported` là route cho ảnh thuộc class nông sản chưa được hỗ trợ trong production v1. Những ảnh này không đi qua Stage B để tránh bị ép thành class supported.

Đây là flow đúng với các class unsupported, vì hệ thống thừa nhận hiện tại chưa hỗ trợ class đó.

## 8. Thế nào là `error`?

`error` là route khi ảnh bị lỗi, không đọc được, sai định dạng hoặc quá trình inference gặp lỗi. Hệ thống cần log rõ reason để debug.

Route này giúp hệ thống xử lý lỗi an toàn thay vì crash hoặc trả kết quả không đáng tin cậy.

## 9. Tổng số ảnh upload test là bao nhiêu?

Trong lần test upload flow đã chạy, tổng số ảnh là 159 ảnh. Trong đó có supported, unsupported v1 và out-of-scope/other.

Con số này dùng để đánh giá flow production với ảnh upload thực tế, không thay thế hoàn toàn test set chuẩn nhưng rất hữu ích để kiểm tra tích hợp thực tế.

## 10. Bao nhiêu ảnh nhận đúng?

Với threshold Stage B 0.85, hệ thống auto accepted 90 ảnh và supported auto accuracy đạt khoảng 82.02%. Điều này nghĩa là trong nhóm ảnh supported được auto accept, khoảng 82% được nhận đúng.

Cần lưu ý manual review không tính là đoán sai production, vì hệ thống không tự động kết luận trong các trường hợp không chắc chắn.

## 11. Bao nhiêu ảnh sai?

Trong test upload flow với threshold 0.85, số ảnh Stage B đoán sai supported là 16. Ngoài ra có 1 ảnh supported bị Stage A chặn nhầm.

Các lỗi này được ghi trong file error cases để phân tích chi tiết.

## 12. Bao nhiêu ảnh đưa vào manual review?

Với threshold Stage B 0.85, có 66 ảnh được đưa vào manual review, tương ứng manual review rate khoảng 41.51%.

Tỷ lệ này khá cao nhưng phù hợp với hướng production an toàn, vì hệ thống ưu tiên không auto accept những ảnh chưa đủ chắc chắn.

## 13. Class nào sai nhiều nhất?

Trong test upload flow, class sai nhiều nhất là `guava`. Đây cũng là class có F1-score thấp nhất trên test set Stage B.

Điều này cho thấy điểm yếu của model khá nhất quán giữa đánh giá offline và test upload thực tế.

## 14. Ảnh nào bị Stage A chặn nhầm?

Danh sách ảnh bị Stage A chặn nhầm được lưu trong log upload flow và file error cases. Khi báo cáo, không nhất thiết liệt kê toàn bộ ảnh trong slide, nhưng cần nói rõ hệ thống có ghi lại từng ảnh lỗi để phân tích.

Trong project, thông tin này nằm trong `logs/upload_flow_error_cases.csv` hoặc thư mục log tương ứng của lần test.

## 15. Ảnh nào bị Stage B nhầm class?

Ảnh bị Stage B nhầm class được lưu trong `upload_flow_error_cases.csv`, kèm true label, predicted label, confidence, top-2 và reason.

Ngoài ra, các cặp nhầm lẫn tổng hợp được lưu trong `upload_flow_confused_pairs.csv`, giúp biết class nào hay bị nhầm với class nào.

## 16. Các file log test được lưu ở đâu?

Các file log test upload flow được lưu trong thư mục `logs/`, gồm:

- `upload_flow_production.log`.
- `upload_flow_predictions.csv`.
- `upload_flow_summary.json`.
- `upload_flow_per_class_summary.csv`.
- `upload_flow_error_cases.csv`.
- `upload_flow_confused_pairs.csv`.

Các file này là bằng chứng quan trọng khi trình bày testing.

## 17. Vì sao manual review không tính là model đoán sai production?

Manual review nghĩa là hệ thống không tự động đưa ra kết luận cuối cùng. Vì vậy nó không được tính là dự đoán sai production, mà được tính là trường hợp cần kiểm tra thêm.

Trong production, một hệ thống AI an toàn cần biết từ chối hoặc chuyển tiếp khi không chắc chắn. Do đó manual review là một phần của thiết kế, không phải lỗi trực tiếp của model.

## 18. Làm sao đánh giá hệ thống trong điều kiện thực tế?

Cần đánh giá bằng ảnh upload thật, không chỉ bằng test set. Các chỉ số quan trọng gồm auto accuracy, auto coverage, manual review rate, số ảnh bị Stage A chặn nhầm, số ảnh Stage B nhầm class và top confused pairs.

Ngoài ra cần tiếp tục thu thập feedback từ người dùng và quản trị viên để cải thiện dataset ở các phiên bản sau.

---

# 11. Câu hỏi phản biện khó

## 1. Tại sao không dùng toàn bộ 40 class nông sản?

Vì không phải class nào cũng đạt chất lượng đủ tốt để đưa vào production. Nếu đưa cả class yếu vào, hệ thống có thể dự đoán sai nhiều hơn và làm giảm độ tin cậy.

Production v1 ưu tiên các class ổn định trước. Các class yếu được đưa vào unsupported/manual review để cải thiện dần ở phiên bản sau.

## 2. Việc loại 8 class yếu có làm giảm giá trị hệ thống không?

Việc loại 8 class yếu không làm giảm giá trị hệ thống mà giúp hệ thống thực tế và đáng tin cậy hơn. Thay vì cố hỗ trợ tất cả nhưng sai nhiều, hệ thống công bố rõ phạm vi hỗ trợ và xử lý class chưa hỗ trợ bằng manual review.

Đây là hướng thiết kế production an toàn, phù hợp với hệ thống có AI.

## 3. Nếu người dùng upload class chưa hỗ trợ thì hệ thống có sai không?

Nếu hệ thống nhận diện được class đó thuộc nhóm unsupported và đưa vào manual review thì không xem là sai flow. Sai chỉ xảy ra nếu hệ thống tự động nhận class chưa hỗ trợ thành một class supported không đúng.

Vì vậy pipeline có rule không cho unsupported class đi vào Stage B production.

## 4. Dataset có đủ đại diện cho ảnh thực tế không?

Dataset đã được làm sạch và có số lượng tương đối lớn, nhưng vẫn chưa thể đại diện hoàn toàn cho mọi ảnh thực tế. Ảnh người dùng upload có thể đa dạng hơn về góc chụp, ánh sáng, nền và chất lượng.

Vì vậy project có thêm bước test upload thực tế và manual review để xử lý domain shift.

## 5. Ảnh từ internet và ảnh người dùng upload khác nhau thế nào?

Ảnh từ internet thường đẹp hơn, rõ hơn, vật thể ở trung tâm và ánh sáng tốt hơn. Ảnh người dùng upload có thể bị mờ, thiếu sáng, nền phức tạp, nhiều vật thể hoặc chụp không đúng trọng tâm.

Sự khác biệt này làm model có thể giảm hiệu năng khi chạy thực tế, nên cần test upload flow và bổ sung dữ liệu thật.

## 6. Nếu ảnh có nhiều loại nông sản trong cùng một hình thì sao?

Pipeline hiện tại là image classification, giả định ảnh chính chứa một loại nông sản cần kiểm tra. Nếu ảnh có nhiều loại nông sản, model có thể dự đoán theo vật thể nổi bật nhất hoặc bị nhầm.

Trong trường hợp này, hướng phát triển phù hợp là dùng object detection hoặc yêu cầu người dùng upload ảnh rõ một sản phẩm chính.

## 7. Nếu ảnh bị che khuất, thiếu sáng, nền phức tạp thì sao?

Model có thể giảm confidence hoặc dự đoán sai. Với ảnh confidence thấp hoặc margin thấp, hệ thống đưa vào manual review thay vì auto accept.

Đây là lý do production flow cần threshold và không chỉ dựa vào top-1 prediction.

## 8. Smart crop có thể làm mất thông tin không?

Có thể, nếu crop quá sát hoặc phát hiện sai vùng object. Vì vậy smart crop trong project được thiết kế nhẹ, có điều kiện kiểm tra bounding box, tỷ lệ vùng crop và kích thước tối thiểu.

Nếu crop không đủ tự tin, hệ thống giữ ảnh gốc đã resize/pad thay vì crop. Mục tiêu là hỗ trợ model học tốt hơn nhưng không phá hỏng hình dạng tổng thể.

## 9. Tại sao không dùng object detection để crop chính xác hơn?

Object detection cần dữ liệu bounding box và quá trình annotation phức tạp hơn. Trong phạm vi đồ án, bài toán chính là phân loại ảnh sản phẩm, nên dùng detector sẽ làm pipeline nặng và vượt phạm vi.

Smart crop nhẹ là giải pháp cân bằng: cải thiện vùng ảnh đầu vào nhưng không yêu cầu dữ liệu bounding box.

## 10. Model có bị bias theo nền ảnh không?

Có khả năng, nếu dataset có nhiều ảnh cùng nền hoặc class nào đó thường xuất hiện với background đặc trưng. Đây là rủi ro phổ biến trong image classification.

Để giảm bias, dataset cần đa dạng nền, augmentation nhẹ và test bằng ảnh upload thực tế. Trong tương lai có thể dùng detection/segmentation để tập trung vào object hơn.

## 11. Làm sao chứng minh hệ thống không học nhầm background?

Không thể chứng minh tuyệt đối chỉ bằng accuracy. Cần phân tích ảnh lỗi, confusion matrix, test ảnh với nhiều nền khác nhau và có thể dùng Grad-CAM để xem model tập trung vào vùng nào.

Trong phạm vi đồ án, việc test upload thực tế và smart crop là bước giảm rủi ro model học nền.

## 12. Vì sao một số class precision thấp?

Precision thấp nghĩa là model kéo nhầm ảnh class khác vào class đó. Nguyên nhân có thể là class đó có đặc trưng quá phổ biến, màu/hình dạng giống nhiều class khác hoặc dữ liệu train làm model bị thiên lệch.

Ví dụ `tomato` và `pomelo` có precision thấp, nghĩa là nhiều ảnh không thuộc class đó bị dự đoán thành class đó.

## 13. Vì sao một số class recall thấp?

Recall thấp nghĩa là model bỏ sót nhiều ảnh thật của class đó. Nguyên nhân có thể là dữ liệu class đó quá đa dạng, nhãn chưa sạch, ảnh khó hoặc class giống class khác.

Ví dụ `guava` có recall thấp, cho thấy nhiều ảnh guava thật bị model dự đoán sang class khác.

## 14. Nếu muốn nâng accuracy lên nữa thì làm gì?

Cần tập trung vào các class yếu trước. Các hướng cải thiện gồm: bổ sung ảnh thật, làm sạch nhãn, cân bằng dữ liệu, phân tích confusion matrix, dùng Grad-CAM, thử object detection/crop tốt hơn và tinh chỉnh threshold.

Không nên chỉ tăng augmentation mạnh vì có thể làm nhiễu đặc trưng fine-grained.

## 15. Nếu triển khai thật cho nhiều người dùng thì cần cải thiện gì?

Cần thêm monitoring production, lưu feedback người dùng, dashboard lỗi, active learning, versioning model, kiểm soát API latency, bảo mật API key và cơ chế rollback model.

Ngoài ra cần mở rộng dataset bằng ảnh upload thật và cải thiện dần các class unsupported.

---

# 12. Câu hỏi về hướng phát triển

## 1. Hướng cải thiện model tiếp theo là gì?

Hướng cải thiện tiếp theo là tập trung vào class yếu như `guava`, `pomelo`, `tomato`, `apple`, `ambarella`. Cần bổ sung dữ liệu thật, kiểm tra nhãn, phân tích ảnh nhầm và cải thiện preprocessing.

Sau khi dữ liệu tốt hơn, có thể train lại ResNet50 hoặc thử thêm các kiến trúc mạnh hơn để so sánh.

## 2. Có nên bổ sung thêm dữ liệu thật từ người dùng không?

Có. Dữ liệu thật từ người dùng rất quan trọng vì nó phản ánh đúng điều kiện production. Ảnh người dùng upload thường khác ảnh dataset ban đầu.

Tuy nhiên cần có quy trình kiểm duyệt và gán nhãn lại trước khi đưa vào train để tránh làm nhiễu dataset.

## 3. Có nên dùng active learning không?

Có thể dùng active learning ở giai đoạn sau. Hệ thống có thể lưu các ảnh model confidence thấp hoặc ảnh bị manual review, sau đó con người gán nhãn lại và đưa vào dataset train mới.

Đây là cách hiệu quả để cải thiện model theo dữ liệu thực tế mà không phải gán nhãn ngẫu nhiên quá nhiều.

## 4. Có nên thêm object detection không?

Có thể cân nhắc nếu hệ thống cần xử lý ảnh có nhiều vật thể hoặc nền phức tạp. Object detection giúp xác định vùng nông sản chính trước khi phân loại.

Tuy nhiên cần dữ liệu bounding box và pipeline phức tạp hơn. Vì vậy đây nên là hướng phát triển sau production v1.

## 5. Có nên mở rộng thêm class không?

Có, nhưng nên mở rộng có kiểm soát. Mỗi class mới cần đủ dữ liệu sạch, test metric và kiểm tra upload flow trước khi đưa vào production.

Không nên thêm class chỉ vì có tên class, vì class yếu sẽ làm giảm chất lượng toàn hệ thống.

## 6. Có nên dùng ensemble nhiều model không?

Ensemble có thể tăng độ chính xác nhưng làm tăng chi phí inference, độ trễ và độ phức tạp triển khai. Với đồ án hiện tại, một model ResNet50 tốt và pipeline threshold/manual review là hợp lý hơn.

Ensemble có thể xem là hướng phát triển nếu hệ thống cần độ chính xác cao hơn và có đủ tài nguyên triển khai.

## 7. Có nên dùng Gemini như lớp xác minh thứ hai không?

Có thể dùng Gemini như lớp hỗ trợ cho các trường hợp model không chắc chắn, ví dụ confidence thấp hoặc margin thấp. Gemini không nên thay thế hoàn toàn model chính.

Thiết kế phù hợp là ResNet50 xử lý chính, Gemini hỗ trợ kiểm tra hoặc gợi ý trong manual review để tăng độ tin cậy.

## 8. Làm sao cập nhật model mà không ảnh hưởng website?

Cần tách model thành AI API riêng, dùng version model và giữ response format ổn định. Khi có model mới, deploy vào API, test nội bộ, sau đó chuyển traffic sang version mới.

Nếu có lỗi, có thể rollback về model cũ mà không cần sửa frontend.

## 9. Làm sao xây dựng dashboard theo dõi lỗi model?

Dashboard có thể hiển thị số ảnh upload, tỷ lệ auto accept, manual review rate, số lỗi theo class, top confused pairs, ảnh confidence thấp và lịch sử model version.

Dữ liệu dashboard lấy từ log inference và feedback manual review. Đây là nền tảng để cải thiện model liên tục.

## 10. Làm sao cải thiện các class yếu như `guava`, `pomelo`, `tomato`?

Cần phân tích từng class yếu theo hướng cụ thể:

- Kiểm tra ảnh bị nhầm trong confusion matrix.
- Kiểm tra nhãn sai hoặc ảnh không đại diện.
- Bổ sung ảnh thật đa dạng nền/góc chụp/ánh sáng.
- Loại ảnh nhiễu hoặc ảnh trùng.
- Test lại sau khi bổ sung dữ liệu.

Với `guava`, cần tăng recall bằng dữ liệu đa dạng hơn. Với `pomelo` và `tomato`, cần giảm false positive bằng cách bổ sung ảnh của các class dễ nhầm và kiểm tra lại threshold.

---

# 13. Câu trả lời ngắn nên học thuộc

## Vì sao chọn ResNet50?

Em chọn ResNet50 vì đây là kiến trúc CNN mạnh, phổ biến, có pretrained ImageNet và cho kết quả tốt hơn MobileNetV2 trong thử nghiệm của project. ResNet50 đạt macro F1-score khoảng `0.8023`, cao hơn MobileNetV2 baseline, nên được chọn làm model chính cho hệ thống.

## Vì sao dùng hai stage?

Vì hai stage giúp tách rõ bài toán kiểm tra ảnh có phải nông sản và bài toán phân loại loại nông sản. Stage A lọc ảnh `other`, Stage B chỉ tập trung phân loại các class nông sản supported. Cách này giảm nhầm lẫn và dễ kiểm soát production hơn.

## Vì sao không chỉ dùng accuracy?

Vì accuracy có thể che lấp class yếu, đặc biệt trong bài toán nhiều class. Em dùng macro F1, macro recall, per-class recall và confusion matrix để đánh giá công bằng hơn giữa các class.

## Model đã đủ dùng cho đồ án chưa?

Đủ dùng cho phạm vi đồ án vì hệ thống đã có dataset pipeline, training pipeline, evaluation, model comparison, threshold logic, manual review và đã tích hợp với website qua API. Tuy nhiên để triển khai thương mại thật, cần tiếp tục cải thiện các class yếu và thu thập thêm ảnh upload thực tế.

## Nếu hội đồng hỏi hạn chế lớn nhất là gì?

Hạn chế lớn nhất là một số class vẫn còn yếu, ví dụ `guava`, `pomelo`, `tomato`, và ảnh upload thực tế có thể khác dataset train. Vì vậy hệ thống dùng threshold và manual review để giảm rủi ro dự đoán sai trong production.

