# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán Việt Nam (LSTM-GRU Sharpe Loss)

Ứng dụng web được xây dựng bằng **Streamlit** giúp tối ưu hóa tỷ trọng danh mục đầu tư bằng mô hình học sâu lai **LSTM-GRU**. Mô hình sử dụng hàm loss tùy chỉnh tối ưu trực tiếp hệ số Sharpe ratio của danh mục đầu tư, kèm theo regularization Entropy để đảm bảo sự đa dạng hóa tài sản.

Dữ liệu lịch sử giá cổ phiếu được tải thời gian thực từ thị trường chứng khoán Việt Nam thông qua thư viện `vnstock`.

---

## 🌟 Các Tính Năng Chính

1. **Tải Dữ Liệu Tự Động & Caching:**
   - Chọn ngành đầu tư (Thép, Ngân hàng, Bất động sản, Bán lẻ, Công nghệ, v.v.).
   - Tự động tải dữ liệu giá lịch sử của toàn bộ các cổ phiếu thuộc ngành đã chọn qua thư viện `vnstock` (nguồn KBS).
   - Tích hợp cơ chế cache tránh tải lại dữ liệu khi thay đổi tham số giao diện, giảm thiểu lỗi chặn IP (Rate Limit).

2. **Lọc Cổ Phiếu Theo Sharpe Ratio:**
   - Tính toán hệ số Sharpe lịch sử cho từng cổ phiếu.
   - Lọc ra Top N cổ phiếu (mặc định là 10) có hiệu quả sinh lời tốt nhất để đưa vào huấn luyện mô hình.

3. **Huấn Luyện Mô Hình Học Sâu LSTM-GRU Trực Quan:**
   - Cấu hình linh hoạt các siêu tham số trong Sidebar: Số epochs, kích thước batch, độ dài cửa sổ (Window Size), số lượng units của LSTM/GRU, v.v.
   - Huấn luyện mô hình trên nhiều Seed ngẫu nhiên khác nhau và tự động chọn ra mô hình tối ưu nhất trên tập dữ liệu thử nghiệm (Test set).
   - Biểu đồ theo dõi trực quan tiến trình Loss (Train/Val Loss) qua các epoch.

4. **Trực Quan Hóa Tỷ Trọng Danh Mục Tối Ưu:**
   - Hiển thị tỷ trọng phân bổ tài sản tối ưu dưới dạng **Treemap** tương tác (Plotly) và biểu đồ cột.
   - Cho phép người dùng rê chuột để xem chi tiết tỷ trọng phần trăm phân bổ cho từng mã cổ phiếu.

5. **So Sánh Hiệu Quả Các Chiến Lược:**
   - Đánh giá chéo hiệu quả của 3 chiến lược đầu tư trên tập Test:
     - **LSTM-GRU (Dynamic):** Phân bổ linh hoạt theo dự báo của mô hình học sâu.
     - **Phân bổ đều (Equal Weight):** Tỷ trọng chia đều cho tất cả các mã cổ phiếu trong Top.
     - **Phân bổ 80-20:** Tập trung 80% vốn vào nhóm có Sharpe tốt nhất trong quá khứ và 20% vào nhóm còn lại.
   - Bảng so sánh chi tiết: Lợi nhuận trung bình năm, Độ lệch chuẩn năm (Rủi ro) và Hệ số Sharpe.
   - Biểu đồ cột kết hợp trục phụ (Dual-Axis Chart) trực quan so sánh cả 3 chỉ số cùng lúc.

---

## 🛠️ Hướng Dẫn Cài Đặt và Chạy Local

Làm theo các bước sau để thiết lập môi trường và chạy ứng dụng trên máy tính của bạn:

### Bước 1: Clone kho lưu trữ
```bash
git clone <URL_GITHUB_REPO_CUA_BAN>
cd <TEN_THU_MUC_DU_AN>
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Trên Windows:**
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Bước 3: Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy ứng dụng Streamlit
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Cloud

Để ứng dụng của bạn chạy trực tiếp trên web cho mọi người cùng sử dụng:

1. Đẩy mã nguồn lên một repository public trên **GitHub** (bao gồm các file `app.py`, `requirements.txt`, `README.md`, và `industry_tickers.py`).
2. Truy cập trang web [Streamlit Community Cloud](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
3. Nhấp vào nút **New app**.
4. Chọn repository, nhánh (branch) và nhập đường dẫn file chính là `app.py`.
5. **QUAN TRỌNG:** Nhấp vào nút **Advanced settings...** trước khi bấm Deploy. Tại mục **Python version**, hãy chọn **3.11** hoặc **3.10** (không dùng phiên bản mặc định quá mới như 3.14 vì TensorFlow chưa hỗ trợ).
6. Nhấp vào **Deploy!** Streamlit sẽ tự động cấu hình môi trường từ `requirements.txt` và khởi chạy trang web của bạn sau vài phút.

> [!WARNING]
> Nếu bạn đã deploy ứng dụng và gặp lỗi `installer returned a non-zero exit code` do TensorFlow, hãy truy cập vào cài đặt ứng dụng (Settings -> Advanced settings) trên Streamlit dashboard và đổi Python version về **3.11**, sau đó bấm **Reboot app**.
