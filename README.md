# Giải phương trình bằng Giải thuật Di truyền (GA)

Project 1: Ứng dụng giải thuật di truyền (Genetic Algorithm) để tìm **tất cả** nghiệm thực của phương trình một ẩn f(x) = 0 trong một khoảng cho trước.

## Tính năng

- [x] Giải thuật di truyền mã hóa số thực: chọn lọc tournament, lai ghép số học, đột biến Gauss, elitism
- [x] Tìm nhiều nghiệm bằng phương pháp chia đệ quy khoảng tìm kiếm
- [x] Xử lý hàm không xác định một phần (log, căn, tan, ...) và tràn số
- [x] Xử lý nghiệm kép, nghiệm nằm sát nhau, nghiệm dày đặc
- [x] Dừng sớm khi đủ chính xác hoặc khi không còn cải thiện
- [x] Vẽ đồ thị hàm số kèm nghiệm và biểu đồ hội tụ
- [x] Bộ test tự động (57 test)
- [ ] Giao diện Streamlit: nhập phương trình dạng chuỗi
- [ ] Cải tiến biểu đồ cho phương trình nhiều nghiệm
- [ ] So sánh với các phương pháp truyền thống (chia đôi, Newton, Brent)

## Phương pháp

### Giải thuật di truyền

Bài toán f(x) = 0 được chuyển thành bài toán tối ưu với hàm thích nghi:

```
fitness(x) = 1 / (1 + |f(x)|)
```

Fitness đạt giá trị lớn nhất bằng 1 tại nghiệm. Mỗi thế hệ gồm các bước:

1. **Đánh giá:** tính |f(x)| cho từng cá thể. Điểm không xác định được gán sai số vô cùng
2. **Elitism:** giữ nguyên 2 cá thể tốt nhất sang thế hệ sau
3. **Chọn lọc tournament:** mỗi lượt chọn ngẫu nhiên k = 3 cá thể, giữ cá thể tốt nhất
4. **Lai ghép số học:** con = α·cha + (1 − α)·mẹ
5. **Đột biến Gauss:** cộng nhiễu N(0, σ), với σ bằng 1% độ rộng khoảng tìm kiếm

GA dừng khi |f(x)| < 10⁻⁸, hoặc sau 30 thế hệ không cải thiện, hoặc khi hết số thế hệ tối đa.

### Tìm nhiều nghiệm bằng chia đệ quy

Một lần chạy GA chỉ hội tụ về một nghiệm. Để tìm tất cả nghiệm:

1. Chia khoảng tìm kiếm thành 20 khoảng con
2. Chạy GA trên từng khoảng con
3. Nếu tìm được nghiệm x, tách khoảng thành [a, x − ε] và [x + ε, b] rồi tiếp tục tìm trên hai nửa
4. Dừng khi không còn khoảng nào chứa nghiệm
5. Gộp các nghiệm trùng (bao gồm nghiệm kép)

```
                 [0, 1]
              tìm được 0.2
             /            \
     [0, 0.1999]       [0.2001, 1]
     không nghiệm      tìm được 0.7
                      /            \
           [0.2001, 0.6999]     [0.7001, 1]
             không nghiệm       không nghiệm
```

Khi f đổi dấu trong một khoảng (chắc chắn có nghiệm theo định lý giá trị trung gian) mà GA không tìm ra, solver chạy lại tối đa 3 lần với seed khác.

## Cài đặt

Yêu cầu: Python 3.10 trở lên

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

## Sử dụng

### Gọi từ Python

```python
import sys
sys.path.insert(0, "src")

import numpy as np
from ga_solver.config import GAConfig, SolverConfig
from ga_solver.solver import find_roots

f = lambda x: np.exp(-x) - np.sin(x)
res = find_roots(f, GAConfig(lower_bound=0, upper_bound=10, seed=42))

for r in res.roots:
    print(f"x = {r.x:.8f}   |f(x)| = {r.error:.2e}")
print(f"Thời gian: {res.elapsed_time:.3f}s")
```

Kết quả:

```
x = 0.58853274   |f(x)| = ...
x = 3.09636393   |f(x)| = ...
x = 6.28504927   |f(x)| = ...
x = 9.42469725   |f(x)| = ...
```

### Giao diện web *(đang phát triển)*

```bash
streamlit run app.py
```

### Chạy test

```bash
pytest           # chạy toàn bộ
pytest -v        # hiển thị tên từng test
```

## Tham số

### `GAConfig`: tham số giải thuật di truyền

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `pop_size` | 100 | Kích thước quần thể |
| `generations` | 200 | Số thế hệ tối đa |
| `crossover_rate` | 0.9 | Xác suất lai ghép |
| `mutation_rate` | 0.2 | Xác suất đột biến |
| `mutation_scale` | 0.01 | Độ lệch chuẩn đột biến, tính theo tỉ lệ độ rộng khoảng |
| `tournament_k` | 3 | Số cá thể mỗi lượt tournament |
| `elite_size` | 2 | Số cá thể tốt nhất được giữ lại |
| `lower_bound`, `upper_bound` | −10, 10 | Khoảng tìm kiếm |
| `tolerance` | 1e-8 | Ngưỡng dừng sớm |
| `patience` | 30 | Số thế hệ không cải thiện trước khi dừng |
| `seed` | None | Hạt giống ngẫu nhiên |

### `SolverConfig`: tham số tìm nhiều nghiệm

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `n_subintervals` | 20 | Số khoảng con ban đầu |
| `root_tolerance` | 1e-6 | Ngưỡng \|f(x)\| để chấp nhận nghiệm |
| `exclusion_radius` | 1e-4 | Bán kính loại bỏ quanh nghiệm đã tìm |
| `min_width` | 1e-3 | Độ rộng tối thiểu của khoảng |
| `max_retries` | 3 | Số lần chạy lại khi khoảng chắc chắn có nghiệm |
| `max_ga_runs` | 2000 | Giới hạn tổng số lần chạy GA |

## Kết quả thử nghiệm

Kiểm tra trên khoảng [−10, 10] (trừ khi ghi khác), đối chiếu với nghiệm tính bằng `scipy.optimize.brentq`:

| Phương trình | Số nghiệm | Sai lệch tối đa | Thời gian |
|---|---|---|---|
| x·eˣ = 1 | 1 | 9.4×10⁻¹⁰ | 121 ms |
| x = cos x | 1 | 4.6×10⁻⁹ | 121 ms |
| e⁻ˣ = sin x, trên [0, 10] | 4 | 6.2×10⁻⁹ | 142 ms |
| ln x + x = 2 | 1 | 2.0×10⁻¹⁰ | 113 ms |
| x² = 2ˣ | 3 | 1.5×10⁻⁹ | 126 ms |
| e^sin x = 2 | 6 | 6.7×10⁻⁹ | 160 ms |
| tan x = x | 5 | 7.8×10⁻⁶ | 1490 ms |
| Bessel J₀(x) = 0, trên [0, 20] | 6 | 3.3×10⁻⁸ | 160 ms |
| sin(10x) = 0 | 63 | < 10⁻⁴ | ~600 ms |
| sin²x = 0 (nghiệm kép), trên [−5, 5] | 3 | 1.6×10⁻⁵ | 429 ms |

## Hạn chế

- Hai nghiệm cách nhau dưới 10⁻⁴ (`exclusion_radius`) không thể tách riêng
- Nghiệm bội (ví dụ tan x − x tại x = 0) làm tăng đáng kể số lần chạy GA và giảm độ chính xác
- Chỉ tìm nghiệm thực trong khoảng người dùng chỉ định
- Chậm hơn các phương pháp truyền thống (Newton, Brent), nhưng không cần đạo hàm, không cần điểm xuất phát và tìm được nghiệm không đổi dấu

## Cấu trúc thư mục

```
PROJECT_1/
├── app.py                      # Giao diện Streamlit
├── src/ga_solver/
│   ├── config.py               # GAConfig, SolverConfig
│   ├── solver.py               # Tìm nhiều nghiệm (chia đệ quy)
│   ├── visualization.py        # Vẽ đồ thị
│   ├── baselines.py            # Phương pháp đối chứng
│   ├── parser/
│   │   └── expression.py       # Chuỗi biểu thức → hàm f(x)
│   └── ga/
│       ├── individual.py       # Khởi tạo quần thể
│       ├── fitness.py          # Hàm thích nghi
│       ├── operators.py        # Chọn lọc, lai ghép, đột biến
│       └── algorithm.py        # Vòng lặp GA
├── tests/
│   ├── test_operators.py       # Toán tử di truyền
│   ├── test_algorithm.py       # Vòng lặp GA, hàm thích nghi
│   ├── test_solver.py          # Tìm nhiều nghiệm
│   ├── test_transcendental.py  # Phương trình siêu việt
│   └── test_expression.py      # Xử lý biểu thức
├── experiments/                # Thực nghiệm, benchmark
├── docs/                       # Báo cáo, slide, hình ảnh
├── requirements.txt
└── pyproject.toml
```

## Công nghệ sử dụng

- **Python**: ngôn ngữ chính
- **NumPy**: tính toán số, vector hóa giải thuật di truyền
- **SciPy**: hàm đặc biệt, phương pháp đối chứng
- **Matplotlib**: trực quan hóa
- **Streamlit**: giao diện web
- **SymPy**: phân tích biểu thức toán học
- **Pytest**: kiểm thử

