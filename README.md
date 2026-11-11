# 🛟 Lifeguard Frame Labeler & YOWO Detection Pipeline

Bộ công cụ desktop (Tkinter, chạy trực tiếp trên Windows, không cần trình duyệt) phục vụ bài toán phát hiện đuối nước (**Lifeguard Drowning Detection**):

1. **Video Frame Extractor**: tách video thành chuỗi frame, cắt và phóng to vùng tuỳ ý.
2. **YOLO Pre-label**: dùng model YOLO (`labeler/models/best-yolo.pt`) gán nhãn tự động trước, người chỉ cần sửa lại.
3. **Frame Labeler**: gán / sửa nhãn bounding box `swim` / `drown` (hoặc `not_drowning` / `drowning`), xuất **CVAT XML 1.1** và **YOLO TXT**.
4. **Export YOLO**: đóng gói dataset đã gán nhãn thành bộ chuẩn Ultralytics (`images/`, `labels/`, `data.yaml`) để up Kaggle / train.
5. **YOWO**: huấn luyện mô hình spatio-temporal (3D CNN + 2D CNN) trên dữ liệu đã gán nhãn.

---

## 📑 Mục Lục
1. [Cài đặt](#-cài-đặt)
2. [Quy trình làm việc](#-quy-trình-làm-việc)
3. [Quy trình YOLO: video → nhãn → Kaggle](#-quy-trình-yolo-video--nhãn--kaggle)
4. [Tách frame từ video](#-tách-frame-từ-video-frame_extractor)
5. [Gán nhãn tự động bằng model](#-gán-nhãn-tự-động-bằng-model-labelerprelabelpy)
6. [Gán nhãn](#-gán-nhãn-labeler)
7. [Xuất dataset YOLO](#-xuất-dataset-yolo-labelerexport_yolopy)
8. [Huấn luyện YOWO](#-huấn-luyện-yowo)
9. [Định dạng đầu ra](#-định-dạng-đầu-ra)
10. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)

---

## 💻 Cài Đặt

Yêu cầu **Python 3.8+**:

```bash
pip install opencv-python pillow numpy          # Labeler + Extractor
pip install ultralytics                         # Gán nhãn tự động (prelabel.py)
pip install torch torchvision                   # Huấn luyện YOWO (tự dùng GPU CUDA nếu có)
```

---

## 🔁 Quy Trình Làm Việc

### Quy trình YOWO (dataset `archive/`, clip 224x224)

```text
Video (.mp4, .webm, ...)
   │  1. frame_extractor/run_extractor.bat  → cắt vùng, chọn đoạn, xuất frame
   ▼
archive/frames/<clip>/000001.jpg ...   (+ tên clip được thêm vào archive/unlabeled.csv)
   │  2. labeler/run_labeler.bat            → vẽ box swim/drown, nội suy, Ctrl+S
   ▼
archive/labels/<clip>.xml  +  archive/labels/<clip>/000001.txt ...
   │  3. Chuyển tên clip sang train.csv / valid.csv / test.csv
   ▼
python yowo/run_yowo.py    → yowo/weights/yowo_best.pth
```

---

## 🚀 Quy Trình YOLO: Video → Nhãn → Kaggle

Dùng cho dataset YOLO riêng (ví dụ `val_yolo/`): frame giữ nguyên khung hình, nhãn `drowning` / `not_drowning`.

```text
Video (.mp4)
   │  1. Tách frame toàn khung, 10 fps                  (frame_extractor)
   ▼
val_yolo/frames/<clip>/000001.jpg ...     + val_yolo/classes.txt
   │  2. Model gán nhãn trước                           (labeler/prelabel.py)
   ▼
val_yolo/labels/<clip>.xml + labels/<clip>/*.txt
   │  3. Người soát và sửa từng frame, Ctrl+S           (labeler/desktop_labeler.py)
   ▼
   │  4. Đóng gói chuẩn YOLO                            (labeler/export_yolo.py)
   ▼
val_yolo_export/images/val + labels/val + data.yaml  →  upload Kaggle
```

**Bước 0: tạo thư mục dataset và `classes.txt`.** Mỗi dòng là một class, số dòng (đếm từ 0) là `class_id` trong file TXT. Thứ tự phải **khớp với dataset bạn sẽ gộp vào** (ví dụ dataset Kaggle):
```text
not_drowning
drowning
```
→ `0 = not_drowning`, `1 = drowning`. Labeler và các tool đều đọc file này. Nếu không có file, mặc định là `0 = swim`, `1 = drown`.

**Bước 1: tách frame** (từ thư mục `unlabeled/`):
```bash
python frame_extractor/video_frame_extractor.py --video "video.mp4" --clip_name my_clip --size original --fps 10 --start 30 --end 125 --output_dir val_yolo/frames/my_clip
```
`--size original` giữ nguyên khung hình gốc (YOLO tự resize khi train). Video dài thì chọn đoạn có sự kiện bằng `--start` / `--end`.

**Bước 2: gán nhãn tự động.**
```bash
python labeler/prelabel.py --dataset val_yolo
```
Chỉ xử lý các clip **chưa có** `labels/<clip>.xml`, nên nhãn đã sửa không bị ghi đè.

**Bước 3: soát và sửa nhãn.**
```bash
python labeler/desktop_labeler.py --dataset val_yolo
```
Model hay **bỏ sót drowning** (người chìm, chỉ còn tay / đầu). Hãy phát (`Space`) toàn bộ clip, thêm box drowning còn thiếu, xoá box sai, rồi `Ctrl + S`.

**Bước 4: đóng gói.** Xoá thư mục export cũ trước để không lẫn file thừa:
```bash
python labeler/export_yolo.py --dataset val_yolo --out val_yolo_export --split val
```

**Bước 5: upload Kaggle.** Up thư mục `val_yolo_export/`, không phải `val_yolo/`. Nếu gộp vào dataset có sẵn, chỉ chép `images/<split>/` và `labels/<split>/` vào, giữ `data.yaml` của dataset đó. File TXT chỉ chứa số class, nên **thứ tự trong `classes.txt` phải khớp `names:` trong `data.yaml` của Kaggle**, nếu không nhãn sẽ bị đảo mà không báo lỗi.

---

## 🎬 Tách Frame Từ Video (`frame_extractor/`)

### Khởi động
- Nhấp đúp **`frame_extractor/run_extractor.bat`**, hoặc chạy `python frame_extractor/video_frame_extractor.py`.
- Hoặc bấm **`🎬 Tách Frame Từ Video...`** trong Frame Labeler.

### Các bước
1. Bấm **📁 Chọn Video...** để mở video.
2. **Chọn đoạn**: tua tới điểm đầu, bấm **`[ Đặt Start`** (phím `[`), tua tới điểm cuối, bấm **`Đặt End ]`** (phím `]`).
3. **Chọn vùng cắt** (khung màu xanh trên video):
   - Kéo chuột trái ra ngoài khung để vẽ vùng mới, kéo bên trong khung để di chuyển, kéo 8 tay nắm để chỉnh kích thước.
   - **Double-click** vào người cần theo dõi để đưa vùng cắt về giữa điểm đó.
   - Nút **Phóng to ×1.5 / ×2 / ×3 / ×4** (hoặc phím `+` / `-`) thu hẹp vùng cắt quanh tâm, nên ảnh xuất được phóng to tương ứng.
   - Nút nhanh: *Toàn khung*, *Giữa 640x640*, *Vuông lớn nhất*, *Lớn nhất theo tỉ lệ*.
   - Nhập chính xác **x, y, w, h** (pixel của video) rồi bấm *Áp dụng* hoặc Enter.
   - **Khoá tỉ lệ**: mặc định *Theo kích thước xuất*, tức vùng cắt luôn cùng tỉ lệ với ảnh xuất nên không bị méo.
4. **Kích thước xuất**: `640x640` (YOLO), `224x224` (YOWO), giữ nguyên vùng cắt, hoặc tuỳ chỉnh W x H. Bật *Giữ tỉ lệ (thêm viền đen)* nếu không muốn ảnh bị kéo giãn.
5. **Tần suất**: tất cả frame, mỗi N frame, hoặc theo FPS đích. Dòng *"Sẽ xuất N frame"* cho biết trước số ảnh.
6. **Tên clip**: chỉ dùng chữ không dấu, số, `_`, `-`. Ảnh được lưu vào `archive/frames/<tên_clip>/`.
7. Kiểm tra khung **👁 Xem trước ảnh xuất**, rồi bấm **🚀 BẮT ĐẦU TÁCH FRAME**.

> 💡 Khung xem trước hiện đúng ảnh sẽ lưu và hệ số phóng. Tool cảnh báo khi phóng quá ×2.5 (ảnh sẽ mờ) hoặc khi ảnh sẽ bị méo.

### Tạo nhiều clip một lần (hàng đợi)
Chỉnh vùng cắt, đoạn và tên, rồi bấm **➕ Thêm vào hàng đợi**. Tên clip tự tăng (`pool_01` → `pool_02`). Lặp lại cho vùng hoặc đoạn khác, hoặc mở video khác. Khi hàng đợi có job, nút chính đổi thành **🚀 CHẠY HÀNG ĐỢI** và chạy lần lượt tất cả.

### Lưu ý
- Nếu thư mục clip đã có ảnh, tool hỏi: **Yes** = xoá ảnh cũ và ghi lại từ `000001`, **No** = ghi tiếp số thứ tự, **Cancel** = huỷ.
- Mỗi clip có log `archive/extract_logs/<clip>.json` ghi video nguồn, vùng cắt, và frame gốc tương ứng với từng ảnh.
- Nếu không mở được video có tên tiếng Việt, hãy đổi tên file sang không dấu.

### Điều khiển (Extractor)

| Phím / Chuột | Chức năng |
| :--- | :--- |
| Lăn chuột / Chuột phải + kéo | Zoom khung xem / di chuyển ảnh |
| `Shift` + lăn chuột | Đổi frame |
| `F` / `Z` | Xem vừa khung / soi sát vùng cắt |
| `Space` | Phát / dừng |
| `A`,`←` / `D`,`→` | Lùi / tiến 1 frame (`Shift`: 10 frame), `Home` / `End`: đầu / cuối |
| `[` / `]` | Đặt Start / End |
| `+` / `-` | Phóng to / thu nhỏ vùng cắt |

### Chạy bằng dòng lệnh (không cần GUI)
```bash
# Cắt giữa 640x640, lấy 10 fps từ giây 5 đến giây 30, đăng ký vào unlabeled.csv
python frame_extractor/video_frame_extractor.py --video pool.mp4 --clip_name pool_01 --center_640 --fps 10 --start 5 --end 30 --add_unlabeled

# Cắt vùng x=800,y=300 kích thước 640x640, phóng to x2 quanh tâm, xuất 224x224, ghi đè thư mục cũ
python frame_extractor/video_frame_extractor.py --video pool.mp4 --clip_name pool_02 --crop 800,300,640,640 --zoom 2 --size 224x224 --overwrite
```

| Tham số | Ý nghĩa |
| :--- | :--- |
| `--crop x,y,w,h` / `--center_640` | Vùng cắt. Nếu bỏ trống, tool lấy vùng lớn nhất đúng tỉ lệ ảnh xuất |
| `--zoom F` | Phóng to F lần quanh tâm vùng cắt |
| `--size WxH` / `original` | Kích thước xuất (mặc định `640x640`) |
| `--keep_aspect` | Thêm viền đen thay vì kéo giãn |
| `--step N` / `--fps F` | Lấy mỗi N frame / theo FPS đích |
| `--start S` / `--end S` | Đoạn cần lấy (giây) |
| `--format jpg\|png`, `--quality Q` | Định dạng / chất lượng JPG |
| `--overwrite` / `--append` | Bắt buộc chọn một khi thư mục đã có ảnh |
| `--add_unlabeled` | Thêm tên clip vào `archive/unlabeled.csv` |
| `--output_dir DIR` | Thư mục xuất khác (mặc định `archive/frames/<clip>`) |

---

## 🤖 Gán Nhãn Tự Động Bằng Model (`labeler/prelabel.py`)

Chạy model YOLO lên mọi frame và ghi kết quả thành file nhãn của labeler, để người chỉ cần sửa thay vì vẽ từ đầu.

```bash
python labeler/prelabel.py --dataset val_yolo                         # dùng labeler/models/best-yolo.pt
python labeler/prelabel.py --dataset val_yolo --clips my_clip --conf 0.2
```
Hoặc nhấp đúp **`labeler/run_prelabel.bat`** và nhập tên thư mục dataset.

- Đọc `<dataset>/frames/<clip>/`, ghi `<dataset>/labels/<clip>.xml` (CVAT XML) và `labels/<clip>/*.txt` (YOLO TXT, không ghi với dataset dạng repo có `datalist.csv`).
- Class của model được ghép **theo tên**, không theo số: `drowning`/`drown` → drown, `not_drowning`/`swim` → swim. Số class ghi ra theo `classes.txt` của dataset, nên model và dataset có thứ tự khác nhau vẫn đúng.
- Một người bị nhận cả 2 class (IoU ≥ `--overlap`, mặc định 0.4) thì giữ box drowning.
- Clip đã có XML bị **bỏ qua**. Muốn làm lại thì thêm `--overwrite` (XML cũ được giữ thành `<clip>.xml.bak`).

| Tham số | Ý nghĩa |
| :--- | :--- |
| `--dataset DIR` | Thư mục dataset (chứa `frames/`) |
| `--model PT` | File weights YOLO, mặc định `labeler/models/best-yolo.pt` (0 = drowning, 1 = not_drowning) |
| `--clips A B` | Chỉ chạy các clip này |
| `--conf` / `--iou` / `--imgsz` | Ngưỡng tin cậy (0.25) / NMS IoU (0.5) / kích thước suy luận (640) |
| `--overlap` | Ngưỡng bỏ box swim trùng box drown (0 = tắt) |
| `--device` | `cpu`, `0`, ... (mặc định tự chọn) |
| `--overwrite` | Gán lại cả clip đã có nhãn |

> ⚠️ Nhãn do model gán chỉ là bản nháp. Luôn soát lại trong labeler trước khi dùng để train / đánh giá.

---

## 📝 Gán Nhãn (`labeler/`)

### Khởi động
Nhấp đúp **`labeler/run_labeler.bat`**, hoặc chạy `python labeler/desktop_labeler.py`. Cửa sổ ứng dụng mở trực tiếp, không qua trình duyệt.

### Các bước
1. **Chọn clip** ở cột trái. Tab **Unlabeled** là các clip trong `unlabeled.csv` (⏳ chưa gán, ✅ đã có nhãn). Tab **All Clips** gồm cả train / valid / test. Clip vừa tách xong chưa hiện thì bấm 🔄.
2. **Chọn class**: `1` = **swim** (hồng), `2` = **drown** (vàng).
3. **Vẽ box**: kéo chuột trái trên vùng trống. Click một box để chọn nó, kéo box để di chuyển, kéo tay nắm trắng để chỉnh kích thước. Khi đang chọn box, bấm `1` / `2` để đổi class của box đó.
4. **Nội suy** để không phải vẽ từng frame:
   - Vẽ box ở frame A (keyframe), tua tới frame B và vẽ box ở vị trí mới.
   - Quay lại frame A, bấm `I`: các frame giữa A và B được tự động điền box.
   - Hoặc vẽ nhiều keyframe rồi bấm `Shift + I` để nội suy toàn clip một lần.
   - Hỗ trợ nhiều người trong một frame: box được ghép cặp theo vị trí gần nhất và cùng class.
   - Box nội suy hiển thị nét đứt. Kéo sửa một box nội suy thì nó thành keyframe.
5. **Kiểm tra**: bấm `Space` để phát xem box có bám đúng người không. Thanh timeline dưới slider đánh dấu các frame có nhãn: vạch cao là keyframe, vạch thấp là frame nội suy, vàng là có `drown`.
6. **Lưu**: `Ctrl + S`. Nếu đổi clip hoặc đóng app khi chưa lưu, labeler sẽ hỏi.

> Labeler tự đọc kích thước thật của frame (224, 640, ...), nên toạ độ nhãn luôn đúng với ảnh.

Ô **Hiện nhãn Swim** nằm dưới các nút chọn class ở cột phải. Mặc định tắt
để ẩn số thứ tự và tên `swim/not_drowning` trên ảnh, chỉ giữ box màu.
Nhãn `drown/drowning` luôn hiển thị. Bật ô này để hiện cả nhãn Swim.
Labeler không lưu điểm tin cậy nên không có phần trăm để hiển thị.
Ẩn chữ vẫn cho phép chọn, kéo và resize box, không thay đổi dữ liệu nhãn.

### Gán nhãn cho `dataset_lifeguard` (dataset của repo drowning-detection)
Đặt thư mục `dataset_lifeguard/` (gồm `frames/`, `labels/`, `datalist.csv`) cạnh thư mục `labeler/`. Khi thư mục này tồn tại, labeler mở nó mặc định thay cho `archive/`.

- Dataset này không có file `.csv` chia tập, nên labeler quét trực tiếp `frames/`: clip chưa có `labels/<clip>.xml` nằm ở tab **Unlabeled**, tab **All Clips** gồm cả clip đã có nhãn.
- `Ctrl + S` chỉ ghi `dataset_lifeguard/labels/<clip>.xml` (CVAT XML 1.1, class `swim` / `drown`, `id` bắt đầu từ 0), cùng định dạng với các file nhãn có sẵn trong repo. Không tạo YOLO TXT.
- Đổi dataset: bấm **📁** cạnh nút 🔄 và chọn thư mục chứa `frames/`, hoặc chạy `python labeler/desktop_labeler.py --dataset archive`.
- Chọn lại cùng thư mục vẫn quét lại clip, xoá ô tìm kiếm và chuyển sang danh sách tất cả. Nếu chưa mở clip nào, tool mở clip đầu tiên; các chỉnh sửa chưa lưu trong clip hiện tại vẫn được giữ.

### Thêm một bộ frame có sẵn (từ thư mục ảnh)
Bấm **➕ Thêm bộ frame từ thư mục...** ở cột trái, chọn thư mục chứa ảnh (`.jpg/.jpeg/.png/.bmp/.webp`), đặt tên clip và chọn tập (`unlabeled` / `train` / `valid` / `test`).

- Ảnh được chép vào `archive/frames/<tên>/` và đổi tên thành `000001.jpg`, `000002.jpg`, ... theo thứ tự tự nhiên của tên file (`frame_2` đứng trước `frame_10`).
- Mặc định resize về **224x224** cho giống dataset gốc (YOWO chia toạ độ box cho 224). Bỏ tick nếu muốn giữ kích thước gốc.
- Tên clip được thêm vào file `.csv` của tập đã chọn; clip vừa thêm được mở ngay để gán nhãn.

### Crop trực tiếp khi phát, chỉ cập nhật frame và nhãn

1. Chọn clip rồi bấm **✂ Crop bộ frame**. Kéo trên chính khung hình đang phát
   để chọn vùng **giữ lại**; phần ngoài vùng được làm tối.
2. Có thể Play/tạm dừng, tua hoặc đổi frame để kiểm tra cùng một vùng crop trên cả clip.
   Kéo trong vùng để di chuyển, kéo tay nắm để đổi cỡ, **Ctrl + kéo** để vẽ lại.
   **Esc** hoặc **Thoát chế độ crop** bỏ lựa chọn mà chưa thay đổi dữ liệu.
3. Bấm **Crop frame và nhãn**. Toàn bộ frame và XML/TXT được cập nhật
   dưới tên hiện tại. Chỉ giữ box nằm trọn trong vùng; box ngoài vùng hoặc bị cắt
   dù chỉ một phần đều bị xoá toàn bộ. Box vừa sát biên nhưng không vượt biên vẫn giữ.
   Box nhỏ còn nằm trọn trong vùng vẫn giữ nhãn. Nhãn đang sửa chưa Ctrl+S cũng
   được dùng. Không tạo thêm clip `_crop` trong danh sách.

Box bị cắt sẽ mất nhãn dù còn thấy một phần người ở mép ảnh; nên chọn biên crop
tránh cắt ngang người khi có thể.

Video nguồn luôn được giữ nguyên; không cần chọn MP4 hoặc cài FFmpeg để crop frame.
Ảnh không resize; chiều rộng/cao vùng vẫn được làm tròn xuống số chẵn như trước.
Khi thành công, tool nạp lại đúng clip và vị trí frame đang xem.

Bản trước crop được lưu tại `_crop_backups/<clip>/<thời_điểm>/`, gồm `frames`, `xml`,
`txt`, `log` nếu trước đó có. `backup.json` ghi đường
dẫn gốc; `annotations_before_crop.xml` còn giữ cả nhãn trong bộ nhớ trước khi cắt.
Backup nằm ngoài thư mục frame/ảnh được export để train. Nếu xử lý lỗi trước khi
hoàn tất, tool khôi phục dữ liệu hiện tại từ backup.

Nếu train bằng dataset YOLO đã export trước đó, chạy lại export để cập nhật ảnh/nhãn
đã crop, ví dụ từ thư mục cha `tool/`:

```powershell
python tool/labeler/export_yolo.py --dataset datasets/val_yolo_review --out datasets/val_yolo_export --split val --clips H35eyL2M-ng
```

Kiểm tra chức năng cắt: `python -B -m unittest discover -s tool/labeler -p test_clip_crop.py -v`.

### Xoá cùng một người qua nhiều frame, có xem trước

**Delete** vẫn chỉ xoá box đang chọn ở frame hiện tại. Để xoá theo chuỗi:

1. Chọn box, bấm **Xoá theo chuỗi…**. Có thể giới hạn **Đến frame** rồi bấm **Tìm lại**.
2. Play/tua trên màn hình chính để xem các box viền xanh đứt và chữ **SẼ XOÁ**.
   Đây chỉ là xem trước, chưa đổi nhãn. Bấm **Huỷ/Esc** để thoát, hoặc **Xoá N box đã xem**
   để áp dụng. Các thao tác sửa nhãn khác tạm khoá trong lúc xem trước.
3. **Ctrl+Z** hoàn tác cả chuỗi trong một bước; **Ctrl+S** lưu kết quả.

Nếu có `track_id`, tool dùng ID để nối box. Nếu không có ID (như bộ review hiện tại),
tool chỉ đề xuất theo vị trí, độ chồng lấp, kích thước và chỉ xét box cùng class.
Nó dừng ở frame thiếu box, không có ứng viên đủ gần hoặc nhiều ứng viên khó phân biệt;
không nối qua khoảng trống. Việc ghép theo hình học vẫn có thể nhầm người, nên cần
soát chuỗi trước khi áp dụng. Với đoạn đông người/che khuất, xoá thủ công hoặc xử lý
từng chuỗi ngắn. Chức năng chỉ xoá nhãn, không xoá hình người khỏi ảnh.

### Xoá một bộ frame (clip)
Chọn clip trong danh sách rồi bấm **🗑️ Xoá bộ frame đang chọn...** ở cuối cột trái, hoặc **chuột phải** vào clip và chọn **Xoá bộ frame...**. Hộp thoại xác nhận cho biết số frame, clip đã có nhãn chưa, và các file `.csv` sẽ bị gỡ tên clip.

- Mặc định, thư mục frame, nhãn XML/TXT và log tách frame được **chuyển vào `archive/_trash/<clip>_<thời gian>/`**, tên clip bị gỡ khỏi `unlabeled.csv` / `train.csv` / `valid.csv` / `test.csv`.
- Bỏ tick *Xoá cả nhãn* để giữ lại file nhãn.
- Tick *Xoá vĩnh viễn* để xoá hẳn, không qua thùng rác.
- **Khôi phục**: chuyển `_trash/<...>/frames` về `archive/frames/<clip>/`, `<clip>.xml` về `archive/labels/`, `labels_txt` về `archive/labels/<clip>/`, rồi thêm lại tên clip vào file `.csv`.
- Thùng rác không tự dọn. Khi chắc chắn không cần nữa, hãy xoá thư mục `archive/_trash/` để giải phóng ổ đĩa.

Chuột phải vào clip còn có mục **📂 Mở thư mục frame**.

### Điều khiển (Labeler)

| Phím / Chuột | Chức năng |
| :--- | :--- |
| Kéo chuột trái (vùng trống) | Vẽ box mới |
| Kéo chuột trái trên box / tay nắm | Di chuyển / chỉnh kích thước box |
| `Ctrl` + kéo | Vẽ box mới đè lên box có sẵn |
| Lăn chuột / Chuột phải + kéo | Zoom tại con trỏ / di chuyển ảnh |
| `Shift` + lăn chuột | Đổi frame |
| `F` / `Z` | Vừa khung / zoom vào box đang chọn |
| `1` / `2` | Class **swim** / **drown** (đổi luôn box đang chọn) |
| `A`,`←` / `D`,`→` | Lùi / tiến 1 frame (`Shift`: 10 frame), `Home` / `End`: đầu / cuối |
| `Space` | Phát / dừng |
| `C` | Copy box sang frame kế tiếp và chuyển tới frame đó |
| `I` / `Shift + I` | Nội suy tới keyframe kế / nội suy toàn clip |
| `Tab` / `Esc` | Chọn box kế tiếp / bỏ chọn |
| `Delete` / `Backspace` | Xoá box đang chọn |
| `X` / `Shift + Del` | Xoá toàn bộ box của frame |
| `Ctrl + Z` / `Ctrl + Y` | Undo / Redo |
| `Ctrl + S` | Lưu XML + YOLO TXT |

Phím tắt không hoạt động khi con trỏ đang nằm trong ô nhập chữ (ô tìm kiếm, tên clip...). Click vào ảnh để dùng lại phím tắt.

---

## 📦 Xuất Dataset YOLO (`labeler/export_yolo.py`)

```bash
python labeler/export_yolo.py --dataset val_yolo --out val_yolo_export --split val
```
Tạo bộ dataset chuẩn Ultralytics:
```text
val_yolo_export/
├── images/val/<clip>_000001.jpg
├── labels/val/<clip>_000001.txt
└── data.yaml        # names theo classes.txt, không ghi đường dẫn tuyệt đối nên chạy được trên Kaggle
```
- Chỉ xuất clip có `labels/<clip>.xml`. Frame không có box vẫn có file TXT rỗng (ảnh nền).
- `--clips A B` để chỉ xuất một số clip, `--split train|val|test` để chọn thư mục tập.

---

## 🧠 Huấn Luyện YOWO

```bash
python yowo/run_yowo.py --epochs 5 --batch_size 8
```

- Dữ liệu huấn luyện là các clip trong `archive/train.csv` và `archive/valid.csv` (mỗi dòng một tên clip), đọc nhãn từ `archive/labels/<clip>.xml`.
- Mỗi mẫu là clip 8 frame liên tiếp. Loss gồm Box + Objectness BCE + Classification BCE.
- Trọng số tốt nhất được lưu vào `yowo/weights/yowo_best.pth`.
- Clip mới gán nhãn xong nằm trong `unlabeled.csv`. Muốn dùng để huấn luyện, hãy chuyển tên clip sang `train.csv` / `valid.csv` / `test.csv`.

> ⚠️ `yowo/dataset.py` hiện chuẩn hoá toạ độ box theo 224 px. Clip dùng cho YOWO nên được xuất ở **224x224** (tuỳ chọn "224 x 224" khi tách frame). Clip 640x640 vẫn dùng tốt cho YOLO nhờ file TXT đã chuẩn hoá [0, 1].

---

## 📦 Định Dạng Đầu Ra

### 1. CVAT XML 1.1 (`archive/labels/<clip>.xml`)
`id` của `<image>` bắt đầu từ 0 (frame `000001.jpg` có `id="0"`). `original_size` là kích thước thật của frame.
```xml
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <name>0xP3fXlLo0o_00-00-00--00-01-05_0-06--0-06--0-87.webm</name>
      <size>1976</size>
      <mode>interpolation</mode>
      <labels>
        <label><name>swim</name><color>#ff00cc</color></label>
        <label><name>drown</name><color>#fafa37</color></label>
      </labels>
      <original_size><width>224</width><height>224</height></original_size>
    </task>
  </meta>
  <image id="50" name="frame_000050" width="224" height="224">
    <box label="swim" source="manual" occluded="0" xtl="71.00" ytl="140.00" xbr="150.00" ybr="200.00" z_order="0"/>
  </image>
</annotations>
```
`source="manual"` là box vẽ tay, `source="interpolation"` là box do nội suy.

### 2. YOLO TXT (`archive/labels/<clip>/000051.txt`)
Mỗi frame một file, đánh số giống tên ảnh (`000051.jpg` ↔ `000051.txt`, tương ứng `id="50"` trong XML). Toạ độ chuẩn hoá [0, 1], dùng được với YOLOv5 / YOLOv8:
```text
<class_id> <x_center> <y_center> <width> <height>
0 0.493304 0.758929 0.352679 0.267857
```
`class_id`: mặc định `0` = swim, `1` = drown. Dataset có `classes.txt` thì theo thứ tự dòng trong file đó. Frame không có box vẫn có file rỗng.

---

## 📂 Cấu Trúc Thư Mục

```text
unlabeled/
├── archive/                       # Dataset dùng chung cho cả 3 phần
│   ├── frames/<clip>/000001.jpg ...   # Frame của từng clip
│   ├── labels/<clip>.xml              # Nhãn CVAT XML
│   ├── labels/<clip>/000001.txt ...   # Nhãn YOLO TXT
│   ├── extract_logs/<clip>.json       # Log tách frame (video nguồn, vùng cắt, frame gốc)
│   ├── _trash/                        # Clip đã xoá (khôi phục được)
│   ├── train.csv / valid.csv / test.csv   # Danh sách clip theo tập (74 / 8 / 9 clip)
│   └── unlabeled.csv                  # Danh sách clip cần gán nhãn
│
├── frame_extractor/               # 🎬 Tool tách video thành frame (độc lập)
│   ├── video_frame_extractor.py   #    GUI + CLI: cắt / phóng to vùng, chọn đoạn, hàng đợi
│   ├── zoom_canvas.py             #    Zoom / pan khung xem
│   └── run_extractor.bat          #    Nhấp đúp để chạy
│
├── labeler/                       # 📝 Tool gán nhãn (độc lập)
│   ├── desktop_labeler.py         #    Ứng dụng gán nhãn
│   ├── annotation_io.py           #    Đọc / ghi CVAT XML 1.1 và YOLO TXT, đọc classes.txt
│   ├── prelabel.py                #    Gán nhãn tự động bằng model YOLO
│   ├── export_yolo.py             #    Xuất dataset chuẩn Ultralytics (images/labels/data.yaml)
│   ├── models/best-yolo.pt        #    Model YOLO dùng cho prelabel (0 = drowning, 1 = not_drowning)
│   ├── zoom_canvas.py             #    Zoom / pan khung xem
│   ├── run_labeler.bat            #    Nhấp đúp để chạy labeler
│   └── run_prelabel.bat           #    Nhấp đúp để gán nhãn tự động
│
├── yowo/                          # 🧠 Huấn luyện YOWO (tách riêng)
│   ├── run_yowo.py                #    CLI huấn luyện
│   ├── train_yowo.py, dataset.py  #    Vòng lặp train, PyTorch Dataset (clip 8 frame)
│   ├── models/yowo.py             #    Kiến trúc 3D CNN + 2D CNN + CFAM + Detection Head
│   ├── utils/annotation_io.py     #    Bản sao đọc nhãn XML
│   └── weights/yowo_best.pth      #    Trọng số sau huấn luyện
│
├── val_yolo/                      # Dataset YOLO (frames/, labels/, classes.txt), chỉ ở local
├── val_yolo_export/               # Bản đóng gói để up Kaggle, chỉ ở local
└── video/                         # Video gốc để tách frame
```

> Repo GitHub chỉ chứa `frame_extractor/`, `labeler/` (gồm cả model `labeler/models/best-yolo.pt`, ~20 MB) và README. Thư mục `archive/` (dataset), `video/` và `yowo/` chỉ có ở máy local. Sau khi clone, hãy tạo thư mục `archive/` cạnh 2 thư mục tool: tool tách frame sẽ tự tạo `archive/frames/` và `archive/unlabeled.csv`.

Mỗi thư mục tool chạy độc lập, không import code của nhau, và luôn đọc/ghi `archive/` ở thư mục gốc (cùng cấp với thư mục tool), dù chạy lệnh từ đâu. Khi copy tool sang máy khác, giữ nguyên cấu trúc `<thư_mục_gốc>/archive` + `<thư_mục_gốc>/<tool>/`.
