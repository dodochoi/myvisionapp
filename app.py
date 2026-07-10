"""
app.py
-------
결함 탐지 데모용 Streamlit 앱 (단일 파일 버전).

모델 로드 / 추론 / 바운딩박스 시각화 로직과 Streamlit UI를 한 파일에 담았습니다.

로컬 실행:
    streamlit run app.py

동작:
1. 사이드바에서 학습된 모델(best.pt) 경로와 confidence threshold를 설정
2. 이미지를 업로드하면 predict_and_draw()로 추론
3. 원본 / 탐지 결과 이미지를 나란히 보여주고, 상세 결과 표를 출력
"""
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 결함 탐지 유틸리티 (모델 로드 / 추론 / 바운딩박스 시각화)
# ---------------------------------------------------------------------------

def imread(path, flags=cv2.IMREAD_COLOR):
    """한글 경로에서도 안전하게 이미지를 읽는 함수 (cv2.imread 대체)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path, image):
    """한글 경로에서도 안전하게 이미지를 저장하는 함수 (cv2.imwrite 대체)."""
    ext = Path(path).suffix if Path(path).suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(str(path))
    return ok


def load_model(weights_path):
    """학습된 best.pt(또는 다른 .pt) 가중치를 불러옵니다."""
    weights_path = str(weights_path)
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"모델 가중치를 찾을 수 없습니다: {weights_path}")
    return YOLO(weights_path)


def draw_boxes(image, boxes, confs, class_ids, class_names, color=(0, 0, 255), thickness=2):
    """
    image      : BGR numpy array
    boxes      : (N, 4) xyxy 좌표 배열
    confs      : (N,) confidence 배열
    class_ids  : (N,) 클래스 id 배열
    class_names: {id: name} 딕셔너리 (model.names)
    """
    annotated = image.copy()
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, class_ids):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{class_names.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 2, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return annotated


def predict_and_draw(model, image, conf=0.05, imgsz=320):
    """
    새 이미지에 대해 결함을 탐지하고 바운딩박스를 그려서 반환합니다.

    Parameters
    ----------
    model : YOLO
        load_model()으로 불러온 모델
    image : str | Path | np.ndarray
        이미지 경로 또는 BGR numpy array
    conf  : float
        confidence threshold
    imgsz : int
        추론 이미지 크기

    Returns
    -------
    annotated_bgr : np.ndarray
        바운딩박스가 그려진 BGR 이미지
    result : ultralytics.engine.results.Results
        원본 추론 결과 객체
    summary : list[dict]
        [{"class": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...]
    """
    if isinstance(image, (str, Path)):
        image_bgr = imread(image)
        if image_bgr is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image}")
    else:
        image_bgr = image

    results = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]
    class_names = model.names

    if len(result.boxes) == 0:
        return image_bgr.copy(), result, []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    annotated_bgr = draw_boxes(image_bgr, boxes, confs, class_ids, class_names)

    summary = [
        {
            "class": class_names.get(int(c), str(int(c))),
            "confidence": round(float(cf), 4),
            "bbox": [round(float(v), 1) for v in box],
        }
        for box, cf, c in zip(boxes, confs, class_ids)
    ]
    return annotated_bgr, result, summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="결함 탐지 데모", page_icon="🔍", layout="wide")

st.title("🔍 결함 탐지 데모 (YOLOv8)")
st.caption("학습된 YOLOv8 모델로 이미지 속 결함을 탐지하고 바운딩박스로 표시합니다.")

with st.sidebar:
    st.header("⚙️ 설정")
    weights_path = st.text_input(
        "모델 가중치 경로 (best.pt)",
        value="model/best.pt",
        help="레포지토리 기준 상대경로 또는 절대경로를 입력하세요.",
    )
    conf_threshold = st.slider("Confidence threshold", min_value=0.01, max_value=0.9, value=0.05, step=0.01)
    imgsz = st.selectbox("이미지 크기 (imgsz)", options=[320, 416, 640], index=0)

    st.divider()
    st.caption(
        "model/best.pt 파일이 커서 GitHub에 올리기 어렵다면 "
        "Git LFS 또는 GitHub Release/외부 스토리지 URL 다운로드 방식을 권장합니다. "
        "자세한 내용은 README를 참고하세요."
    )


@st.cache_resource(show_spinner="모델을 불러오는 중...")
def get_model(path: str):
    return load_model(path)


uploaded_file = st.file_uploader("검사할 이미지를 업로드하세요", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    if not Path(weights_path).exists():
        st.error(f"모델 가중치를 찾을 수 없습니다: `{weights_path}`\n\n사이드바에서 경로를 다시 확인해주세요.")
    else:
        model = get_model(weights_path)

        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("이미지를 디코딩할 수 없습니다. 다른 파일을 시도해주세요.")
        else:
            with st.spinner("결함을 탐지하는 중..."):
                annotated_bgr, _, summary = predict_and_draw(
                    model, image_bgr, conf=conf_threshold, imgsz=imgsz
                )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("원본 이미지")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("탐지 결과")
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("📋 탐지 상세 결과")
            if summary:
                st.table(summary)
            else:
                st.info("탐지된 결함이 없습니다. 왼쪽 사이드바에서 threshold를 낮춰보세요.")
else:
    st.info("왼쪽 사이드바에서 모델 경로를 설정한 뒤, 이미지를 업로드하면 결과가 표시됩니다.")
