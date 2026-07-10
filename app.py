"""
app.py
-------
결함 탐지 데모용 Streamlit 앱 (단일 파일 버전 + Gemini AI 해석 기능 추가).

로컬 실행:
    streamlit run app.py
"""
from pathlib import Path
import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
# 2026년 기준 공식 최신 google-genai 라이브러리를 사용합니다.
from google import genai
from google.genai import types

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
    """바운딩 박스 시각화 함수"""
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
    """새 이미지에 대해 결함을 탐지하고 바운딩박스를 그려서 반환합니다."""
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
# Gemini API 분석 연동 로직
# ---------------------------------------------------------------------------

def analyze_defects_with_gemini(summary_data):
    """YOLO 탐지 결과를 바탕으로 Gemini 모델에게 종합 분석 리포트를 요청합니다."""
    # Streamlit Secret 또는 환경 변수에서 API 키를 가져옵니다.
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return "⚠️ Gemini API 키가 설정되지 않았습니다. 사이드바안내 또는 Streamlit secrets 설정을 확인해주세요."

    try:
        # 최신 google-genai SDK 가이드라인 준수
        client = genai.Client(api_key=api_key)
        
        # 프롬프트 조립
        prompt = f"""
        당신은 제조 공정 전문 품질 관리(QC) 분석가입니다. 
        비전 AI 모델(YOLO)이 제품 표면이나 외관에서 탐지한 결함 데이터 리스트를 바탕으로 전문가용 '결함 종합 분석 리포트'를 작성해 주세요.

        [탐지 데이터 리스트]
        {summary_data}

        [요청 사항]
        1. 발견된 결함들의 유형과 개수 요약
        2. 신뢰도(Confidence)와 위치를 고려할 때 심각성 판단 (예: 특정 구역에 결함 밀집 여부 등)
        3. 현장 작업자나 품질 검사원이 취해야 할 추천 조치 사항 (출하 중지, 재작업, 공정 설비 점검 등)
        
        친절하고 전문적인 한국어로 작성해 주고, 가독성 좋게 마크다운 형식을 활용해 주세요.
        """

        response = client.models.generate_content(
            model='gemini-flash-latest', # 범용적이고 빠른 속도의 2.5-flash 모델 권장
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini API 요청 중 오류가 발생했습니다: {str(e)}"


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="결함 탐지 및 AI 분석 데모", page_icon="🔍", layout="wide")

st.title("🔍 결함 탐지 및 AI 분석 데모")
st.caption("YOLOv8 모델로 결함을 탐지한 뒤, Gemini AI를 활용해 진단 및 조치 사항을 실시간으로 분석합니다.")

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
    st.markdown("### 🔑 API 설정")
    st.caption(
        "Gemini 분석 기능을 사용하려면 `GEMINI_API_KEY`를 설정해야 합니다. "
        "로컬 구동 시 프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래와 같이 입력하세요:\n\n"
        "`GEMINI_API_KEY = \"YOUR_KEY_HERE\"`"
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

            # 1. 이미지 결과 나란히 배치
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("원본 이미지")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("탐지 결과")
                st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            # 2. 하단 레이아웃 분할: 왼쪽(YOLO 상세 테이블), 오른쪽(Gemini AI 분석)
            st.divider()
            analysis_col1, analysis_col2 = st.columns([1, 1])

            with analysis_col1:
                st.subheader("📋 YOLO 탐지 상세 데이터")
                if summary:
                    st.table(summary)
                else:
                    st.info("탐지된 결함이 없습니다. 왼쪽 사이드바에서 threshold를 낮춰보세요.")

            with analysis_col2:
                st.subheader("🤖 Gemini AI 품질 리포트")
                if summary:
                    with st.spinner("Gemini가 결함 데이터를 분석하고 조치 사항을 생성하는 중..."):
                        ai_report = analyze_defects_with_gemini(summary)
                        st.markdown(ai_report)
                else:
                    st.success("✅ 탐지된 데이터가 없어 AI 종합 진단이 불필요합니다. 정상 제품으로 판단됩니다.")
else:
    st.info("왼쪽 사이드바에서 모델 경로를 설정한 뒤, 이미지를 업로드하면 결과가 표시됩니다.")
