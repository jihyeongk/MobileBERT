import re
import pandas as pd
from pathlib import Path


# ==================================================
# 1. 경로 및 기본 설정
# ==================================================

# PyCharm에서는 현재 파이썬 파일이 있는 폴더,
# Jupyter에서는 현재 작업 폴더를 프로젝트 폴더로 사용
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd()

# 원본 CSV 파일 폴더
DATA_DIR = PROJECT_DIR / "data"

# 결과 저장 폴더
LABEL_DIR = PROJECT_DIR / "label"
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# 같은 데이터를 다시 추출하기 위한 난수 고정
RANDOM_STATE = 42

# 제품별, 별점별 추출 개수
# 5개 제품 × 4개 별점 × 150건 = 3,000건
SAMPLE_PER_RATING = 150

# 리뷰 본문 최소 단어 조건
# 10단어 초과이므로 실제로는 11단어 이상
MIN_REVIEW_WORDS = 10


# ==================================================
# 2. 제품별 CSV 파일 설정
# ==================================================

PRODUCTS = [
    {
        "product_id": 16567,
        "product_name": "Magnesium",
        "all_file": "iherb_reviews_16567_mg.csv",
        "negative_file": "iherb_reviews_16567_rating_1_2.csv"
    },
    {
        "product_id": 61865,
        "product_name": "VitaminC",
        "all_file": "iherb_reviews_61865_VitaminC.csv",
        "negative_file": "iherb_reviews_61865_VitaminC_rating_1_2.csv"
    },
    {
        "product_id": 62118,
        "product_name": "Omega",
        "all_file": "iherb_reviews_62118_omega.csv",
        "negative_file": "iherb_reviews_62118_omega_rating_1_2.csv"
    },
    {
        "product_id": 64009,
        "product_name": "LactoBif",
        "all_file": "iherb_reviews_64009_lactobif.csv",
        "negative_file": "iherb_reviews_64009_lactobif_rating_1_2.csv"
    },
    {
        "product_id": 67051,
        "product_name": "VeganVitaB",
        "all_file": "iherb_reviews_67051_Vegan_VitaB.csv",
        "negative_file": "iherb_reviews_67051_Vegan_VitaB_rating_1_2.csv"
    }
]


# ==================================================
# 3. 영어 데이터 확인 함수
# ==================================================

def is_english_text(value):
    """
    제목 또는 본문이 영어 데이터인지 확인한다.

    조건:
    1. NULL이 아니어야 한다.
    2. 빈 문자열이 아니어야 한다.
    3. 영어 알파벳이 최소 한 글자 이상 있어야 한다.
    4. 한글, 중국어, 일본어, 이모지 등의
       비ASCII 문자가 없어야 한다.
    """

    if pd.isna(value):
        return False

    text = str(value).strip()

    if text == "":
        return False

    # 영어 알파벳이 최소 한 글자 이상 있어야 함
    if re.search(r"[A-Za-z]", text) is None:
        return False

    # 한글, 중국어, 일본어, 이모지 등 제외
    if not text.isascii():
        return False

    return True


# ==================================================
# 4. CSV 파일 정리 함수
# ==================================================

def load_and_clean_csv(file_path):
    """
    CSV 파일을 불러오고 다음 조건으로 정리한다.

    - 별점, 제목, 본문 NULL 제거
    - 빈 제목과 빈 본문 제거
    - 제목과 본문이 모두 영어인 리뷰만 유지
    - 본문이 10단어를 초과하는 리뷰만 유지
    - 중복된 리뷰 본문 제거
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    original_count = len(df)

    # 필요한 열 확인
    required_columns = [
        "review_title",
        "review_text",
        "rating"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name}에 필요한 열이 없습니다: "
            f"{missing_columns}"
        )

    # 별점을 숫자로 변환
    # 변환할 수 없는 값은 NaN으로 처리
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # 별점, 제목, 본문 중 하나라도 NULL이면 제거
    df = df.dropna(
        subset=[
            "rating",
            "review_title",
            "review_text"
        ]
    ).copy()

    # 제목과 본문을 문자열로 변환하고 양쪽 공백 제거
    df["review_title"] = (
        df["review_title"]
        .astype(str)
        .str.strip()
    )

    df["review_text"] = (
        df["review_text"]
        .astype(str)
        .str.strip()
    )

    # 제목 또는 본문이 빈 문자열이면 제거
    df = df[
        (df["review_title"] != "")
        & (df["review_text"] != "")
    ].copy()

    # 제목과 본문이 모두 영어인 데이터만 유지
    df = df[
        df["review_title"].apply(is_english_text)
        & df["review_text"].apply(is_english_text)
    ].copy()

    # 리뷰 본문의 단어 수 계산
    df["word_count"] = (
        df["review_text"]
        .str.split()
        .str.len()
    )

    # 본문이 10단어를 초과하는 데이터만 유지
    # 즉, 11단어 이상
    df = df[
        df["word_count"] > MIN_REVIEW_WORDS
    ].copy()

    # 같은 리뷰 본문이 여러 번 존재하면 하나만 유지
    df = df.drop_duplicates(
        subset=["review_text"]
    ).reset_index(drop=True)

    # 별점을 정수로 변환
    df["rating"] = df["rating"].astype(int)

    removed_count = original_count - len(df)

    print(
        f"{file_path.name}: "
        f"원본 {original_count:,}건 → "
        f"필터링 후 {len(df):,}건 "
        f"({removed_count:,}건 제거)"
    )

    return df


# ==================================================
# 5. 특정 별점 데이터 추출 함수
# ==================================================

def sample_rating_data(
    df,
    rating,
    sample_size,
    label,
    sentiment,
    random_state
):
    """
    특정 별점 데이터를 정해진 개수만큼 무작위로 추출하고
    긍정 또는 부정 라벨을 추가한다.
    """

    rating_df = df[
        df["rating"] == rating
    ].copy()

    available_count = len(rating_df)

    # 추출 가능한 데이터가 부족한 경우 오류 발생
    if available_count < sample_size:
        raise ValueError(
            f"{rating}점 리뷰가 부족합니다.\n"
            f"필요 개수: {sample_size}건\n"
            f"현재 개수: {available_count}건"
        )

    sampled_df = rating_df.sample(
        n=sample_size,
        random_state=random_state
    ).copy()

    # MobileBERT 학습용 숫자 라벨
    # 0: 부정, 1: 긍정
    sampled_df["label"] = label

    # 사람이 확인하기 위한 감성 이름
    sampled_df["sentiment"] = sentiment

    return sampled_df


# ==================================================
# 6. 제품별 데이터 추출 및 라벨링
# ==================================================

all_product_data = []

for product_index, product in enumerate(PRODUCTS):

    print("\n" + "=" * 60)
    print(f"{product['product_name']} 데이터 처리 중")
    print("=" * 60)

    # data 폴더 안에 있는 CSV 파일 경로
    all_file_path = DATA_DIR / product["all_file"]
    negative_file_path = DATA_DIR / product["negative_file"]

    # 전체 리뷰 데이터
    all_df = load_and_clean_csv(all_file_path)

    # 별점 1점과 2점이 들어 있는 부정 리뷰 데이터
    negative_df = load_and_clean_csv(negative_file_path)

    # 필터링 후 별점별 데이터 개수 확인
    print("\n[전체 리뷰 파일의 별점별 개수]")
    print(
        all_df["rating"]
        .value_counts()
        .sort_index()
    )

    print("\n[부정 리뷰 파일의 별점별 개수]")
    print(
        negative_df["rating"]
        .value_counts()
        .sort_index()
    )

    # 제품마다 서로 다른 추출 결과를 만들기 위한 난수값
    product_seed = RANDOM_STATE + product_index

    # --------------------------------------------------
    # 긍정 리뷰
    # 4점 150건 + 5점 150건 = 제품당 긍정 300건
    # --------------------------------------------------

    positive_4 = sample_rating_data(
        df=all_df,
        rating=4,
        sample_size=SAMPLE_PER_RATING,
        label=1,
        sentiment="positive",
        random_state=product_seed
    )

    positive_5 = sample_rating_data(
        df=all_df,
        rating=5,
        sample_size=SAMPLE_PER_RATING,
        label=1,
        sentiment="positive",
        random_state=product_seed + 10
    )

    # --------------------------------------------------
    # 부정 리뷰
    # 1점 150건 + 2점 150건 = 제품당 부정 300건
    # --------------------------------------------------

    negative_1 = sample_rating_data(
        df=negative_df,
        rating=1,
        sample_size=SAMPLE_PER_RATING,
        label=0,
        sentiment="negative",
        random_state=product_seed + 20
    )

    negative_2 = sample_rating_data(
        df=negative_df,
        rating=2,
        sample_size=SAMPLE_PER_RATING,
        label=0,
        sentiment="negative",
        random_state=product_seed + 30
    )

    # 제품별 긍정·부정 리뷰 결합
    product_df = pd.concat(
        [
            negative_1,
            negative_2,
            positive_4,
            positive_5
        ],
        ignore_index=True
    )

    # 제품 정보 추가
    product_df["product_id"] = product["product_id"]
    product_df["product_name"] = product["product_name"]

    # MobileBERT 입력용 text 열 생성
    # 제목과 본문을 하나로 결합
    product_df["text"] = (
        product_df["review_title"]
        + " "
        + product_df["review_text"]
    ).str.strip()

    all_product_data.append(product_df)

    print("\n[제품별 추출 결과]")
    print(
        f"긍정 리뷰: "
        f"{(product_df['label'] == 1).sum():,}건"
    )
    print(
        f"부정 리뷰: "
        f"{(product_df['label'] == 0).sum():,}건"
    )
    print(
        f"제품별 합계: {len(product_df):,}건"
    )


# ==================================================
# 7. 5개 제품 데이터 통합
# ==================================================

final_df = pd.concat(
    all_product_data,
    ignore_index=True
)

# 전체 데이터를 무작위로 섞기
final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# 저장할 열 순서
final_df = final_df[
    [
        "product_id",
        "product_name",
        "rating",
        "review_title",
        "review_text",
        "word_count",
        "text",
        "label",
        "sentiment"
    ]
]


# ==================================================
# 8. 최종 데이터 검증
# ==================================================

total_count = len(final_df)
positive_count = (final_df["label"] == 1).sum()
negative_count = (final_df["label"] == 0).sum()

# 전체 3,000건 확인
if total_count != 3000:
    raise ValueError(
        f"전체 데이터가 3,000건이 아닙니다: "
        f"{total_count:,}건"
    )

# 긍정 1,500건 확인
if positive_count != 1500:
    raise ValueError(
        f"긍정 데이터가 1,500건이 아닙니다: "
        f"{positive_count:,}건"
    )

# 부정 1,500건 확인
if negative_count != 1500:
    raise ValueError(
        f"부정 데이터가 1,500건이 아닙니다: "
        f"{negative_count:,}건"
    )

# 제목과 본문의 NULL 확인
if final_df["review_title"].isna().any():
    raise ValueError(
        "리뷰 제목에 NULL 값이 있습니다."
    )

if final_df["review_text"].isna().any():
    raise ValueError(
        "리뷰 본문에 NULL 값이 있습니다."
    )

# 제목 영어 여부 확인
if not final_df["review_title"].apply(
    is_english_text
).all():
    raise ValueError(
        "영어가 아닌 리뷰 제목이 포함되어 있습니다."
    )

# 본문 영어 여부 확인
if not final_df["review_text"].apply(
    is_english_text
).all():
    raise ValueError(
        "영어가 아닌 리뷰 본문이 포함되어 있습니다."
    )

# 본문 단어 수 확인
if (final_df["word_count"] <= MIN_REVIEW_WORDS).any():
    raise ValueError(
        "10단어 이하의 리뷰 본문이 포함되어 있습니다."
    )

# 중복 본문 확인
if final_df["review_text"].duplicated().any():
    print(
        "\n주의: 서로 다른 제품 사이에 "
        "같은 리뷰 본문이 존재합니다."
    )


# ==================================================
# 9. CSV 파일 저장
# ==================================================

save_path = (
    LABEL_DIR
    / "mobilebert_labeled_3000.csv"
)

final_df.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)


# ==================================================
# 10. 최종 결과 출력
# ==================================================

print("\n" + "=" * 60)
print("학습 데이터 구축 완료")
print("=" * 60)

print(f"저장 경로: {save_path}")
print(f"전체 데이터: {len(final_df):,}건")

print("\n[감성별 데이터 개수]")
print(
    final_df["sentiment"]
    .value_counts()
)

print("\n[라벨별 데이터 개수]")
print(
    final_df["label"]
    .value_counts()
    .sort_index()
)

print("\n[별점별 데이터 개수]")
print(
    final_df["rating"]
    .value_counts()
    .sort_index()
)

print("\n[제품별·감성별 데이터 개수]")
print(
    pd.crosstab(
        final_df["product_name"],
        final_df["sentiment"],
        margins=True
    )
)

print("\n[제품별·별점별 데이터 개수]")
print(
    pd.crosstab(
        final_df["product_name"],
        final_df["rating"],
        margins=True
    )
)

print("\n[리뷰 본문 단어 수 통계]")
print(
    final_df["word_count"]
    .describe()
)

print("\n[저장 데이터 앞부분]")
print(final_df.head())