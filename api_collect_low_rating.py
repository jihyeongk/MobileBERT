import requests
import pandas as pd
import time
import os
import random
from tqdm import tqdm


# =========================
# 1. 기본 설정
# =========================

PRODUCT_ID = 16567

# 1점, 2점 리뷰만 수집
RATINGS = [1, 2]

BASE_URL = "https://www.iherb.com/ugc/api/review/v1/search"

PARAMS = {
    "ratings": RATINGS,
    "pid": PRODUCT_ID,
    "lc": "en-US",
    "textToSearch": "",
    "sortId": 6,
    "withImagesOnly": "false",
    "isShowTranslated": "true",
    "withUgcSummary": "true",
    "limit": 20,
    "withoutDefaultTitle": "true"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://www.iherb.com/"
}

# 기존 전체 리뷰 파일과 다른 이름으로 저장
OUTPUT_FILE = f"iherb_reviews_{PRODUCT_ID}_rating_1_2.csv"

MAX_PAGES = 300
SAVE_EVERY = 20


# =========================
# 2. 텍스트 정리 함수
# =========================

def clean_text(text):
    if text is None:
        return ""

    return " ".join(
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .split()
    )


# =========================
# 3. 리뷰 파싱 함수
# =========================

def parse_reviews(data, page):
    result = []

    for item in data.get("items", []):
        review_title = item.get("reviewTitle")
        review_text = item.get("reviewText")
        rating_raw = item.get("ratingValue")

        if rating_raw is None:
            continue

        rating = rating_raw / 10

        # 혹시 다른 별점이 들어오면 제외
        if rating not in RATINGS:
            continue

        result.append({
            "page": page,
            "review_title": clean_text(review_title),
            "review_text": clean_text(review_text),
            "rating": rating
        })

    return result


# =========================
# 4. 기존 파일 있으면 이어받기
# =========================

if os.path.exists(OUTPUT_FILE):
    df_existing = pd.read_csv(OUTPUT_FILE)
    all_reviews = df_existing.to_dict("records")

    if "page" in df_existing.columns and len(df_existing) > 0:
        start_page = int(df_existing["page"].max()) + 1
    else:
        start_page = 1

    print(f"기존 파일 발견: {len(df_existing)}개")
    print(f"{start_page}페이지부터 이어서 수집합니다.")

else:
    all_reviews = []
    start_page = 1
    print("기존 파일 없음. 1페이지부터 수집합니다.")


# =========================
# 5. 크롤링 시작
# =========================

for page in tqdm(range(start_page, MAX_PAGES + 1)):
    params = PARAMS.copy()
    params["page"] = page

    try:
        response = requests.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=20
        )

        if response.status_code == 400:
            print(f"{page}페이지에서 400 오류 발생. 수집을 종료합니다.")
            break

        if response.status_code == 429:
            wait_time = random.uniform(30, 90)
            print(
                f"429 Too Many Requests 발생. "
                f"{wait_time:.1f}초 대기 후 종료합니다."
            )
            time.sleep(wait_time)
            break

        if response.status_code in [403, 503]:
            wait_time = random.uniform(60, 180)
            print(
                f"차단 가능성 있는 응답 {response.status_code}. "
                f"{wait_time:.1f}초 대기 후 종료합니다."
            )
            time.sleep(wait_time)
            break

        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])

        if not items:
            print(f"{page}페이지에서 리뷰가 없어 종료합니다.")
            break

        reviews = parse_reviews(data, page)
        all_reviews.extend(reviews)

        print(
            f"{page}페이지 수집 완료 / "
            f"1·2점 리뷰 누적 {len(all_reviews)}개"
        )

        # 중간 저장
        if page % SAVE_EVERY == 0:
            df = pd.DataFrame(all_reviews)

            df = df.dropna(subset=["review_text"])
            df = df[df["review_text"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["review_text"])
            df = df[
                ["page", "review_title", "review_text", "rating"]
            ]

            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig"
            )

            print(f"중간 저장 완료: {OUTPUT_FILE}")

        time.sleep(random.uniform(1.5, 3.5))

    except requests.exceptions.RequestException as e:
        print(f"{page}페이지 요청 오류:", e)

        df = pd.DataFrame(all_reviews)

        if not df.empty:
            df = df.dropna(subset=["review_text"])
            df = df[df["review_text"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["review_text"])
            df = df[
                ["page", "review_title", "review_text", "rating"]
            ]

            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig"
            )

        print("오류 발생 전까지 저장 완료")
        break

    except KeyboardInterrupt:
        print("\n실행이 중단되었습니다.")

        df = pd.DataFrame(all_reviews)

        if not df.empty:
            df = df.dropna(subset=["review_text"])
            df = df[df["review_text"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["review_text"])
            df = df[
                ["page", "review_title", "review_text", "rating"]
            ]

            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig"
            )

        print("중단 전까지 저장 완료")
        break

    except Exception as e:
        print(f"{page}페이지 기타 오류:", e)

        df = pd.DataFrame(all_reviews)

        if not df.empty:
            df = df.dropna(subset=["review_text"])
            df = df[df["review_text"].astype(str).str.strip() != ""]
            df = df.drop_duplicates(subset=["review_text"])
            df = df[
                ["page", "review_title", "review_text", "rating"]
            ]

            df.to_csv(
                OUTPUT_FILE,
                index=False,
                encoding="utf-8-sig"
            )

        print("오류 발생 전까지 저장 완료")
        break


# =========================
# 6. 최종 저장
# =========================

df = pd.DataFrame(all_reviews)

if not df.empty:
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["review_text"])
    df = df[
        ["page", "review_title", "review_text", "rating"]
    ]

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

print("최종 저장 완료")
print("저장 파일:", OUTPUT_FILE)
print("총 리뷰 수:", len(df))

if not df.empty:
    print("\n별점 분포")
    print(df["rating"].value_counts().sort_index())
    print(df.head())