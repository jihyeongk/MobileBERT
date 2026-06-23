import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import MobileBertForSequenceClassification, MobileBertTokenizer
from tqdm import tqdm


def main():
    # 0. GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용하는 장치:", device)

    # 1. 파일 경로
    data_path = "data/iherb_reviews_16567_mg.csv"
    model_path = "mobilebert_imdb.pt"
    save_path = "label/iherb_reviews_16567_mg_prediction.csv"

    # 2. 원본 데이터 불러오기
    df = pd.read_csv(data_path, encoding="utf-8-sig")

    # 제목과 본문이 모두 없는 행 제거
    df = df.dropna(
        subset=["review_text"]
    ).copy()

    # 제목의 NULL 값은 빈 문자열로 변경
    df["review_title"] = df["review_title"].fillna("")

    # 문자열 변환 및 공백 제거
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

    # 빈 본문 제거
    df = df[
        df["review_text"] != ""
    ].copy()

    # 학습 데이터와 같은 방식으로 제목과 본문 결합
    df["text"] = (
        df["review_title"]
        + " "
        + df["review_text"]
    ).str.strip()

    text = df["text"].tolist()

    print("예측할 리뷰 개수:", len(text))
    print("첫 번째 리뷰:", text[0])

    # 3. 토크나이저 불러오기
    tokenizer = MobileBertTokenizer.from_pretrained(
        "google/mobilebert-uncased"
    )

    # 4. 토큰화
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
        padding="max_length"
    )

    input_ids = torch.tensor(inputs["input_ids"])
    attention_mask = torch.tensor(inputs["attention_mask"])

    # 5. DataLoader 생성
    dataset = TensorDataset(
        input_ids,
        attention_mask
    )

    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=8
    )

    # 6. 학습된 모델 불러오기
    model = MobileBertForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(device)
    model.eval()

    # 7. 예측
    predictions = []
    probabilities = []

    progress_bar = tqdm(
        dataloader,
        desc="리뷰 예측 중"
    )

    with torch.inference_mode():

        for batch in progress_bar:
            batch_ids, batch_masks = (
                tensor.to(device)
                for tensor in batch
            )

            outputs = model(
                batch_ids,
                attention_mask=batch_masks
            )

            logits = outputs.logits

            # 긍정·부정 확률 계산
            probs = torch.softmax(
                logits,
                dim=1
            )

            # 확률이 더 높은 라벨 선택
            preds = torch.argmax(
                probs,
                dim=1
            )

            # 선택된 라벨의 확률
            confidence = torch.max(
                probs,
                dim=1
            ).values

            predictions.extend(
                preds.cpu().numpy()
            )

            probabilities.extend(
                confidence.cpu().numpy()
            )

    # 8. 결과 열 추가
    df["predicted_label"] = predictions

    df["predicted_sentiment"] = df[
        "predicted_label"
    ].map({
        0: "negative",
        1: "positive"
    })

    df["prediction_probability"] = probabilities

    # 9. 결과 저장
    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 10. 결과 확인
    print("\n=== 예측 완료 ===")
    print("저장 위치:", save_path)

    print("\n=== 예측 결과 개수 ===")
    print(
        df["predicted_sentiment"]
        .value_counts()
    )

    print("\n=== 예측 결과 비율 ===")
    print(
        df["predicted_sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\n=== 예측 결과 예시 ===")
    print(
        df[
            [
                "review_text",
                "predicted_sentiment",
                "prediction_probability"
            ]
        ].head()
    )


if __name__ == "__main__":
    main()