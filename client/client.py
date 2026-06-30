import argparse
from typing import List

import httpx


DEFAULT_URL = "http://localhost:8000"
DEFAULT_BEAMS = 4
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 16
DEFAULT_TIMEOUT = 300.0


TEST_PAIRS = [
    ("میں نے اسے میسج کیا", "main ne usse message kiya"),
    ("وہ روز ایکسرسائز کرتا ہے", "woh roz exercise karta hai"),
    ("کل ایگزام ہے میرا", "kal exam hai mera"),
    ("وہ میتھ میں فیل ہو گیا", "woh math mein fail ho gaya"),
    ("اس نے پورا پروجیکٹ ڈیلیٹ کر دیا", "usne poora project delete kar diya"),
    ("بیک اپ لینا ضروری ہے", "backup lena zaroori hai"),
    ("یہ فائل کرپٹ ہو گئی", "yeh file corrupt ho gayi"),
    ("مجھے کل انٹرویو دینا ہے", "mujhe kal interview dena hai"),
    ("اس کا لیپ ٹاپ کریش ہو گیا", "uska laptop crash ho gaya"),
    ("میں نے سسٹم ری اسٹارٹ کیا", "maine system restart kiya"),
    ("وہ آن لائن کلاس میں تھا", "woh online class mein tha"),
    ("ایپ اپڈیٹ کر لو", "app update kar lo"),
    ("سرور ڈاؤن ہے ابھی", "server down hai abhi"),
    ("پاس ورڈ چینج کرو اپنا", "password change karo apna"),
    ("اسکرین شاٹ لے لو اس کا", "screenshot le lo iska"),
    ("وہ گھر پر نہیں ہے", "woh ghar par nahi hai"),
    ("میں کل آؤں گا", "main kal aaunga"),
    ("اس نے مجھے بلایا تھا", "usne mujhe bulaya tha"),
    ("کھانا تیار ہو گیا", "khana tayar ho gaya"),
    ("بارش ہو رہی ہے باہر", "baarish ho rahi hai bahar"),
    ("مجھے نیند آ رہی ہے", "mujhe neend aa rahi hai"),
    ("وہ بہت تھک گیا ہے", "woh bahut thak gaya hai"),
    ("یہ کام کل تک ہو جائے گا", "yeh kaam kal tak ho jayega"),
    ("اس نے سچ نہیں بولا", "usne sach nahi bola"),
    ("میں صبح اٹھ کر چلا گیا", "main subah uth kar chala gaya"),
] 

def transliterate(text: str, base_url: str = DEFAULT_URL, beams: int = DEFAULT_BEAMS) -> str:
    response = httpx.post(
        f"{base_url.rstrip('/')}/translate",
        json={"text": text, "num_beams": beams},
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["translation"]


def transliterate_batch(
    texts: List[str],
    base_url: str = DEFAULT_URL,
    beams: int = DEFAULT_BEAMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
) -> List[str]:
    if not texts:
        return []

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if batch_size > MAX_BATCH_SIZE:
        batch_size = MAX_BATCH_SIZE

    translations: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = httpx.post(
            f"{base_url.rstrip('/')}/translate",
            json={"texts": batch, "num_beams": beams},
            timeout=timeout,
        )
        response.raise_for_status()
        translations.extend(response.json()["translations"])

    return translations


def run_test_pairs(
    base_url: str,
    beams: int = DEFAULT_BEAMS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    print("Sending Urdu→Roman test pairs to server:\n")
    texts = [urdu for urdu, _ in TEST_PAIRS]
    predictions = transliterate_batch(
        texts,
        base_url=base_url,
        beams=beams,
        batch_size=batch_size,
        timeout=timeout,
    )

    for (urdu, reference), prediction in zip(TEST_PAIRS, predictions):
        print(f"Input Urdu: {urdu}")
        print(f"Pred Roman: {prediction}")
        print(f"Ref Roman : {reference}")
        print("---")

    print(f"\nCompleted {len(predictions)} test pairs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Urdu → Roman Urdu transliteration client")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument("--beams", type=int, default=DEFAULT_BEAMS, help="Beam search width")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for /translate requests (max {MAX_BATCH_SIZE})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds for each batch request",
    )
    parser.add_argument(
        "--text",
        help="Translate a single sentence instead of running the built-in test pairs",
    )
    args = parser.parse_args()

    if args.text:
        translation = transliterate(
            args.text,
            base_url=args.url,
            beams=args.beams,
        )
        print(translation)
    else:
        run_test_pairs(
            args.url,
            beams=args.beams,
            batch_size=args.batch_size,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()
