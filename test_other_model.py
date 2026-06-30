import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def test():
    model_id = "Mavkif/m2m100_rup_rur_to_ur"
    tokenizer_id = "Mavkif/m2m100_rup_tokenizer_both"
    
    print("Loading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_id)
    
    print("Loading model...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # Direction 1: Roman Urdu -> Urdu
    print("\n--- Testing Roman Urdu -> Urdu ---")
    roman_text = "main ne usse message kiya"
    tokenizer.src_lang = "ru"
    encoded_1 = tokenizer(roman_text, return_tensors="pt")
    forced_bos_1 = tokenizer.get_lang_id("ur")
    
    with torch.no_grad():
        generated_1 = model.generate(
            **encoded_1,
            forced_bos_token_id=forced_bos_1,
            num_beams=4,
            max_new_tokens=128
        )
    print(f"Decoded (ru->ur): '{tokenizer.decode(generated_1[0], skip_special_tokens=True)}'")
    
    # Direction 2: Urdu -> Roman Urdu
    print("\n--- Testing Urdu -> Roman Urdu ---")
    urdu_text = "میں نے اسے میسج کیا"
    tokenizer.src_lang = "ur"
    encoded_2 = tokenizer(urdu_text, return_tensors="pt")
    forced_bos_2 = tokenizer.get_lang_id("ru")
    
    with torch.no_grad():
        generated_2 = model.generate(
            **encoded_2,
            forced_bos_token_id=forced_bos_2,
            num_beams=4,
            max_new_tokens=128
        )
    print(f"Decoded (ur->ru): '{tokenizer.decode(generated_2[0], skip_special_tokens=True)}'")

if __name__ == "__main__":
    test()
