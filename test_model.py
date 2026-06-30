import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def test():
    model_id = "Mavkif/m2m100_rup_ur_to_rur"
    tokenizer_id = "Mavkif/m2m100_rup_tokenizer_both"
    
    print("Loading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_id)
    
    print("Loading model...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    
    urdu_text = "میں نے اسے میسج کیا"
    print(f"Input Urdu: {urdu_text}")
    
    # Tokenize input using Urdu
    tokenizer.src_lang = "ur"
    tokenizer.tgt_lang = "ru"
    encoded = tokenizer(urdu_text, return_tensors="pt")
    
    # Force target language to Roman Urdu (which is ru)
    forced_bos_token_id = tokenizer.get_lang_id("ru")
    print(f"Forced BOS Token ID for 'ru': {forced_bos_token_id}")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            num_beams=4,
            max_new_tokens=128
        )
    
    print(f"Generated IDs: {generated[0].tolist()}")
    raw_tokens = tokenizer.convert_ids_to_tokens(generated[0])
    print(f"Raw tokens: {raw_tokens}")
    
    decoded_skip = tokenizer.decode(generated[0], skip_special_tokens=True)
    decoded_noskip = tokenizer.decode(generated[0], skip_special_tokens=False)
    
    print(f"Decoded (skip_special_tokens=True): '{decoded_skip}'")
    print(f"Decoded (skip_special_tokens=False): '{decoded_noskip}'")

if __name__ == "__main__":
    test()
