import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

def test_reverse():
    model_id = "Mavkif/m2m100_rup_ur_to_rur"
    tokenizer_id = "Mavkif/m2m100_rup_tokenizer_both"
    
    print("Loading tokenizer...")
    tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_id)
    
    print("Loading model...")
    model = M2M100ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32)
    
    roman_text = "main ne usse message kiya"
    print(f"Input Roman: {roman_text}")
    
    # Tokenize input using 'ru' (repurposed for Roman Urdu)
    tokenizer.src_lang = "ru"
    encoded = tokenizer(roman_text, return_tensors="pt")
    
    # Force target language to Urdu ('ur')
    forced_bos_token_id = tokenizer.get_lang_id("ur")
    print(f"Forced BOS Token ID for 'ur': {forced_bos_token_id}")
    
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
    test_reverse()
