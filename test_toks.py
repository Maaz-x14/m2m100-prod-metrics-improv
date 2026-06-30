from transformers import M2M100Tokenizer

def compare():
    t_base = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    t_ft = M2M100Tokenizer.from_pretrained("Mavkif/m2m100_rup_tokenizer_both")
    
    text = "میں نے اسے میسج کیا"
    print("Base tokens:", t_base.tokenize(text))
    print("FT tokens  :", t_ft.tokenize(text))
    
    word = "میسج"
    print("Base word tokens:", t_base.tokenize(word))
    print("FT word tokens  :", t_ft.tokenize(word))

if __name__ == "__main__":
    compare()
