from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/AraT5-base-title-generation")  
model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/AraT5-base-title-generation")
from flask import Flask, render_template,request,flash
app =Flask(__name__ )
app.secret_key = "super secret key"
@app.route("/Title")
def indexx():
    flash("Write The Arabic Text: ")
    flash(" ")
    return render_template ("index.html")



@app.route("/titles", methods=["POST", "GET"])
def generate():
    x=request.form['text_input']
    encoding = tokenizer.encode_plus(str(x),pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_masks,max_length=256,do_sample=True,top_k=120,top_p=0.95,early_stopping=True,num_return_sequences=1)     
    for id, output in enumerate(outputs):
        title = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    flash("The Title Generated:  "+title)
    flash("The Text: "+  str(x))
    return render_template ("index.html")



if __name__ == "__main__":
     app.run(debug=True)