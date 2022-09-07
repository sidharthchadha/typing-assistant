import json
from flask import Flask, render_template, redirect, request, flash, session, jsonify


from flask import Flask 
from turbo_flask import Turbo

from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
turbo = Turbo(app)
app.debug = True
debug = True

from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""
  
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
  print(predicted_word)
  return predicted_word

@app.route('/')
def index():
    """Homepage."""

    return render_template("index.html")




@app.route("/update", methods=["POST"])
def upvote():
    words=request.values.get('words')
    # print(words)

    text = words
    
    try:
        text = text.split(" ")
        text = text[-3:]
        print(text)
          
        return Predict_Next_Words(model, tokenizer, text)
          
    except Exception as e:
        print("Error occurred: ",e)
        return []


    return str(" ")




@app.route("/error")
def error():
    raise Exception("Error!")


if __name__ == "__main__":
    # Set debug=True here to invoke the DebugToolbarExtension
    app.run(debug=True,use_reloader=False)
   
    



# #  code to build model: commented out 
# file = open("/content/drive/MyDrive/book (2).txt", "r", encoding = "utf8")
# lines = []
# for i in file:
#     lines.append(i)

# data = ""
# for i in lines:
#   data = ' '. join(lines) 

# data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  #new line, carriage return, unicode character --> replace by space

# data = data.split()
# data = ' '.join(data)
# data[:500]
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([data])


# pickle.dump(tokenizer, open('/content/drive/MyDrive/token.pkl', 'wb'))

# sequence_data = tokenizer.texts_to_sequences([data])[0]
# sequence_data[:15]
# len(sequence_data)

# model = Sequential()
# model.add(Embedding(vocab_size, 10, input_length=3))
# model.add(LSTM(1000, return_sequences=True))
# model.add(LSTM(1000))
# model.add(Dense(1000, activation="relu"))
# model.add(Dense(vocab_size, activation="softmax"))
# model.summary()

# from tensorflow import keras
# from keras.utils.vis_utils import plot_model

# keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)
# from tensorflow.keras.callbacks import ModelCheckpoint

# checkpoint = ModelCheckpoint("/content/drive/MyDrive/next_words.h5", monitor='loss', verbose=1, save_best_only=True)
# model.compile(loss="categorical_crossentropy")
# model.fit(X, y, epochs=30, batch_size=64, callbacks=[checkpoint])