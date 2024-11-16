import predict
import time
import random
from flask import Flask, request, send_file, make_response, url_for, render_template
#import Main_Code
#s=Main_Code.initialize()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload')
def upload():
    return render_template("index1.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
    form = request.form
    img = request.files['in_img']
    i_path = "static\\image.jpg"
    img.save(i_path)
    t1 = time.time()
    #res_lis = s.Main_Code().split("@@@")
    #print(res_lis)
    
    #patho = "Pathogen: "+res_lis[0]
    #nspots = "Number of infected Spots: "+res_lis[1]
    #heal_per = "Leaf Health Based on Fuzzy Logic: "+res_lis[2]
    res = list(predict.process())
    res[0]=res[0].replace("_", " ")
    if "healthy" in res[0]:
        res[0] = "No Disease Found, Identified Leaf: "+res[0]
        res[0]=res[0].replace("healthy", "")
    else:
        res[0] = "Predicted Disease: \""+res[0]+"\""
    #res[0] = "1" #Markup(res[0])
    return render_template("result_withoutMatlab.html", disease=res[0], time=time.time()-t1)#, pathogen=patho, n_spots=nspots, health_percent=heal_per)

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5001)