import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify, make_response
from functions_only_save import make_face_df_save, find_face_shape
from recommender import process_rec_pics, run_recommender_face_shape


app = Flask(__name__, static_url_path="")

df = pd.DataFrame(columns = ['0','1','2','3','4','5','6','7','8','9','10','11',	'12',	'13',	'14',	'15',	'16','17',
                             '18',	'19',	'20',	'21',	'22',	'23',	'24','25',	'26',	'27',	'28',	'29',
                             '30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',
                             '42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',
                             '54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',
                             '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',
                             '78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',
                             '90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',
                             '102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',
                             '114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',
                             '126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',
                             '138',	'139',	'140',	'141',	'142',	'143','A1','A2','A3','A4','A5','A6','A7','A8','A9',
                            'A10','A11','A12','A13','A14','A15','A16','Width','Height','H_W_Ratio','Jaw_width','J_F_Ratio',
                             'MJ_width','MJ_J_width'])

@app.route('/')
def index():
    """Return the main page."""
    return render_template('theme.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    test_photo = 'data/pics/recommendation_pics/' + data['file_name']
    file_num = 2035
    style_df = pd.DataFrame()
    style_df = pd.DataFrame(columns = ['face_shape','hair_length','location','filename','score'])
    hair_length_input = 'Updo'
    updo_input = data['person_see_up_dos']
    if updo_input in ['n','no','N','No','NO']:
        hair_length_input = data['person_hair_length']
        if hair_length_input in ['short','Short','s','S']:
                hair_length_input = 'Short'
        if hair_length_input in ['long','longer','l','L']:
                hair_length_input = 'Long'
    
    make_face_df_save(test_photo,file_num,df)
    face_shape = find_face_shape(df,file_num)
    process_rec_pics(style_df)
    img_filename = run_recommender_face_shape(face_shape[0],style_df,hair_length_input)
    return jsonify({'Face Shape': face_shape[0], 'img_filename': img_filename})

@app.route('/predict_user_face_shape', methods=['GET', 'POST'])
def predict_user_face_shape():
    """Return a user face shape."""
    data = request.json
    test_photo = 'data/pics/recommendation_pics/' + data['file_name']
    file_num = 2035
    
    make_face_df_save(test_photo,file_num,df)
    face_shape = find_face_shape(df,file_num)
    return jsonify({'face_shape': face_shape[0]})

@app.route('/output/<img_filename>')
def output_image(img_filename):
    """Send the output image."""
    with open(f"output/{img_filename}", 'rb') as f:
        img_data = f.read()
    response = make_response(img_data)
    response.headers['Content-Type'] = 'image/png'
    return response
        
        