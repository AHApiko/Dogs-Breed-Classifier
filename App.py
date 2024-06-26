
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import io
from PIL import Image 

app = Flask(__name__, template_folder='templates')

# Load your trained model
model_path = 'C:/Users/HP/Desktop/PY/Dogs project/my_model'
model = tf.keras.models.load_model(model_path)

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Define a route to render the HTML form
@app.route('/')
def index():
    templates_path = os.path.abspath('templates')
    print(templates_path)
    return render_template('index.txt')

# Define a route to handle image classification
@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file
    image_file = request.files['image']

# Read the file content
    image_content = image_file.read()

    # Convert to a PIL Image
    pil_image = Image.open(io.BytesIO(image_content))

   # Preprocess the image
    pil_image = pil_image.resize((224, 224))  # Resize the image
    image = np.array(pil_image)  # Convert PIL Image to numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.resnet.preprocess_input(image)


    # Perform classification
    prediction = model.predict(image)

    # Get the predicted breed
    predicted_class = np.argmax(prediction)
    class_names = ['n02085620-chihuahua', 'n02085782-japanese_spaniel', 'n02085936-maltese_dog', 'n02086079-pekinese', 'n02086240-shih-tzu', 'n02086646-blenheim_spaniel', 'n02086910-papillon', 
                   'n02087046-toy_terrier', 'n02087394-rhodesian_ridgeback', 'n02088094-afghan_hound', 'n02088238-basset', 'n02088364-beagle', 'n02088466-bloodhound', 'n02088632-bluetick', 
                   'n02089078-black-and-tan_coonhound', 'n02089867-walker_hound', 'n02089973-english_foxhound', 'n02090379-redbone', 'n02090622-borzoi', 'n02090721-irish_wolfhound', 
                   'n02091032-italian_greyhound', 'n02091134-whippet', 'n02091244-ibizan_hound', 'n02091467-norwegian_elkhound', 'n02091635-otterhound', 'n02091831-saluki', 'n02092002-scottish_deerhound', 
                   'n02092339-weimaraner', 'n02093256-staffordshire_bullterrier', 'n02093428-american_staffordshire_terrier', 'n02093647-bedlington_terrier', 'n02093754-border_terrier', 'n02093859-kerry_blue_terrier', 
                   'n02093991-irish_terrier', 'n02094114-norfolk_terrier', 'n02094258-norwich_terrier', 'n02094433-yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 'n02095570-lakeland_terrier', 
                   'n02095889-sealyham_terrier', 'n02096051-airedale', 'n02096177-cairn', 'n02096294-australian_terrier', 'n02096437-dandie_dinmont', 'n02096585-boston_bull', 'n02097047-miniature_schnauzer', 
                   'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer', 'n02097298-scotch_terrier', 'n02097474-tibetan_terrier', 'n02097658-silky_terrier', 'n02098105-soft-coated_wheaten_terrier', 
                   'n02098286-west_highland_white_terrier', 'n02098413-lhasa', 'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 'n02099601-golden_retriever', 'n02099712-labrador_retriever', 
                   'n02099849-chesapeake_bay_retriever', 'n02100236-german_short-haired_pointer', 'n02100583-vizsla', 'n02100735-english_setter', 'n02100877-irish_setter', 'n02101006-gordon_setter', 'n02101388-brittany_spaniel', 
                   'n02101556-clumber', 'n02102040-english_springer', 'n02102177-welsh_springer_spaniel', 'n02102318-cocker_spaniel', 'n02102480-sussex_spaniel', 'n02102973-irish_water_spaniel', 'n02104029-kuvasz', 
                   'n02104365-schipperke', 'n02105056-groenendael', 'n02105162-malinois', 'n02105251-briard', 'n02105412-kelpie', 'n02105505-komondor', 'n02105641-old_english_sheepdog', 'n02105855-shetland_sheepdog', 
                   'n02106030-collie', 'n02106166-border_collie', 'n02106382-bouvier_des_flandres', 'n02106550-rottweiler', 'n02106662-german_shepherd', 'n02107142-doberman', 'n02107312-miniature_pinscher', 
                   'n02107574-greater_swiss_mountain_dog', 'n02107683-bernese_mountain_dog', 'n02107908-appenzeller', 'n02108000-entlebucher', 'n02108089-boxer', 'n02108422-bull_mastiff', 'n02108551-tibetan_mastiff', 
                   'n02108915-french_bulldog', 'n02109047-great_dane', 'n02109525-saint_bernard', 'n02109961-eskimo_dog', 'n02110063-malamute', 'n02110185-siberian_husky', 'n02110627-affenpinscher', 'n02110806-basenji', 
                   'n02110958-pug', 'n02111129-leonberg', 'n02111277-newfoundland', 'n02111500-great_pyrenees', 'n02111889-samoyed', 'n02112018-pomeranian', 'n02112137-chow', 'n02112350-keeshond', 'n02112706-brabancon_griffon', 
                   'n02113023-pembroke', 'n02113186-cardigan', 'n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle', 'n02113978-mexican_hairless', 'n02115641-dingo', 'n02115913-dhole', 
                   'n02116738-african_hunting_dog']
    predicted_class_name = class_names[predicted_class]

    # Render the result template with the predicted breed
    return render_template('result.txt', breed=predicted_class_name)

if __name__ == '__main__':
    app.run(debug=True)
