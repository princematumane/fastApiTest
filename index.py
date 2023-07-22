import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from langcorn import create_service

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
MODEL = tf.keras.models.load_model("./models/2")
CLASS_NAMES = ["Apple1stClass", "Apple2ndClass", "Apple3rdClass", "Pepper1stClass", "Pepper2ndClass",
               "Pepper3rdClass", "Tomato1stClass", "Tomato2ndClass", "Tomato3rdClass"]
class_counts = {label: 0 for label in CLASS_NAMES}
multipliers = [5, 3.5, 2, 5, 3.5, 2, 5, 3.5, 2]


###########################################################################


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    if predicted_class in class_counts and confidence > 0.99:
        class_counts[predicted_class] += 1
    else:
        class_counts[predicted_class] = 0

    first_class = ["Apple1stClass", "Pepper1stClass", "Tomato1stClass"]
    second_class = ["Apple2ndClass", "Pepper2ndClass", "Tomato2ndClass"]
    third_class = ["Apple3rdClass", "Pepper3rdClass", "Tomato3rdClass"]

    if predicted_class in first_class and confidence > 0.99:
        result = f"This product is first class and is going locally in South Africa"

    elif predicted_class in second_class and confidence > 0.99:
        result = f"This product is second class and is going  to our neighbor countries within Africa"

    elif predicted_class in third_class and confidence > 0.99:
        result = f"This product is third class are going out of Africa to supply other continent"
    else:
        predicted_class ="N/A"
        result = "This product is not yet in our AI model, we are working hard to improve it"

    confi = round(float(confidence * 100))

    return {
        'class': predicted_class,
        'confidence': confi,
        'message': result,
    }


@app.post("/send_email")
async def send_email(image: UploadFile = File(...), product: str = Body(...)):
    # Process the image
    image_bytes = await image.read()
    # Create a multipart message
    product_name = product.upper()
    msg = MIMEMultipart()
    msg["Subject"] = f"New email"
    msg["From"] = "thabangmabena12@gmail.com"
    msg["To"] = "thabangmabena12@gmail.com"

    # Attach the message as plain text
    text = MIMEText(
        f"The name of the product I sent is : {product_name}")
    msg.attach(text)

    # Attach the image
    image_mime = MIMEImage(image_bytes)
    image_mime.add_header("Content-Disposition", "attachment", filename=image.filename)
    msg.attach(image_mime)

    # Connect to the SMTP server and send the email
    with smtplib.SMTP("smtp.elasticemail.com", port=2525) as server:
        server.starttls()
        #server.login("eliasmabena12@gmail.com", "6709B7D804D725185D955C792745F1F9F847")
        server.login("thabangmabena12@gmail.com", "E76036768594F0F16FCCA089AFD6218B480E")
        server.send_message(msg)
    return {"message": "Email sent successfully!"}

@app.get("/")
async def main():
      return {"message": "Running..."}

@app.get("/display_counts")
async def display_class_counts():
    labels = list(class_counts.keys())
    counts = list(class_counts.values())
    colors =['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

    plt.bar(labels, counts, color=colors)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Counts')
    plt.xticks(rotation=45, ha='right')
    plt.show()

    mult_count = []
    multiplier_counts = {}
    for label, count, multiplier in zip(labels, counts, multipliers):
        if multiplier not in multiplier_counts:
            multiplier_counts[multiplier] = count
        else:
            multiplier_counts[multiplier] += count

    # Print the sum of counts for each multiplier
    for multiplier, count in multiplier_counts.items():
        prices = float(multiplier * count)
        mult_count.append({"Multiplier": multiplier, "Count": count, "Price": prices})

    return {"multiplier_counts": mult_count,
            "Labels": labels,
            "Counts": counts,
            }


pass
if __name__ == "__main__":
    print("Hello world main")
    uvicorn.run(app, host='localhost', port=8000)
