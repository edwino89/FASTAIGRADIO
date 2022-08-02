from fastai.vision.all import*
import gradio as gr

def is_cutlery(x): return x[0].isupper()

learn = load_learner('cutlery.pkl')

categories = ('forks', "knife",'spoon')

def classify_image_(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ["spoon01.jpg","fork01.jpeg","cleaver01.jpg"]

intf = gr.Interface(fn=classify_image_,inputs=image, outputs=label, example=examples)
intf.launch(inline=False)