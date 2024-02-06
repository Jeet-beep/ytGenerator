from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import gunicorn
from flask_cors import CORS
from langchain_community.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_openai import ChatOpenAI

app = Flask(__name__)
CORS(app)
load_dotenv()

OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=1.0, model_name="gpt-3.5-turbo")

title_template = PromptTemplate(
    input_variables=["keyword"],
    template="write a creative youtube video title about {keyword}"
)

desc_template = PromptTemplate(
    input_variables=["title"],
    template="based on the {title} of the youtube video, generate a youtube description in almost 150 words and also generate hashtags"
)

title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title")
desc_chain = LLMChain(llm=llm, prompt=desc_template, output_key="description")

sequential_chain = SequentialChain(chains=[title_chain, desc_chain], input_variables=["keyword"],
                                    output_variables=["title", "description"], verbose=True)

@app.route('/generate_title_and_description', methods=['POST'])
def generate_title_and_description():
    data = request.get_json()
    keyword = data.get('keyword')
    if keyword:
        response = sequential_chain({"keyword": keyword})
        return jsonify(response)
    else:
        return jsonify({'error': 'Keyword not provided'})

if __name__ == '__main__':
    app.run(debug=True)
