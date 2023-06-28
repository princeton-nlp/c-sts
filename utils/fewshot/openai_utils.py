import openai
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


LEGACY_MODELS = {
    'davinci',
    'curie',
    'babbage',
    'ada',
}

GPT3_MODELS = {
    'text-davinci-003',
    'text-davinci-002',
    'text-davinci-001',
    'text-curie-001',
    'text-babbage-001',
    'text-ada-001',
}

CHAT_MODELS = {
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-32k',
    'gpt-4-32k-0314',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301',
}

OPENAI_MODELS = LEGACY_MODELS | GPT3_MODELS | CHAT_MODELS


def parse_response(model, response, prompt):
    if model in CHAT_MODELS:
        response_text = response['choices'][0]['message']['content']
    else:
        response_text = response['choices'][0]['text'].replace(prompt, '')
    response_text.strip()
    return response_text

def call_chat(model, prompt):
    response = openai.ChatCompletion.create(
                model=model,
                messages= [{'role': 'user', 'content': prompt}, ],
                temperature=0,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                )
    return response


def call_gpt(model, prompt):
    response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=0,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                )
    return response


def get_gpt_prediction(model, prompt):
    retries = 0
    while retries < 3:
        try:
            if model in CHAT_MODELS:
                response = call_chat(model, prompt)
            else:
                response = call_gpt(model, prompt)
            return parse_response(model, response, prompt)
        except Exception as e:
            logger.warning('Exception while getting gpt prediction: {}'.format(e))
            logger.warning(f'Retrying... {3 - retries} more times.')
            retries += 1
            time.sleep(20 * retries)
    raise Exception('Failed to get gpt prediction after 3 retries. Aborting run.')


def authenticate(api_key):
    with open(api_key) as f:
        api_key = f.readlines()[0].strip()
    openai.api_key = api_key
