import argparse
import re
import json
import requests
from pathlib import Path


def send_post_request(email, predictions, filename):
    # Prepare the data to be sent
    if len(filename) > 200:
        raise ValueError('Submission name (%s) longer than 200 characters. Please choose a shorter filename or set the name with --name' % filename)
    data = {
        'email': email,
        'predictions': predictions,
        'filename': filename,
    }
    data_str = json.dumps({'body': json.dumps(data)})
    headers = {'content-type': 'application/json'}
    # url = 'https://rcxnewlbk5.execute-api.us-east-2.amazonaws.com/test/eval-csts'
    url = "https://0sy74d2tog.execute-api.us-east-2.amazonaws.com/dev/c-sts-eval-lambda"
    # Create the request object
    request_object = {
        "url": url,
        "headers": headers,
        "data": data
    }
    json.dump(request_object, open('request.json', 'w'), indent=4)
    response = requests.post(url, headers=headers, data=data_str)    
    if response.status_code == 200:
        print("Evaluation successful!")
        print(response.json()['body'])
        print("See email: \"C-STS Evaluation Results for %s\"" % filename)


def main(email, predictions_file, name):
    predictions_file = Path(predictions_file).resolve(strict=True)
    if name is None:
        name = predictions_file.as_posix()
    if not re.match(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+(?:[A-Za-z0-9.-]+)*\b", email):
        raise ValueError("Email %s is invalid" % email)
    with open(predictions_file, 'r') as f:
        preds = json.load(f)
    keys, preds = zip(*sorted(preds.items(), key=lambda x: int(x[0])))
    preds = list(map(float, preds))
    if len(keys) != 4732:
        raise ValueError("There should be exactly 4732 predictions, but got %d instead" % len(keys))
    send_post_request(email, preds, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send email and predictions to server')
    parser.add_argument('email', type=str, help='The email to be sent')
    parser.add_argument('predictions_file', type=str, help='The path to the JSON file containing the predictions')
    parser.add_argument('--name', type=str, help='The name of the submission. Uses the filename if not specified')
    args = parser.parse_args()
    main(**vars(args))
