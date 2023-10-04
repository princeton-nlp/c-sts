import argparse
import json
import requests

def send_post_request(email, predictions, filename):
    # Prepare the data to be sent
    data = {
        'email': email,
        'predictions': predictions,
        'filename': filename,
    }
    data_str = json.dumps({'body': json.dumps(data)})
    headers = {'content-type': 'application/json'}

    url = 'https://rcxnewlbk5.execute-api.us-east-2.amazonaws.com/test/eval-csts'
    
    response = requests.post(url, headers=headers, data=data_str)
    
    print(response.text)


def main(email, predictions_file):
    with open(predictions_file, 'r') as f:
        preds = json.load(f)

    keys, preds = zip(*sorted(preds.items(), key=lambda x: int(x[0])))
    
    assert len(keys) == 4732, f"There should be exactly 4732 predictions, but got {len(keys)}"
    
    send_post_request(email, preds, predictions_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send email and predictions to server')
    parser.add_argument('email', type=str, help='The email to be sent')
    parser.add_argument('predictions_file', type=str, help='The path to the JSON file containing the predictions')
    args = parser.parse_args()
    main(**vars(args))
