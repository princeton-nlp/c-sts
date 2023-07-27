import argparse
import json
import subprocess

def send_post_request(email, predictions, filename):
    # Prepare the data to be sent
    data = {
        'email': email,
        'predictions': predictions,
        'filename': filename,
    }
    data_str = json.dumps(data)
    command = [
        'curl', '-v', '-X', 'POST',
        'https://rcxnewlbk5.execute-api.us-east-2.amazonaws.com/test/eval-csts',
        '-H', 'content-type: application/json',
        '-d', data_str
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))


def main(email, predictions_file):
    preds = json.load(open(predictions_file, 'r'))
    keys, preds = zip(*sorted(preds.items(), key=lambda x: int(x[0])))
    assert len(keys) == 4732, "There should be exactly 4732 predictions, but got {}".format(len(keys))
    send_post_request(email, preds, predictions_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send email and predictions to server')
    parser.add_argument('email', type=str, help='The email to be sent')
    parser.add_argument('predictions_file', type=str, help='The path to the JSON file containing the predictions')
    args = parser.parse_args()
    main(**vars(args))
